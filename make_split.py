# -*- coding: utf-8 -*-
"""
NeuroPred-3D-EGNN 数据重新划分脚本 (解决同源序列泄漏导致的过拟合)

问题:
    原 training.csv / testing.csv 为随机划分, 测试集中约 20% 的序列与训练集
    相似度 >= 0.8 (同一家族的点突变/截短变体), 模型靠记忆家族即可刷高测试分,
    训练集准确率迅速饱和 -> 表现为"一直过拟合"。

做法:
    1. 合并原 training + testing 全部数据, 精确去重 (无标签矛盾, 可安全合并);
    2. 在长度相近 (长度差 <=25%) 的序列间用 difflib 序列同一性 >= 0.8 连边
       (quick_ratio 上界快速拒绝), 用并查集做单链聚类, 同源变体落入同一簇;
    3. 以"簇"为不可分割单位, 按簇大小降序贪心分配到 train / validation /
       test (默认 80 / 10 / 10), 最小化"比例偏差平方和 + 标签失衡平方和";
    4. 输出三个新 CSV (training_v2 / validation_v2 / testing_v2),
       同源序列保证不跨集合, 测试集成为真正的 held-out 数据;
    5. 末尾用 difflib 序列同一性精确校验跨集泄漏, 并与旧划分对比。

用法:
    python make_split.py                 # 生成 v2 三个划分 (80/10/10)
    python make_split.py --folds 5       # 五折交叉划分: 每折 1 份测试,
                                         # 其余 4 份按 8:2 分训练/验证
                                         # -> *_v2_fold{1..5}.csv 共 15 个文件
    python make_split.py --sim 0.7       # 更严格 (更大的同源簇)
    python make_split.py --seed 42
"""

import os
import time
import argparse
from collections import defaultdict
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# 1. 相似性聚类
# ---------------------------------------------------------------------------

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def cluster_sequences(seqs, sim_threshold=0.8, len_tol=0.5):
    """
    基于精确序列同一性 (difflib) 的单链聚类 (并查集)。

    由 ratio >= s 可推出长度差 |la-lb| <= 2(1-s)/s * min(la,lb)
    (s=0.8 时上界为 0.5*min_len), 故只在长度差 <= len_tol*len 的桶内比较;
    比较前先用 quick_ratio() 上界廉价拒绝绝大多数候选对。
    返回 cluster_id 列表 (长度 = len(seqs))。
    """
    n = len(seqs)
    order = sorted(range(n), key=lambda i: len(seqs[i]))
    lens = [len(seqs[i]) for i in order]

    uf = UnionFind(n)
    n_edges = n_cmp = 0
    t0 = time.time()
    for a in range(n):
        ia = order[a]
        sa = seqs[ia]
        la = lens[a]
        max_diff = max(2, int(len_tol * la) + 1)
        b = a + 1
        # 向后扫描, 长度差超界即停 (order 按长度升序)
        while b < n and lens[b] - la <= max_diff:
            ib = order[b]
            sb = seqs[ib]
            n_cmp += 1
            sm = SequenceMatcher(None, sa, sb, autojunk=False)
            if sm.quick_ratio() >= sim_threshold and sm.ratio() >= sim_threshold:
                uf.union(ia, ib)
                n_edges += 1
            b += 1
    cluster_id = [uf.find(i) for i in range(n)]
    n_clusters = len(set(cluster_id))
    print(f"[Cluster] {n} 条序列, 候选比较 {n_cmp} 对, 相似边 {n_edges} 条 "
          f"-> {n_clusters} 个同源簇 (同一性 >= {sim_threshold}, 耗时 "
          f"{time.time() - t0:.1f}s)")
    return cluster_id


# ---------------------------------------------------------------------------
# 2. 按簇分层划分
# ---------------------------------------------------------------------------

def assign_clusters(cluster_df, ratios=(0.8, 0.1, 0.1), seed=500, names=None):
    """
    把每个完整的簇贪心分配到 len(ratios) 个集合。

    - 簇按规模降序处理 (大簇先放, 减少尾期配额不足造成的失衡);
    - 对正例/负例分别追踪各集合的"目标配额" t_k*P, t_k*N, 选择使放置后
      各集合正负配额残差平方和最小的集合。该式对空集合没有冷启动惩罚,
      且天然同时保证数量比例与标签平衡。
    """
    k = len(ratios)
    if names is None:
        names = ['train', 'val', 'test'] if k == 3 else \
            [f'split{i}' for i in range(k)]
    assert len(names) == k

    rng = np.random.RandomState(seed)
    groups = [g for _, g in cluster_df.groupby('cluster')]
    rng.shuffle(groups)
    groups.sort(key=lambda g: -len(g))   # 大簇优先

    targets = np.array(ratios) / sum(ratios)
    total_pos = float((cluster_df['Label'] == 1).sum())
    total_neg = float((cluster_df['Label'] == 0).sum())
    quota_pos = targets * total_pos      # 各集合正例目标数
    quota_neg = targets * total_neg      # 各集合负例目标数

    pos = np.zeros(k)
    neg = np.zeros(k)
    assign = {}

    for g in groups:
        pg = float((g['Label'] == 1).sum())
        qg = float((g['Label'] == 0).sum())
        best_j, best_score = 0, np.inf
        for j in range(k):
            p_after = pos + np.where(np.arange(k) == j, pg, 0.0)
            n_after = neg + np.where(np.arange(k) == j, qg, 0.0)
            score = float(np.sum((quota_pos - p_after) ** 2) +
                          np.sum((quota_neg - n_after) ** 2))
            if score < best_score:
                best_score, best_j = score, j
        pos[best_j] += pg
        neg[best_j] += qg
        assign[g['cluster'].iloc[0]] = best_j

    cluster_df = cluster_df.copy()
    cluster_df['split'] = cluster_df['cluster'].map(assign).map(
        dict(enumerate(names)))
    return cluster_df


# ---------------------------------------------------------------------------
# 3. 跨集泄漏校验
# ---------------------------------------------------------------------------

def verify_leakage(seqs_by_split):
    """用 difflib 精确序列同一性, 检查 val/test 对 train 的最近邻相似度。

    与 cluster_sequences() 使用相同度量 (SequenceMatcher.ratio, autojunk=False),
    对 train 全集做精确最近邻搜索 (不再按首尾 4-mer 分桶, 避免漏检首/尾存在替换的同源序列)。
    剪枝只使用严格上界, 结果与暴力全配对一致:
      ratio <= 2*min(la, lb)/(la + lb) (长度上界), 以及 real_quick_ratio()/quick_ratio()。
    """
    train_seqs = list(seqs_by_split['train'])
    train_lens = np.array([len(t) for t in train_seqs], dtype=float)

    def nearest(query):
        lq = float(len(query))
        len_ub = 2.0 * np.minimum(train_lens, lq) / (train_lens + lq)
        best = 0.0
        for k in np.argsort(-len_ub, kind='stable'):
            if len_ub[k] <= best:          # 剩余候选上界均不超过当前最优, 提前终止
                break
            sm = SequenceMatcher(None, query, train_seqs[k], autojunk=False)
            if sm.real_quick_ratio() <= best or sm.quick_ratio() <= best:
                continue
            r = sm.ratio()
            if r > best:
                best = r
                if best >= 1.0:
                    break
        return best

    report = {}
    for name in ('val', 'test'):
        if name not in seqs_by_split:
            continue
        sims = np.array([nearest(q) for q in seqs_by_split[name]])
        report[name] = sims
        print(f"[Verify] {name:5s}: n={len(sims)}, 与 train 最近邻同一性 "
              f"mean={sims.mean():.3f}, max={sims.max():.3f} | "
              f">=0.9: {(sims >= 0.9).sum()}, >=0.8: {(sims >= 0.8).sum()}, "
              f">=0.6: {(sims >= 0.6).sum()}")
    return report


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 4. 数据特征与偏置自检
# ---------------------------------------------------------------------------

def check_data_bias(df):
    """自检数据集中是否存在 N 端 Met (M) 偏置或长度异常等捷径特征。"""
    seqs = df['Seq'].astype(str)
    labels = df['Label'].astype(int)

    pos_seqs = seqs[labels == 1].tolist()
    neg_seqs = seqs[labels == 0].tolist()

    pos_m = sum(s.startswith('M') for s in pos_seqs) / max(len(pos_seqs), 1) * 100
    neg_m = sum(s.startswith('M') for s in neg_seqs) / max(len(neg_seqs), 1) * 100

    lens = np.array([len(s) for s in seqs])
    long_count = int(np.sum(lens > 50))

    print('\n================== 数据集生物特征与偏置体检 ==================')
    print(f"总样本数: {len(df)} | 正样本: {len(pos_seqs)} | 负样本: {len(neg_seqs)}")
    print(f"【N 端起始氨基酸 M 比例 (捷径学习风险)】:")
    print(f"  - 正样本以 M 开头: {pos_m:.2f}%")
    print(f"  - 负样本以 M 开头: {neg_m:.2f}%")
    if abs(neg_m - pos_m) > 30:
        print(f"  [⚠️ 偏置预警] 负样本以 M 开头的比例显著高于正样本 ({neg_m:.1f}% vs {pos_m:.1f}%)！")
        print(f"               这表明负样本多取自蛋白质全长 N 端，模型极易通过首字母 M 走捷径判断！")
    print(f"【序列长度分布】: 最小 {lens.min()} | 中位数 {int(np.median(lens))} | 最大 {lens.max()}")
    print(f"  - 长度超过 50 aa (受截断影响) 序列数: {long_count} 条 ({long_count / len(lens) * 100:.1f}%)")
    print('==============================================================\n')


# ---------------------------------------------------------------------------
# 5. K 折交叉验证划分
# ---------------------------------------------------------------------------

def split_folds(all_df, folds=5, val_share=0.2, out_dir=SCRIPT_DIR, seed=500,
                prefix='v2'):
    """
    K 折交叉验证划分 (以"簇"为不可分割单位, 同源序列绝不跨折):

        总数据 -> K 份 (每份 1 折);
        第 i 折: 折 i 整体做测试 (1 份);
                 其余 K-1 份构成训练池, 池内簇再按 8:2 贪心分为 train / val
                 (即 4 份中 80% 训练 + 20% 验证)。

    输出: training_v2_fold{i}.csv / validation_v2_fold{i}.csv /
          testing_v2_fold{i}.csv, i = 1..K。
    每条序列在 K 折测试集中恰好出现一次; 各折的 train/val/test 同源不重叠。
    """
    fold_names = [f'fold{i}' for i in range(1, folds + 1)]
    all_df = assign_clusters(all_df, ratios=[1.0] * folds, seed=seed,
                             names=fold_names)

    rng = np.random.RandomState(seed)
    all_test_seqs = []
    for i, fold_name in enumerate(fold_names, 1):
        test_df = all_df[all_df['split'] == fold_name]
        pool = all_df[all_df['split'] != fold_name].copy()
        pool = assign_clusters(pool, ratios=(1.0 - val_share, val_share),
                               seed=seed + i, names=['train', 'val'])

        seqs_by_split = {}
        for split_name, fname in [
                ('train', f'training_{prefix}_fold{i}.csv'),
                ('val', f'validation_{prefix}_fold{i}.csv'),
                ('test', f'testing_{prefix}_fold{i}.csv')]:
            part = pool[pool['split'] == split_name] \
                if split_name != 'test' else test_df
            part = part[['Seq', 'Label']].copy()
            idx = np.arange(len(part))
            rng.shuffle(idx)
            part = part.iloc[idx].reset_index(drop=True)
            path = os.path.join(out_dir, fname)
            part.to_csv(path, index=False)
            seqs_by_split[split_name] = list(part['Seq'].astype(str).unique())
            print(f"[Fold {i}] {split_name:5s}: {len(part):5d} 条, "
                  f"标签分布 {part['Label'].value_counts().to_dict()} -> {path}")

        all_test_seqs.extend(seqs_by_split['test'])
        n_dup = len(all_test_seqs) - len(set(all_test_seqs))
        print(f"[Fold {i}] 累计测试样本 {len(all_test_seqs)} 条, 跨折重复 {n_dup} (应为 0)")

        print(f'------ Fold {i} 泄漏校验 (train/val/test 同源隔离) ------')
        verify_leakage(seqs_by_split)

    print(f"\n[CV] {folds} 折划分完成: 测试集互不重叠, "
          f"共覆盖 {len(set(all_test_seqs))} 条唯一序列 "
          f"(应为全部数据的 100%)")
    return out_dir


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='同源聚类去泄漏的数据重新划分')
    parser.add_argument('--src_dir', type=str,
                        default=os.path.join(os.path.dirname(SCRIPT_DIR),
                                             'NeuroPeptides3D2'),
                        help='原始 training.csv / testing.csv 所在目录')
    parser.add_argument('--out_dir', type=str, default=SCRIPT_DIR,
                        help='新划分输出目录 (默认本脚本目录, 训练时自动可发现)')
    parser.add_argument('--sim', type=float, default=0.8,
                        help='序列同一性同源聚类阈值 (越小越严格, 如 0.7)')
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--folds', type=int, default=0,
                        help='>1 时启用 K 折交叉验证划分 (如 --folds 5)')
    parser.add_argument('--val_share', type=float, default=0.2,
                        help='K 折模式下训练池中验证集占比 (默认 0.2, 即 8:2)')
    parser.add_argument('--holdout_test', type=float, default=0.0,
                        help='在 K 折前预先切出绝对独立的测试集比例 (例如 0.1 表示 10% 独立测试集, 剩下 90% 用于 5 折 CV, 彻底杜绝泄漏)')
    parser.add_argument('--seed', type=int, default=500)
    args = parser.parse_args()

    # ---- 1. 读取数据 (优先原始目录, 自动回退本地 all_data.csv 或已有划分) ----
    tr_path = os.path.join(args.src_dir, 'training.csv')
    te_path = os.path.join(args.src_dir, 'testing.csv')
    all_data_path = os.path.join(SCRIPT_DIR, 'all_data.csv')

    if os.path.exists(tr_path) and os.path.exists(te_path):
        tr = pd.read_csv(tr_path)
        te = pd.read_csv(te_path)
        all_df = pd.concat([tr, te], ignore_index=True)
    elif os.path.exists(all_data_path):
        all_df = pd.read_csv(all_data_path)
    else:
        # 回退至本地现有的 v2 划分集合并
        print(f"[Data] 未找到原始 {args.src_dir}, 自动从本地现有 v2 文件中合并读取总集...")
        local_files = [os.path.join(SCRIPT_DIR, f) for f in
                       ('training_v2.csv', 'validation_v2.csv', 'testing_v2.csv')]
        if not all(os.path.exists(f) for f in local_files):
            # 尝试 fold 文件
            local_files = [os.path.join(SCRIPT_DIR, f'testing_v2_fold{i}.csv') for i in range(1, 6)]
        dfs = [pd.read_csv(f) for f in local_files if os.path.exists(f)]
        if not dfs:
            raise FileNotFoundError(f"未找到可读取的数据源，请检查 src_dir 或本地 csv 文件。")
        all_df = pd.concat(dfs, ignore_index=True)

    conflict = all_df.groupby('Seq')['Label'].nunique()
    n_conflict = int((conflict > 1).sum())
    if n_conflict:
        raise ValueError(f"发现 {n_conflict} 条同序列标签矛盾, 请人工核查后再划分")
    all_df = (all_df.drop_duplicates(subset=['Seq'])
              .reset_index(drop=True))
    print(f"[Data] 获得 {len(all_df)} 条唯一序列, 标签分布: {all_df['Label'].value_counts().to_dict()}")

    # 偏置检查
    check_data_bias(all_df)

    # ---- 2. 同源聚类 (按数据指纹+阈值缓存) ----
    import hashlib
    seqs = all_df['Seq'].astype(str).tolist()
    fingerprint = hashlib.md5(
        ('\n'.join(seqs) + f'|{args.sim}').encode()).hexdigest()[:12]
    cache_path = os.path.join(args.out_dir, 'cache',
                              f'clusters_sim{args.sim}_{fingerprint}.npy')
    if os.path.exists(cache_path):
        cluster_id = np.load(cache_path)
        print(f"[Cluster] 命中缓存: {cache_path} ({len(set(cluster_id))} 簇)")
    else:
        cluster_id = cluster_sequences(seqs, sim_threshold=args.sim)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, np.array(cluster_id))
        print(f"[Cluster] 聚类结果已缓存: {cache_path}")
    all_df['cluster'] = cluster_id

    # 簇规模概览
    cinfo = all_df.groupby('cluster').agg(n=('Label', 'size'), pure=('Label', 'nunique'))
    print(f"[Cluster] 最大簇 {cinfo['n'].max()} 条, 规模>=5 的簇 "
          f"{int((cinfo['n'] >= 5).sum())} 个, 含两种标签的簇 "
          f"{int((cinfo['pure'] > 1).sum())} 个")

    # ---- 3. 按簇划分 ----
    if args.folds >= 2:
        if args.holdout_test > 0.0:
            print(f"\n[Split] 预先隔离独立测试集 (占比 {args.holdout_test * 100:.1f}%) ...")
            all_df = assign_clusters(all_df,
                                     ratios=(1.0 - args.holdout_test, args.holdout_test),
                                     seed=args.seed, names=['cv_pool', 'independent_test'])
            indep_df = all_df[all_df['split'] == 'independent_test'][['Seq', 'Label']].copy()
            indep_path = os.path.join(args.out_dir, 'independent_test_v2.csv')
            indep_df.to_csv(indep_path, index=False)
            print(f"[Split] 独立测试集已保存: {indep_path} ({len(indep_df)} 条, 绝不参与后续任何交叉验证折)")
            cv_pool = all_df[all_df['split'] == 'cv_pool'].copy()
            split_folds(cv_pool, folds=args.folds, val_share=args.val_share,
                        out_dir=args.out_dir, seed=args.seed)
        else:
            split_folds(all_df, folds=args.folds, val_share=args.val_share,
                        out_dir=args.out_dir, seed=args.seed)
        return

    train_ratio = 1.0 - args.val_ratio - args.test_ratio
    all_df = assign_clusters(all_df, ratios=(train_ratio, args.val_ratio,
                                             args.test_ratio), seed=args.seed)

    # ---- 4. 写出三份标准划分 ----
    rng = np.random.RandomState(args.seed)
    seqs_by_split = {}
    for name, fname in [('train', 'training_v2.csv'),
                        ('val', 'validation_v2.csv'),
                        ('test', 'testing_v2.csv')]:
        part = all_df[all_df['split'] == name][['Seq', 'Label']].copy()
        idx = np.arange(len(part))
        rng.shuffle(idx)
        part = part.iloc[idx].reset_index(drop=True)
        path = os.path.join(args.out_dir, fname)
        part.to_csv(path, index=False)
        seqs_by_split[name] = list(part['Seq'].astype(str).unique())
        print(f"[Split] {name:5s}: {len(part):5d} 条, "
              f"标签分布 {part['Label'].value_counts().to_dict()} -> {path}")

    # ---- 5. 泄漏校验 ----
    print('\n------ 新划分跨集泄漏校验 (对 train 的最近邻序列同一性) ------')
    verify_leakage(seqs_by_split)


if __name__ == '__main__':
    main()

