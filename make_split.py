# -*- coding: utf-8 -*-
"""Single-linkage sequence homology clustering and quota-balanced dataset partitioning."""

import os
import time
import hashlib
import argparse
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


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
    """Single-linkage sequence clustering under SequenceMatcher identity >= sim_threshold."""
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
    print(f"[Cluster] {n} sequences, {n_cmp} pairs checked, {n_edges} edges "
          f"-> {n_clusters} clusters (sim >= {sim_threshold}, {time.time() - t0:.1f}s)")
    return cluster_id


def assign_clusters(cluster_df, ratios=(0.8, 0.1, 0.1), seed=500, names=None):
    """Greedily assign intact homology clusters in descending size order to match target class quotas."""
    k = len(ratios)
    if names is None:
        names = ['train', 'val', 'test'] if k == 3 else [f'split{i}' for i in range(k)]
    assert len(names) == k

    rng = np.random.RandomState(seed)
    groups = [g for _, g in cluster_df.groupby('cluster')]
    rng.shuffle(groups)
    groups.sort(key=lambda g: -len(g))

    targets = np.array(ratios) / sum(ratios)
    total_pos = float((cluster_df['Label'] == 1).sum())
    total_neg = float((cluster_df['Label'] == 0).sum())
    quota_pos = targets * total_pos
    quota_neg = targets * total_neg

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
    cluster_df['split'] = cluster_df['cluster'].map(assign).map(dict(enumerate(names)))
    return cluster_df


def verify_leakage(seqs_by_split):
    """Compute exact nearest-neighbor sequence identity from val/test against train."""
    train_seqs = list(seqs_by_split['train'])
    train_lens = np.array([len(t) for t in train_seqs], dtype=float)

    def nearest(query):
        lq = float(len(query))
        len_ub = 2.0 * np.minimum(train_lens, lq) / (train_lens + lq)
        best = 0.0
        for k in np.argsort(-len_ub, kind='stable'):
            if len_ub[k] <= best:
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
        print(f"[Verify] {name:5s}: n={len(sims)}, nearest train identity "
              f"mean={sims.mean():.3f}, max={sims.max():.3f} | "
              f">=0.9: {(sims >= 0.9).sum()}, >=0.8: {(sims >= 0.8).sum()}, "
              f">=0.6: {(sims >= 0.6).sum()}")
    return report


def check_data_bias(df):
    """Report sequence length statistics and N-terminal Met1 prevalence by class."""
    seqs = df['Seq'].astype(str)
    labels = df['Label'].astype(int)

    pos_seqs = seqs[labels == 1].tolist()
    neg_seqs = seqs[labels == 0].tolist()

    pos_m = sum(s.startswith('M') for s in pos_seqs) / max(len(pos_seqs), 1) * 100
    neg_m = sum(s.startswith('M') for s in neg_seqs) / max(len(neg_seqs), 1) * 100

    lens = np.array([len(s) for s in seqs])
    long_count = int(np.sum(lens > 50))

    print(f"[Corpus] N={len(df)} (Pos={len(pos_seqs)}, Neg={len(neg_seqs)}) | "
          f"Len min={lens.min()}, median={int(np.median(lens))}, max={lens.max()}, "
          f">50aa={long_count} ({long_count / len(lens) * 100:.1f}%) | "
          f"N-term Met1: Pos={pos_m:.2f}%, Neg={neg_m:.2f}%")


def split_folds(all_df, folds=5, val_share=0.2, out_dir=SCRIPT_DIR, seed=500, prefix='v2'):
    """Partition clusters into K disjoint test folds, splitting remaining K-1 folds into train/val."""
    fold_names = [f'fold{i}' for i in range(1, folds + 1)]
    all_df = assign_clusters(all_df, ratios=[1.0] * folds, seed=seed, names=fold_names)

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
            part = pool[pool['split'] == split_name] if split_name != 'test' else test_df
            part = part[['Seq', 'Label']].copy()
            idx = np.arange(len(part))
            rng.shuffle(idx)
            part = part.iloc[idx].reset_index(drop=True)
            path = os.path.join(out_dir, fname)
            part.to_csv(path, index=False)
            seqs_by_split[split_name] = list(part['Seq'].astype(str).unique())
            print(f"[Fold {i}] {split_name:5s}: n={len(part):5d}, "
                  f"labels={part['Label'].value_counts().to_dict()} -> {path}")

        all_test_seqs.extend(seqs_by_split['test'])
        n_dup = len(all_test_seqs) - len(set(all_test_seqs))
        print(f"[Fold {i}] cumulative test sequences={len(all_test_seqs)}, cross-fold duplicates={n_dup}")
        verify_leakage(seqs_by_split)

    print(f"[CV] {folds}-fold split complete: {len(set(all_test_seqs))} unique test sequences.")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description='Homology-isolated dataset partitioning for NeuroEGNN')
    parser.add_argument('--src_dir', type=str,
                        default=os.path.join(os.path.dirname(SCRIPT_DIR), 'NeuroPeptides3D2'),
                        help='Optional directory containing raw training.csv and testing.csv')
    parser.add_argument('--out_dir', type=str, default=SCRIPT_DIR,
                        help='Output directory for split CSV files')
    parser.add_argument('--sim', type=float, default=0.8,
                        help='Sequence identity clustering threshold (default: 0.8)')
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--folds', type=int, default=0,
                        help='Number of cross-validation folds when >= 2')
    parser.add_argument('--val_share', type=float, default=0.2,
                        help='Validation fraction within the K-1 non-test folds (default: 0.2)')
    parser.add_argument('--holdout_test', type=float, default=0.0,
                        help='Optional fraction reserved as an external hold-out set prior to K-fold splitting')
    parser.add_argument('--seed', type=int, default=500)
    args = parser.parse_args()

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
        local_files = [os.path.join(SCRIPT_DIR, f) for f in
                       ('training_v2.csv', 'validation_v2.csv', 'testing_v2.csv')]
        if not all(os.path.exists(f) for f in local_files):
            local_files = [os.path.join(SCRIPT_DIR, f'testing_v2_fold{i}.csv') for i in range(1, 6)]
        dfs = [pd.read_csv(f) for f in local_files if os.path.exists(f)]
        if not dfs:
            raise FileNotFoundError("No input dataset CSV files found.")
        all_df = pd.concat(dfs, ignore_index=True)

    conflict = all_df.groupby('Seq')['Label'].nunique()
    n_conflict = int((conflict > 1).sum())
    if n_conflict:
        raise ValueError(f"Found {n_conflict} sequences with conflicting labels.")
    all_df = all_df.drop_duplicates(subset=['Seq']).reset_index(drop=True)
    check_data_bias(all_df)

    seqs = all_df['Seq'].astype(str).tolist()
    fingerprint = hashlib.md5(('\n'.join(seqs) + f'|{args.sim}').encode()).hexdigest()[:12]
    cache_path = os.path.join(args.out_dir, 'cache', f'clusters_sim{args.sim}_{fingerprint}.npy')
    if os.path.exists(cache_path):
        cluster_id = np.load(cache_path)
        print(f"[Cluster] Loaded cached clusters: {cache_path} ({len(set(cluster_id))} clusters)")
    else:
        cluster_id = cluster_sequences(seqs, sim_threshold=args.sim)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, np.array(cluster_id))
        print(f"[Cluster] Saved cluster assignments: {cache_path}")
    all_df['cluster'] = cluster_id

    cinfo = all_df.groupby('cluster').agg(n=('Label', 'size'), pure=('Label', 'nunique'))
    print(f"[Cluster] largest_cluster={cinfo['n'].max()}, clusters_ge_5={int((cinfo['n'] >= 5).sum())}, "
          f"mixed_label_clusters={int((cinfo['pure'] > 1).sum())}")

    if args.folds >= 2:
        if args.holdout_test > 0.0:
            all_df = assign_clusters(all_df,
                                     ratios=(1.0 - args.holdout_test, args.holdout_test),
                                     seed=args.seed, names=['cv_pool', 'independent_test'])
            indep_df = all_df[all_df['split'] == 'independent_test'][['Seq', 'Label']].copy()
            indep_path = os.path.join(args.out_dir, 'independent_test_v2.csv')
            indep_df.to_csv(indep_path, index=False)
            print(f"[Split] Saved hold-out test set: {indep_path} (n={len(indep_df)})")
            cv_pool = all_df[all_df['split'] == 'cv_pool'].copy()
            split_folds(cv_pool, folds=args.folds, val_share=args.val_share,
                        out_dir=args.out_dir, seed=args.seed)
        else:
            split_folds(all_df, folds=args.folds, val_share=args.val_share,
                        out_dir=args.out_dir, seed=args.seed)
        return

    train_ratio = 1.0 - args.val_ratio - args.test_ratio
    all_df = assign_clusters(all_df, ratios=(train_ratio, args.val_ratio, args.test_ratio), seed=args.seed)

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
        print(f"[Split] {name:5s}: n={len(part):5d}, labels={part['Label'].value_counts().to_dict()} -> {path}")

    verify_leakage(seqs_by_split)


if __name__ == '__main__':
    main()
