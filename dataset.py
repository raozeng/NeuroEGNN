# -*- coding: utf-8 -*-
"""ESM-2 feature caching, Torgerson MDS coordinate reconstruction, and 3D residue graph dataset."""

import os
import ssl
import time
import hashlib
import urllib.request

import numpy as np
import esm
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

STANDARD_AMINO_ACIDS = set('ARNDCQEGHILKMFPSTWYV')
ONE_HOT_ELEMENTS = 'ALRKNMDFCPQSETGWHYIV'

# Kyte-Doolittle hydropathy, Grantham polarity, Dawson volume, pH 7 net charge, Vihinen flexibility
_HYDRO = dict(A=1.8, R=-4.5, N=-3.5, D=-3.5, C=2.5, Q=-3.5, E=-3.5, G=-0.4,
              H=-3.2, I=4.5, L=3.8, K=-3.9, M=1.9, F=2.8, P=-1.6, S=-0.8,
              T=-0.7, W=-0.9, Y=-1.3, V=4.2)
_POLAR = dict(A=8.1, R=10.5, N=11.6, D=13.0, C=5.5, Q=10.5, E=12.3, G=9.0,
              H=10.4, I=5.2, L=4.9, K=11.3, M=5.7, F=5.2, P=8.0, S=9.2,
              T=8.6, W=5.4, Y=6.2, V=5.9)
_VOLUME = dict(A=88.6, R=173.4, N=114.1, D=111.1, C=108.5, Q=143.8, E=138.4,
               G=60.1, H=153.2, I=166.7, L=166.7, K=168.6, M=162.9, F=189.9,
               P=112.7, S=89.0, T=116.1, W=227.8, Y=193.6, V=140.0)
_CHARGE = dict(A=0.0, R=1.0, N=0.0, D=-1.0, C=0.0, Q=0.0, E=-1.0, G=0.0,
               H=0.1, I=0.0, L=0.0, K=1.0, M=0.0, F=0.0, P=0.0, S=0.0,
               T=0.0, W=0.0, Y=0.0, V=0.0)
_FLEX = dict(A=0.36, R=0.53, N=0.46, D=0.51, C=0.35, Q=0.49, E=0.50, G=0.54,
             H=0.32, I=0.46, L=0.37, K=0.53, M=0.30, F=0.31, P=0.51, S=0.51,
             T=0.44, W=0.31, Y=0.42, V=0.39)

PROP_TABLES = [_HYDRO, _POLAR, _VOLUME, _CHARGE, _FLEX]


def _normalize(table):
    vals = np.array([table[a] for a in 'ARNDCQEGHILKMFPSTWYV'], dtype=np.float32)
    scale = np.max(np.abs(vals)) + 1e-8
    return {a: float(v / scale) for a, v in zip('ARNDCQEGHILKMFPSTWYV', vals)}


PROP_TABLES = [_normalize(t) for t in PROP_TABLES]
NODE_FEAT_DIM = 21 + len(PROP_TABLES)


def get_node_features(seq):
    """Return per-residue 26-D biochemical features (21-D one-hot + 5 normalized properties)."""
    L = len(seq)
    feats = np.zeros((L, NODE_FEAT_DIM), dtype=np.float32)
    onehot = np.eye(21, dtype=np.float32)
    for i, aa in enumerate(seq):
        idx = ONE_HOT_ELEMENTS.find(aa)
        if idx == -1:
            idx = 20
        feats[i, :21] = onehot[idx]
        for k, table in enumerate(PROP_TABLES):
            feats[i, 21 + k] = table.get(aa, 0.0)
    return feats


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_data_path(path):
    if os.path.exists(path):
        return path
    for base in (SCRIPT_DIR, os.path.dirname(SCRIPT_DIR)):
        cand = os.path.join(base, os.path.basename(path))
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(f"Data file not found: {path}")


def seq_hash(seq):
    return hashlib.md5(seq.encode('utf-8')).hexdigest()[:16]


def clean_sequence(seq, max_len=50, trunc='left'):
    cleaned = ''.join(aa if aa in STANDARD_AMINO_ACIDS else 'X' for aa in str(seq))
    if len(cleaned) > max_len:
        return cleaned[-max_len:] if trunc == 'right' else cleaned[:max_len]
    return cleaned


ESMFOLD_API_URL = 'https://api.esmatlas.com/foldSequence/v1/pdb/'


def fetch_esmfold_api(seq, timeout=30, retries=2):
    body = seq.encode('utf-8')
    for attempt in range(retries + 1):
        try:
            ctx = ssl.create_default_context()
            req = urllib.request.Request(
                ESMFOLD_API_URL, data=body, method='POST',
                headers={'Content-Type': 'text/plain'})
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
                pdb_text = resp.read().decode('utf-8', errors='ignore')
            if 'ATOM' in pdb_text:
                return pdb_text
        except Exception:
            if attempt < retries:
                time.sleep(1.0)
    return None


def parse_pdb_ca_coords(pdb_text, max_res=100):
    coords = []
    for line in pdb_text.splitlines():
        if line.startswith('ENDMDL'):
            break
        if line.startswith('ATOM') and line[12:16].strip() == 'CA':
            try:
                coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            except ValueError:
                continue
            if len(coords) >= max_res:
                break
    return np.array(coords, dtype=np.float32) if coords else None


def coords_from_contacts(contact_map):
    """Reconstruct 3D C-alpha coordinates from a contact map via calibrated Torgerson MDS."""
    c = np.asarray(contact_map, dtype=np.float64)
    dist = 1.0 - c
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0.0, None)

    n = dist.shape[0]
    if n < 4:
        return None
    try:
        d2 = dist ** 2
        j = np.eye(n) - np.ones((n, n)) / n
        b = -0.5 * j @ d2 @ j
        eigvals, eigvecs = np.linalg.eigh(b)
        top = np.argsort(eigvals)[::-1][:3]
        lam = np.clip(eigvals[top], 0.0, None)
        coords = eigvecs[:, top] * np.sqrt(lam)
        if coords.shape[1] < 3:
            coords = np.pad(coords, ((0, 0), (0, 3 - coords.shape[1])))
        if not np.all(np.isfinite(coords)):
            return None
        # Rescale dimensionless MDS embedding to physical Angstrom units (C-alpha virtual bond ~3.8 A)
        if n >= 2:
            bond_dists = np.linalg.norm(coords[1:] - coords[:-1], axis=-1)
            mean_bond = float(np.mean(bond_dists))
            if mean_bond > 1e-4:
                scale = float(np.clip(3.8 / mean_bond, 1.0, 50.0))
                coords = coords * scale
        return coords.astype(np.float32)
    except Exception:
        return None


def coords_synthetic(seq):
    """Deterministic helical fallback trace used only for degenerate inputs (L < 4)."""
    seed = int(seq_hash(seq), 16) % (2 ** 32)
    rng = np.random.RandomState(seed)
    L = max(len(seq), 1)
    t = np.arange(L, dtype=np.float64)
    radius, rise, angular = 2.5, 1.5, 2.0 * np.pi / 3.6
    wobble = 0.8 * np.sin(t / 3.0 + rng.rand() * 6.28)
    coords = np.stack([
        radius * np.cos(angular * t) + wobble,
        radius * np.sin(angular * t) + 0.8 * np.cos(t / 4.0 + rng.rand() * 6.28),
        rise * t,
    ], axis=1)
    coords += rng.normal(scale=0.2, size=coords.shape)
    return coords.astype(np.float32)


def pad_missing_ca(coords, target_len, seq):
    if coords.shape[0] >= target_len:
        return coords[:target_len]
    seed = int(seq_hash(seq), 16) % (2 ** 32)
    rng = np.random.RandomState(seed)
    out = [coords]
    last = coords[-1]
    direction = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    while sum(c.shape[0] for c in out) < target_len:
        theta = rng.rand() * 2 * np.pi
        direction = direction + 0.4 * np.array([np.cos(theta), np.sin(theta), 0.0])
        direction /= (np.linalg.norm(direction) + 1e-8)
        out.append((last + 3.8 * direction)[None, :])
        last = out[-1][0]
    return np.concatenate(out, axis=0)[:target_len]


class StructureStore:
    """Disk-backed cache of per-sequence 3D C-alpha coordinates."""

    def __init__(self, cache_dir):
        self.cache_dir = os.path.join(cache_dir, 'structures')
        os.makedirs(self.cache_dir, exist_ok=True)
        self._mem = {}

    def path(self, seq):
        return os.path.join(self.cache_dir, seq_hash(seq) + '.npy')

    def get(self, seq):
        if seq in self._mem:
            return self._mem[seq]
        p = self.path(seq)
        if os.path.exists(p):
            try:
                coords = np.load(p)
                self._mem[seq] = coords
                return coords
            except Exception:
                pass
        return None

    def put(self, seq, coords):
        self._mem[seq] = coords
        try:
            np.save(self.path(seq), coords.astype(np.float32))
        except Exception:
            pass


class ESMFeaturizer:
    """Frozen ESM-2 wrapper for batched extraction of residue embeddings and contact maps."""

    def __init__(self, model_name='esm2_t33_650M_UR50D', device=None):
        self.model_name = model_name
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[ESM] Loading {model_name} on {self.device}...")
        import esm
        torch_home = os.environ.get('TORCH_HOME', os.path.expanduser('~/.cache/torch'))
        local_ckpt = os.path.join(torch_home, 'hub', 'checkpoints', f'{model_name}.pt')
        if os.path.isfile(model_name):
            self.model, self.alphabet = esm.pretrained.load_model_and_alphabet_local(model_name)
        elif os.path.isfile(local_ckpt):
            self.model, self.alphabet = esm.pretrained.load_model_and_alphabet_local(local_ckpt)
        elif hasattr(esm.pretrained, model_name):
            self.model, self.alphabet = getattr(esm.pretrained, model_name)()
        else:
            self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model.eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        cfg = getattr(self.model, 'args', None) or getattr(self.model, 'cfg', None)
        self.embed_dim = int(getattr(cfg, 'encoder_embed_dim', 1280))
        print(f"[ESM] Loaded {model_name} (embed_dim={self.embed_dim})")

    @torch.no_grad()
    def embed_batch(self, seqs, batch_size=64):
        n = len(seqs)
        max_len = max(len(s) for s in seqs)
        emb = np.zeros((n, max_len, self.embed_dim), dtype=np.float32)
        contacts = None
        for start in range(0, n, batch_size):
            chunk = seqs[start:start + batch_size]
            data = [(f's{i}', s) for i, s in enumerate(chunk)]
            _, _, tokens = self.batch_converter(data)
            tokens = tokens.to(self.device)
            results = self.model(tokens, repr_layers=[self.model.num_layers],
                                 return_contacts=True)
            reprs = results['representations'][self.model.num_layers][:, 1:-1]
            try:
                ct = results['contacts'].cpu().numpy()
            except Exception:
                ct = None
            for i, s in enumerate(chunk):
                emb[start + i, :len(s)] = reprs[i, :len(s)].cpu().numpy()
            if ct is not None:
                if contacts is None:
                    contacts = np.zeros((n, max_len, max_len), dtype=np.float32)
                for i, s in enumerate(chunk):
                    L = len(s)
                    contacts[start + i, :L, :L] = ct[i, :L, :L]
        return emb, contacts


def prepare_esm_cache(csv_path, cache_dir, esm_model='esm2_t33_650M_UR50D',
                      max_len=50, use_api=False, api_limit=0, api_sleep=0.2):
    """Extract and cache ESM-2 features and 3D coordinates for all unique sequences in a CSV."""
    csv_path = resolve_data_path(csv_path)
    df = pd.read_csv(csv_path)
    seq_col = 'Seq' if 'Seq' in df.columns else 'text'
    raw_seqs = df[seq_col].astype(str).tolist()
    uniq = list(dict.fromkeys(clean_sequence(s, max_len) for s in raw_seqs))
    print(f"[Prepare] {os.path.basename(csv_path)}: {len(raw_seqs)} rows, "
          f"{len(uniq)} unique sequences (max_len={max_len})")

    store = StructureStore(cache_dir)

    n_api = 0
    if use_api:
        budget = len(uniq) if api_limit <= 0 else min(api_limit, len(uniq))
        print(f"[Prepare] Querying ESMFold API (budget={budget})...")
        for i, seq in enumerate(uniq):
            if store.get(seq) is not None:
                continue
            if n_api >= budget:
                break
            pdb = fetch_esmfold_api(seq)
            if pdb is not None:
                coords = parse_pdb_ca_coords(pdb)
                if coords is not None and coords.shape[0] >= min(len(seq), 4):
                    store.put(seq, pad_missing_ca(coords, len(seq), seq))
                    n_api += 1
            time.sleep(api_sleep)
            if (i + 1) % 50 == 0:
                print(f"[Prepare] ESMFold progress {i + 1}/{budget} (fetched={n_api})")
    else:
        print("[Prepare] ESMFold API disabled; using offline contact-map MDS reconstruction.")

    tag = esm_model.split('_')[1] + '_' + esm_model.split('_')[2]
    esm_cache_dir = os.path.join(cache_dir, 'esm')
    os.makedirs(esm_cache_dir, exist_ok=True)
    split_name = os.path.splitext(os.path.basename(csv_path))[0]
    npz_path = os.path.join(esm_cache_dir, f'{split_name}_{tag}_L{max_len}.npz')

    featurizer = ESMFeaturizer(esm_model)
    emb, contacts = featurizer.embed_batch(uniq)
    emb16 = emb.astype(np.float16)
    ct16 = contacts.astype(np.float16) if contacts is not None else np.zeros(
        (len(uniq), 1, 1), dtype=np.float16)
    np.savez_compressed(
        npz_path, emb=emb16, contacts=ct16,
        seqs=np.array(uniq, dtype=object),
        has_contacts=np.array(contacts is not None),
        esm_dim=np.array(featurizer.embed_dim))
    size_mb = os.path.getsize(npz_path) / (1024 ** 2)
    print(f"[Prepare] Saved ESM cache: {npz_path} ({size_mb:.1f} MB)")
    del featurizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    has_ct = contacts is not None
    n_mds = n_synth = 0
    for i, seq in enumerate(uniq):
        if store.get(seq) is not None:
            continue
        coords = None
        if has_ct:
            coords = coords_from_contacts(contacts[i, :len(seq), :len(seq)])
            if coords is not None:
                n_mds += 1
        if coords is None:
            coords = coords_synthetic(seq)
            n_synth += 1
        store.put(seq, coords)
    print(f"[Prepare] Structure cache ready: ESMFold={n_api}, MDS={n_mds}, Fallback={n_synth}")
    return npz_path


def build_graph(seq, coords, knn=8, contact_dist=10.0):
    """Build dense 3D spatial and sequential adjacency tensors for a single peptide."""
    L = len(seq)
    node_feats = get_node_features(seq)
    delta = coords[:, None, :] - coords[None, :, :]
    dist = np.linalg.norm(delta, axis=-1)

    adj = np.zeros((L, L), dtype=np.float32)
    np.fill_diagonal(adj, 1.0)
    for offset in (1, 2):
        idx = np.arange(L - offset)
        adj[idx, idx + offset] = 1.0
        adj[idx + offset, idx] = 1.0
    adj[dist < contact_dist] = 1.0

    effective_knn = min(knn, max(L - 1, 0))
    if effective_knn > 0:
        for i in range(L):
            order = np.argsort(dist[i] + np.where(np.arange(L) == i, 1e9, 0.0))
            adj[i, order[:effective_knn]] = 1.0
            adj[order[:effective_knn], i] = 1.0

    seq_adj = np.zeros((L, L), dtype=np.float32)
    np.fill_diagonal(seq_adj, 1.0)
    for offset in (1, 2):
        idx = np.arange(L - offset)
        seq_adj[idx, idx + offset] = 1.0
        seq_adj[idx + offset, idx] = 1.0

    return node_feats, coords.astype(np.float32), adj, seq_adj


class Neuro3DDataset(Dataset):
    """Dataset loading cached ESM-2 embeddings and 3D coordinates into dense residue graphs."""

    def __init__(self, csv_path, esm_npz, max_len=50, knn=8, contact_dist=10.0,
                 cache_dir=None):
        csv_path = resolve_data_path(csv_path)
        df = pd.read_csv(csv_path)
        seq_col = 'Seq' if 'Seq' in df.columns else 'text'
        label_col = 'Label' if 'Label' in df.columns else 'labels'
        self.raw_sequences = df[seq_col].astype(str).tolist()
        self.labels = df[label_col].astype(int).tolist()
        self.max_len = max_len
        self.knn = knn
        self.contact_dist = contact_dist

        data = np.load(esm_npz, allow_pickle=True)
        cache_seqs = [str(s) for s in data['seqs']]
        self.esm_dim = int(data['esm_dim'])
        self._emb = data['emb']
        self._contacts = data['contacts']
        self._has_contacts = bool(data['has_contacts'])
        self._seq2row = {s: i for i, s in enumerate(cache_seqs)}

        self._store = StructureStore(cache_dir or os.path.join(SCRIPT_DIR, 'cache'))
        self.processed_sequences = [clean_sequence(s, max_len) for s in self.raw_sequences]

    def __len__(self):
        return len(self.raw_sequences)

    def _get_coords(self, seq):
        coords = self._store.get(seq)
        if coords is not None:
            return coords
        row = self._seq2row.get(seq)
        if row is not None and self._has_contacts:
            L = len(seq)
            if row < self._contacts.shape[0] and self._contacts.shape[1] >= L:
                coords = coords_from_contacts(
                    np.asarray(self._contacts[row, :L, :L], dtype=np.float32))
        if coords is None:
            coords = coords_synthetic(seq)
        self._store.put(seq, coords)
        return coords

    def __getitem__(self, index):
        seq = self.processed_sequences[index]
        L = len(seq)
        row = self._seq2row[seq]
        esm_emb = np.asarray(self._emb[row, :L], dtype=np.float32)

        coords = self._get_coords(seq)[:L]
        node_feats, coords, adj, seq_adj = build_graph(
            seq, coords, knn=self.knn, contact_dist=self.contact_dist)

        return {
            'seq': seq,
            'raw_seq': self.raw_sequences[index],
            'label': int(self.labels[index]),
            'node_feats': node_feats,
            'coords': coords,
            'adj': adj,
            'seq_adj': seq_adj,
            'esm_emb': esm_emb,
            'mask': np.ones(L, dtype=np.float32),
        }


def collate_3d(batch, max_len=50, esm_dim=1280, node_feat_dim=NODE_FEAT_DIM):
    B = len(batch)
    labels = torch.tensor([b['label'] for b in batch], dtype=torch.long)
    node_feats = torch.zeros(B, max_len, node_feat_dim)
    coords = torch.zeros(B, max_len, 3)
    adj = torch.zeros(B, max_len, max_len)
    seq_adj = torch.zeros(B, max_len, max_len)
    esm_emb = torch.zeros(B, max_len, esm_dim)
    mask = torch.zeros(B, max_len)

    for i, b in enumerate(batch):
        L = len(b['seq'])
        node_feats[i, :L] = torch.from_numpy(b['node_feats'])
        coords[i, :L] = torch.from_numpy(b['coords'])
        adj[i, :L, :L] = torch.from_numpy(b['adj'])
        seq_adj[i, :L, :L] = torch.from_numpy(b['seq_adj'])
        esm_emb[i, :L] = torch.from_numpy(b['esm_emb'])
        mask[i, :L] = 1.0

    seqs = [b['raw_seq'] for b in batch]
    return {
        'seqs': seqs,
        'labels': labels,
        'node_feats': node_feats,
        'coords': coords,
        'adj': adj,
        'seq_adj': seq_adj,
        'esm_emb': esm_emb,
        'mask': mask,
    }


def get_dataloader_3d(csv_path, esm_npz, batch_size=128, shuffle=False,
                      max_len=50, knn=8, contact_dist=10.0, num_workers=0,
                      limit=0):
    dataset = Neuro3DDataset(csv_path, esm_npz, max_len=max_len,
                             knn=knn, contact_dist=contact_dist)
    esm_dim = dataset.esm_dim
    if limit and limit > 0:
        dataset = torch.utils.data.Subset(dataset, range(min(limit, len(dataset))))
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        collate_fn=lambda b: collate_3d(b, max_len=max_len, esm_dim=esm_dim))
    return loader, dataset


def ensure_caches(csv_path, cache_dir, esm_model='esm2_t33_650M_UR50D', max_len=50,
                  use_api=False, api_limit=0):
    tag = esm_model.split('_')[1] + '_' + esm_model.split('_')[2]
    split_name = os.path.splitext(os.path.basename(resolve_data_path(csv_path)))[0]
    npz_path = os.path.join(cache_dir, 'esm', f'{split_name}_{tag}_L{max_len}.npz')
    if os.path.exists(npz_path):
        return npz_path
    print(f"[Cache] Cache missing at {npz_path}; generating ESM-2 and 3D structure caches...")
    return prepare_esm_cache(csv_path, cache_dir, esm_model=esm_model,
                             max_len=max_len, use_api=use_api, api_limit=api_limit)


if __name__ == '__main__':
    demo_csv = os.path.join(SCRIPT_DIR, 'testing_v2.csv')
    cache = os.path.join(SCRIPT_DIR, 'cache')
    npz = ensure_caches(demo_csv, cache, max_len=50)
    loader, ds = get_dataloader_3d(demo_csv, npz, batch_size=4, shuffle=False, limit=8)
    batch = next(iter(loader))
    for k, v in batch.items():
        print(f"  {k}: {v.shape if torch.is_tensor(v) else type(v)}")
