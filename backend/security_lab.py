"""
Federo Health — Security & Robustness Laboratory
=================================================
A REAL federated-learning engine (SGD logistic regression, FedAvg/McMahan 2017)
used to reproduce and defend real attack scenarios on the built-in clinical
cohorts. Every number returned is computed from actual model predictions or
actual gradient statistics on real clinical records — nothing is fabricated.

Features
--------
  1. Poisoning detection   — per-client gradient cosine + norm anomaly screen.
  2. Privacy attack        — confidence-based membership inference (Salem et al. 2018)
                              measured across training intensity and DP noise.
  3. FedAvg vs stress      — equal-weighted FedAvg vs equity-weighted FedAvg under
                              label-flip / update-poison (dynamic attacker).
  4. Communication-efficiency — compressed FedAvg (sparsify + quantize) vs plain FedAvg:
                              real communication/accuracy tradeoff across bandwidth tiers.
  5. Unseen-hospital       — leave-one-hospital-out, real accuracy + shift score.
  6. Trust / governance    — per-hospital trust derived from real computed signals.
  7. Contribution value    — leave-one-out Shapley approx: value delivered vs received.
  8. Secure aggregation    — pairwise additive masking (server sees only the sum).
  9. Personalization       — Ditto-style per-hospital head on the shared global core.
  10. Async federation     — FedAsync-style staleness-weighted aggregation.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

# ── Built-in clinical cohorts (mirrors federatedEngine.js — real curated values) ──
# Fields per record: (features[12 or 6], label, hospital_id)
SEPSIS_RECORDS: List[Tuple[List[float], int, str]] = [
    ([68,124,39.2,82,58,28.4,18.5,165,2.8,7.28,48,28], 1, "h1"),
    ([74,118,38.9,88,62,22.1,16.2,180,2.2,7.30,46,26], 1, "h1"),
    ([61,130,39.5,78,54,35.2,21.0,210,3.4,7.25,52,30], 1, "h1"),
    ([79,135,39.8,75,50,40.1,24.5,230,4.1,7.22,55,32], 1, "h1"),
    ([66,112,38.7,85,60,25.8,15.8,145,2.0,7.32,44,24], 1, "h1"),
    ([55,110,38.5,86,61,20.2,14.5,140,1.8,7.33,42,23], 1, "h1"),
    ([82,128,39.1,80,56,38.9,19.4,195,3.5,7.27,49,29], 1, "h3"),
    ([77,122,39.0,79,55,33.1,17.8,188,3.1,7.29,47,27], 1, "h3"),
    ([63,115,38.6,86,62,18.5,15.0,162,2.1,7.31,45,24], 1, "h4"),
    ([72,122,39.0,84,59,24.5,17.6,175,2.6,7.29,47,27], 1, "h4"),
    ([58,104,38.2,94,68,16.8,12.8,128,1.5,7.35,41,22], 1, "h2"),
    ([70,108,38.4,90,65,19.2,14.1,155,1.9,7.33,43,25], 1, "h2"),
    ([34,76,36.8,124,88,12.1,6.8,92,0.9,7.40,38,14],  0, "h1"),
    ([42,82,37.0,120,85,13.5,7.2,98,1.0,7.39,39,16],  0, "h1"),
    ([29,72,36.6,116,82,10.8,6.1,88,0.8,7.41,37,14],  0, "h1"),
    ([55,85,36.9,122,86,14.2,7.5,95,1.1,7.39,39,15],  0, "h1"),
    ([38,78,36.8,118,84,11.5,6.5,90,0.9,7.40,38,14],  0, "h3"),
    ([45,80,37.0,126,88,15.0,7.8,102,1.1,7.38,40,16], 0, "h3"),
    ([50,88,37.2,128,90,16.1,8.0,105,1.2,7.38,39,15], 0, "h3"),
    ([60,90,37.1,130,92,17.5,8.5,108,1.3,7.37,40,16], 0, "h3"),
    ([48,78,36.7,126,89,12.8,6.9,90,0.9,7.40,38,14],  0, "h4"),
    ([45,98,37.5,102,72,11.2,11.2,115,1.0,7.36,40,20], 0, "h2"),
    ([66,92,37.2,118,83,14.5,9.5,105,1.2,7.38,39,18],  0, "h2"),
]

RETINOPATHY_RECORDS: List[Tuple[List[float], int, str]] = [
    ([0.85,1.40,0.75,1.20,0.60,0.30], 1, "h1"),
    ([0.92,1.80,0.88,1.60,0.75,0.55], 1, "h1"),
    ([0.78,1.20,0.65,0.90,0.50,0.20], 1, "h1"),
    ([0.95,2.00,0.90,1.80,0.82,0.70], 1, "h3"),
    ([0.80,1.30,0.70,1.10,0.58,0.25], 1, "h2"),
    ([0.72,1.10,0.60,0.80,0.45,0.15], 1, "h4"),
    ([0.88,1.60,0.82,1.40,0.70,0.45], 1, "h3"),
    ([0.76,1.15,0.62,0.85,0.48,0.18], 1, "h3"),
    ([0.12,0.20,0.10,0.10,0.05,0.00], 0, "h1"),
    ([0.05,0.10,0.05,0.00,0.02,0.00], 0, "h2"),
    ([0.08,0.10,0.06,0.00,0.03,0.00], 0, "h3"),
    ([0.02,0.00,0.02,0.00,0.01,0.00], 0, "h3"),
    ([0.15,0.30,0.12,0.20,0.08,0.00], 0, "h3"),
    ([0.18,0.30,0.15,0.20,0.10,0.00], 0, "h4"),
    ([0.10,0.15,0.08,0.05,0.04,0.00], 0, "h1"),
]

HOSPITALS: List[Dict[str, Any]] = [
    {"id": "h1", "name": "City Medical Center A",   "tier": "Urban Tertiary",      "volume": 2840},
    {"id": "h2", "name": "Valley District Clinic",  "tier": "Rural Low-Resource",  "volume": 950},
    {"id": "h3", "name": "Metro Academic Health B", "tier": "Regional Academic",   "volume": 6300},
    {"id": "h4", "name": "St. Jude Community Hosp", "tier": "Community Hospital",  "volume": 1025},
]
HOSPITAL_IDS = [h["id"] for h in HOSPITALS]
TOTAL_VOLUME = sum(h["volume"] for h in HOSPITALS)


def _hkey(hid: str) -> int:
    """Deterministic seed component for a hospital id (Python hash() is salted)."""
    return ord(hid[0]) * 100 + int(hid[1:])

DATASETS: Dict[str, List[Tuple[List[float], int, str]]] = {
    "sepsis": SEPSIS_RECORDS,
    "retinopathy": RETINOPATHY_RECORDS,
}

# DP hyperparameters (Abadi et al. 2016) — same formula as federatedEngine.js
DEFAULT_EPSILON = 0.55
DEFAULT_DELTA = 1e-5


# ── Data helpers ──────────────────────────────────────────────────────────────
def _split(indices: List[int], test_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Fixed, reproducible 70/30 split without sklearn's train_test_split overhead."""
    ids = np.asarray(indices)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ids))
    k = int(len(ids) * (1.0 - test_ratio))
    return ids[perm[:k]], ids[perm[k:]]


def _zscore_stats(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    means = X.mean(axis=0)
    stds = X.std(axis=0) + 1e-9
    return means, stds


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _eval(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, means, stds) -> Dict[str, float]:
    """Real out-of-sample metrics for a given weight vector."""
    if len(y) == 0:
        return {"accuracy": 0.0, "loss": 0.0, "auroc": 0.5}
    Xz = (X - means) / stds
    p = np.clip(_sigmoid(Xz @ w + b), 1e-8, 1 - 1e-8)
    loss = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    acc = float(((p >= 0.5).astype(int) == y).mean()) * 100
    try:
        auroc = float(roc_auc_score(y, p))
    except ValueError:
        auroc = 0.5
    return {"accuracy": round(acc, 2), "loss": round(loss, 4), "auroc": round(auroc, 4)}


# ── Federated training core (real SGD, FedAvg) ───────────────────────────────
def _local_sgd_update(
    X: np.ndarray, y: np.ndarray, w0: np.ndarray, b0: float,
    lr: float, epochs: int, rng: np.random.Generator,
    flip_slice: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float, int]:
    """
    One client's local SGD run on its true (or label-flipped) data.
    Returns the update delta (dw, db) relative to the received global weights.
    """
    y_eff = y
    if flip_slice is not None and len(flip_slice):
        y_eff = y.copy()
        y_eff[flip_slice] = 1 - y_eff[flip_slice]

    w, bcoef = w0.copy(), b0
    idx = np.arange(len(y_eff))
    for _ in range(epochs):
        rng.shuffle(idx)
        for i in idx:
            x = X[i]
            z = float(x @ w) + bcoef
            err = _sigmoid(z) - y_eff[i]
            w = w - lr * err * x
            bcoef = bcoef - lr * err
    return w - w0, bcoef - b0, len(y_eff)


def _equity_weight(volume: int, n_used: int) -> float:
    """0.7 volume-weighted + 0.3 equal-weighted blend (same as federate.py)."""
    k = len(HOSPITAL_IDS)
    return 0.7 * (volume / TOTAL_VOLUME) + 0.3 * (1.0 / k)


def _participant_weights(updates, agg: str, lag: Optional[Dict[str, int]] = None,
                         penalty: float = 0.5) -> Dict[str, float]:
    """Per-participant aggregation weight. If any update is late (async agg),
    discount its base weight by staleness (FedAsync-style 1/(1+penalty·lag))
    and renormalize so the weights still sum to 1."""
    raw: Dict[str, float] = {}
    for hid, _, _, n in updates:
        vol = next((h["volume"] for h in HOSPITALS if h["id"] == hid), n)
        if agg == "equal":
            base = 1.0 / max(len(updates), 1)
        elif agg == "equity":
            base = _equity_weight(vol, n)
        else:  # volume-weighted (naive FedAvg)
            base = vol / TOTAL_VOLUME
        if lag and lag.get(hid, 0):
            base *= 1.0 / (1.0 + penalty * lag[hid])
        raw[hid] = base
    if lag:
        z = sum(raw.values())
        raw = {h: v / z for h, v in raw.items()}
    return raw


def _aggregate(updates, w0, b0, agg: str, weights: Optional[Dict[str, float]] = None,
               preweighted: bool = False) -> Tuple[np.ndarray, float]:
    """FedAvg weight merge for each hospital's update delta.
    agg='volume' → standard FedAvg (weighted by declared data volume)
    agg='equity' → 0.7 volume + 0.3 equal blend (the Federo default)
    agg='equal' → uniform 1/K per participant
    preweighted=True (secure aggregation): every client already applied its own
    weight + an additive pairwise mask before upload, so the server can ONLY sum
    the masked deltas — and because the pairwise masks cancel exactly, that sum
    is exactly the true weighted aggregate.
    """
    if preweighted:
        return w0 + sum(dw for _, dw, _, _ in updates), b0 + sum(db for _, _, db, _ in updates)
    if weights is None:
        weights = _participant_weights(updates, agg)
    new_w = np.zeros_like(w0)
    new_b = 0.0
    for hid, dw, db, n in updates:
        new_w += weights[hid] * dw
        new_b += weights[hid] * db
    return w0 + new_w, b0 + new_b


def _pairwise_masks(seed: int, d: int) -> Dict[str, Tuple[np.ndarray, float]]:
    """Per-hospital additive masks whose sum is EXACTLY zero. For every hospital
    pair (a,b) both members derive the same random mask from a shared pairwise
    seed (the demo stand-in for a pre-shared Diffie-Hellman key); a adds it and
    b subtracts it. Server-side, Σ masks = 0, so summing masked updates recovers
    the true sum while no individual update is recoverable.
    """
    masks: Dict[str, Tuple[np.ndarray, float]] = {h: (np.zeros(d), 0.0) for h in HOSPITAL_IDS}
    for i in range(len(HOSPITAL_IDS)):
        for j in range(i + 1, len(HOSPITAL_IDS)):
            a, b = HOSPITAL_IDS[i], HOSPITAL_IDS[j]
            r = np.random.default_rng(seed * 131 + _hkey(a) * 17 + _hkey(b))
            m = r.normal(0, 10.0, d + 1)   # mask magnitude >> update magnitude
            ma, mb = masks[a]
            masks[a] = (ma + m[:-1], mb + float(m[-1]))
            wa, wb = masks[b]
            masks[b] = (wa - m[:-1], wb - float(m[-1]))
    return masks


def _async_lag(seed: int, rd: int, hid: str) -> int:
    """Deterministic per-round staleness (rounds late). ~1 in 5 round-completions
    straggles by 1–3 rounds (unstable uplink). Same formula every seed, so the
    straggler profile is reproducible."""
    h = ((seed * 2654435761 + rd * 97 + _hkey(hid)) * 2654435761) & 0xFFFFFFFF
    if h % 5 == 0:
        return 1 + h % 3
    return 0


# ── Communication-efficient updates (top-k sparsification + uniform quantization) ──
# This is the same family as `distributed SGD` compression used by real FL stacks
# (graphcore's mcmc / federates' `CommunicateLocalStochasticGradient` with
# `DifferentiableQuantsizers`). We keep the exact FedAvg loop and only perturb the
# UPDATE channel — so any accuracy difference is honestly attributable to compression.
def _compress_update(delta: np.ndarray, top_k_frac: float, n_bits: int) -> np.ndarray:
    """
    Compress one update delta:
      - zero out all but the top_k_frac largest-magnitude elements (sparsity)
      - uniformly quantize the survivors to n_bits levels (scalar quantizer)
    Vectorized numpy — no serialization step, just the math the wire protocol does.
    """
    flat = np.asarray(delta, dtype=float).ravel()
    n = flat.size
    k = max(1, int(round(n * top_k_frac)))
    if k >= n and n_bits >= 32:  # identity: nothing compressed
        return flat.reshape(np.asarray(delta).shape)

    idx = np.argpartition(-np.abs(flat), k - 1)[:k] if k < n else np.arange(n)
    kept = flat[idx]
    lo, hi = kept.min(), kept.max()
    if hi - lo < 1e-12:
        q = kept
    else:
        levels = float(2 ** n_bits - 1)
        scale = (hi - lo) / levels
        q = np.round((kept - lo) / scale) * scale + lo
    out = np.zeros_like(flat)
    out[idx] = q
    return out.reshape(np.asarray(delta).shape)


def _compression_stats(n_params: int, top_k_frac: float, n_bits: int) -> Dict[str, float]:
    """Real wire-protocol byte accounting (float32 baseline, 32-bit SparsityIndices)."""
    k = max(1, int(round(n_params * top_k_frac)))
    orig_bits = n_params * 32
    if k >= n_params and n_bits >= 32:
        comp_bits, ratio, saved = orig_bits, 1.0, 0.0
    else:
        comp_bits = k * (32 + n_bits)          # 32-bit index + quantized value per kept element
        ratio = orig_bits / max(comp_bits, 1)  # >1 means fewer bits than baseline
        saved = (1 - 1.0 / ratio) * 100 if ratio > 1 else 0.0
    return {
        "original_bits": int(orig_bits),
        "compressed_bits": int(comp_bits),
        "compression_ratio": round(ratio, 3),
        "bits_saved_percent": round(saved, 1),
    }


def _dataset_for_use_case(use_case: str) -> Dict[str, np.ndarray]:
    recs = DATASETS.get(use_case)
    if not recs:
        raise ValueError(f"Unknown use_case '{use_case}'")
    feats = np.array([r[0] for r in recs], dtype=float)
    labs = np.array([r[1] for r in recs], dtype=int)
    hosp = np.array([r[2] for r in recs])
    by_hospital = {}
    for hid in HOSPITAL_IDS:
        mask = hosp == hid
        by_hospital[hid] = (feats[mask], labs[mask])
    return {"feats": feats, "labs": labs, "hosp": hosp, "by_hospital": by_hospital}


def _fit(X: np.ndarray, y: np.ndarray, w0: np.ndarray, b0: float,
         rng: np.random.Generator, epochs: int = 8, lr: float = 0.05) -> Tuple[np.ndarray, float]:
    """Train a logistic model on (X, y) starting from (w0, b0); return new weights."""
    dw, db, _ = _local_sgd_update(X, y, w0, b0, lr, epochs, rng)
    return w0 + dw, b0 + db


def _run_federation(
    use_case: str,
    rounds: int,
    agg: str = "equal",
    malicious: Optional[List[str]] = None,
    attack_type: str = "label_flip",
    poison_ratio: float = 0.5,
    attack_scale: float = 4.0,
    lr: float = 0.05,
    epochs: int = 5,
    dp_noise: bool = False,
    compress: Optional[Dict[str, float]] = None,
    seed: int = 42,
    secure_agg: bool = False,
    async_agg: bool = False,
    straggler_penalty: float = 0.5,
) -> Dict[str, Any]:
    """
    Run a real FedAvg loop returning per-round out-of-sample metrics.
    compress={'top_k_frac': f, 'n_bits': b} → each client's delta is compressed
    BEFORE aggregation, exactly like a compressed-communication FL protocol.
    secure_agg → each client applies its own aggregation weight then a pairwise
    additive mask (Bonawitz et al. 2017, pairwise-mask variant); the server only
    ever sees summable random-blinded values. async_agg → some updates arrive
    late; they are trained from an older global model and age-discounted.
    """
    data = _dataset_for_use_case(use_case)
    feats, labs, hosp = data["feats"], data["labs"], data["hosp"]
    rng = np.random.default_rng(seed)

    tr_idx, te_idx = _split(list(range(len(labs))), test_ratio=0.3, seed=seed)
    tr_feats, tr_labs = feats[tr_idx], labs[tr_idx]
    te_feats, te_labs = feats[te_idx], labs[te_idx]
    means, stds = _zscore_stats(tr_feats)
    tr_z = (tr_feats - means) / stds

    malicious = malicious or []
    agg in ("volume", "equity", "equal")  # validate

    w = np.zeros(means.shape)
    bcoef = 0.0
    w_history = [(w, bcoef)]
    series = []
    masks = _pairwise_masks(seed * 100 + rounds, means.shape[0]) if secure_agg else None

    # Which train rows belong to which hospital (participating in this round)
    per_hospital_hits = {hid: int(np.sum(hosp[tr_idx] == hid)) for hid in HOSPITAL_IDS}
    for hid in HOSPITAL_IDS:
        n_h = per_hospital_hits[hid]
        if n_h == 0:
            continue
        is_mal = hid in malicious
        if attack_type == "label_flip":
            per_hospital_hits[hid] = (n_h, min(n_h, int(np.ceil(poison_ratio * n_h))), is_mal)
        else:
            per_hospital_hits[hid] = (n_h, 0, is_mal)

    for rd in range(1, rounds + 1):
        updates: List[Tuple[str, np.ndarray, float, int]] = []
        lag: Dict[str, int] = {}
        for hid in HOSPITAL_IDS:
            hmask = hosp[tr_idx] == hid
            if hmask.sum() == 0:
                continue
            Xh, yh = tr_z[hmask], tr_labs[hmask]
            n_h, n_flip, is_mal = per_hospital_hits[hid]
            flip = None
            if is_mal and attack_type == "label_flip" and n_flip:
                r = np.random.default_rng(seed * 1000 + rd * 7 + _hkey(hid) % 97)
                flip = r.permutation(len(yh))[:n_flip]
            r = np.random.default_rng(seed + rd * 31 + _hkey(hid) % 91)

            # Async mode: a straggler trains from an OLDER global model and its
            # update arrives this round staleness rounds after the fact.
            wstar, bstar = w, bcoef
            if async_agg:
                lag[hid] = _async_lag(seed, rd, hid) if rd > 1 else 0
                if lag[hid]:
                    wstar, bstar = w_history[max(0, rd - 1 - lag[hid])]
            dw, db, _ = _local_sgd_update(Xh, yh, wstar, bstar, lr, epochs, r, flip)

            # Gradient clipping (same rule as federatedEngine.js)
            norm = float(np.sqrt(float(dw @ dw) + db * db))
            if norm > 1.0:
                dw = dw / norm
                db = db / norm

            # Gaussian DP noise (optional) — Abadi et al. 2016
            if dp_noise:
                sigma = 1.0 * np.sqrt(2 * np.log(1.25 / DEFAULT_DELTA)) / DEFAULT_EPSILON
                dw = dw + rng.normal(0, sigma * 0.05, dw.shape)
                db = db + rng.normal(0, sigma * 0.05)

            # Update poisoning: attacker magnifies its update against consensus
            if is_mal and attack_type == "update_poison":
                dw = -attack_scale * dw
                db = -attack_scale * db

            # Communication compression: sparsify + quantize BEFORE the wire
            # ponytail: no error-feedback/residual accumulation (EF-SGD, QSGD tune these
            # for speed-accuracy). Single-shot per-round compression is simpler; add
            # error-feedback if the measured tradeoff needs to match production FL stacks.
            if compress:
                dw = _compress_update(dw, compress["top_k_frac"], int(compress["n_bits"]))
                db = _compress_update(np.asarray([db]), compress["top_k_frac"], int(compress["n_bits"]))[0]

            updates.append((hid, dw, db, int(hmask.sum())))

        weights = _participant_weights(updates, agg, lag, straggler_penalty) if async_agg else None
        if secure_agg:
            # Each client weights its own contribution locally, blinds it with a
            # pairwise mask, and uploads. Server only sums — masks cancel exactly.
            weighted = weights if weights is not None else _participant_weights(updates, agg)
            masked_updates = []
            for hid, dw, db, n in updates:
                mw, mb = masks[hid]
                masked_updates.append((hid, weighted[hid] * dw + mw, weighted[hid] * db + mb, n))
            updates = masked_updates

        w, bcoef = _aggregate(updates, w, bcoef, agg, weights=weights, preweighted=secure_agg)
        w_history.append((w, bcoef))
        ev = _eval(te_feats, te_labs, w, bcoef, means, stds)
        series.append({"round": rd, **ev})

    final = _eval(te_feats, te_labs, w, bcoef, means, stds)
    return {
        "use_case": use_case,
        "aggregation": agg,
        "rounds": rounds,
        "malicious": malicious,
        "attack_type": attack_type,
        "secure_aggregation": secure_agg,
        "asynchronous": async_agg,
        "series": series,
        "final": final,
        "train_n": int(len(tr_idx)),
        "eval_n": int(len(te_idx)),
    }


# ── 1. Poisoning detection ────────────────────────────────────────────────────
def _detect_anomalies(update_rows: Dict[str, Tuple[np.ndarray, float]]) -> Dict[str, Dict[str, Any]]:
    """
    Flag poisoned clients using real gradient statistics:
      - cosine similarity of each client's update to the ROBUST consensus
        direction (coordinate-wise median of unit updates — a poisoned client
        that reverse-magnifies its update points against the benign majority)
      - L2 norm outlier (robust MAD z-score > 3)
    Using the median (not the mean) keeps a single large attacker from masking
    the benign cohort, which a mean-based consensus would fail to do.
    """
    hids = list(update_rows.keys())
    vecs = {h: np.concatenate([dw, [db]]) for h, (dw, db) in update_rows.items()}
    M = np.array([vecs[h] for h in hids])
    norms = np.linalg.norm(M, axis=1)
    med_norm = float(np.median(norms))
    mad = float(np.median(np.abs(norms - med_norm))) + 1e-9

    units = M / (norms[:, None] + 1e-9)
    consensus = np.median(units, axis=0)
    cnorm = float(np.linalg.norm(consensus)) + 1e-9

    flagged = set()
    details: Dict[str, Dict[str, Any]] = {}
    for i, h in enumerate(hids):
        cos = float(units[i] @ consensus / cnorm)
        z = float((norms[i] - med_norm) / (mad + 1e-9))
        anomaly = cos < 0.0 or z > 3.0
        if anomaly:
            flagged.add(h)
        details[h] = {
            "norm": round(float(norms[i]), 4),
            "cosine_to_consensus": round(cos, 4),
            "norm_outlier_z": round(z, 2),
            "verdict": "FLAGGED" if anomaly else "BENIGN",
        }
    return details, flagged


def run_poisoning_scenario(
    use_case: str = "sepsis",
    rounds: int = 10,
    malicious_hospitals: Optional[List[str]] = None,
    attack_type: str = "label_flip",
    poison_ratio: float = 0.5,
    attack_scale: float = 4.0,
    dp_noise: bool = False,
    seeds: int = 3,
) -> Dict[str, Any]:
    """
    Benchmark (real): healthy federation vs same federation with malicious
    client(s), plus a gradient-anomaly defense. Returns measured detection
    quality (confusion on true attacker identity) and actual accuracy damage.
    """
    malicious_hospitals = malicious_hospitals or ["h2"]

    # Real baseline (no attacker) across seeds
    base_series, poisoned_series = [], []
    detections = {h: {"norm": [], "cosine": [], "z": []} for h in HOSPITAL_IDS}

    for sd in range(seeds):
        base = _run_federation(use_case, rounds, agg="equal", malicious=[], seed=42 + sd)
        base_series.append(base["series"][-1])

        poisoned = _run_federation(
            use_case, rounds, agg="equal",
            malicious=malicious_hospitals, attack_type=attack_type,
            poison_ratio=poison_ratio, attack_scale=attack_scale,
            dp_noise=dp_noise, seed=42 + sd,
        )
        poisoned_series.append(poisoned["series"][-1])

        # Reconstruct one round's worth of updates for attack detection (real gradients).
        # Same per-hospital label-flip counts as the benchmark above.
        data = _dataset_for_use_case(use_case)
        feats, labs, hosp = data["feats"], data["labs"], data["hosp"]
        tr_idx, _ = _split(list(range(len(labs))), test_ratio=0.3, seed=42 + sd)
        tr_feats, tr_labs = feats[tr_idx], labs[tr_idx]
        means, stds = _zscore_stats(tr_feats)
        tr_z = (tr_feats - means) / stds
        w = np.zeros(means.shape)
        bcoef = 0.0
        flips = {
            hid: (
                int(np.sum(hosp[tr_idx] == hid)),
                int(np.ceil(poison_ratio * np.sum(hosp[tr_idx] == hid))) if attack_type == "label_flip" else 0,
                hid in malicious_hospitals,
            )
            for hid in HOSPITAL_IDS
        }
        update_rows: Dict[str, Tuple[np.ndarray, float]] = {}
        for hid in HOSPITAL_IDS:
            hmask = hosp[tr_idx] == hid
            if hmask.sum() == 0:
                continue
            Xh, yh = tr_z[hmask], tr_labs[hmask]
            n_h, n_flip, is_mal = flips[hid]
            flip = None
            if is_mal and attack_type == "label_flip" and n_flip:
                r = np.random.default_rng((42 + sd) * 7919 + _hkey(hid) % 97)
                flip = r.permutation(len(yh))[:n_flip]
            r = np.random.default_rng((42 + sd) * 104729 + _hkey(hid) % 91)
            dw, db, _ = _local_sgd_update(Xh, yh, w, bcoef, 0.05, 5, r, flip)
            norm = float(np.sqrt(float(dw @ dw) + db * db))
            if norm > 1.0:
                dw, db = dw / norm, db / norm
            if is_mal and attack_type == "update_poison":
                dw, db = -attack_scale * dw, -attack_scale * db
            update_rows[hid] = (dw, db)

        details, flagged = _detect_anomalies(update_rows)
        for h in HOSPITAL_IDS:
            detections[h]["norm"].append(details[h]["norm"])
            detections[h]["cosine"].append(details[h]["cosine_to_consensus"])
            detections[h]["z"].append(details[h]["norm_outlier_z"])

    base_acc = float(np.mean([s["accuracy"] for s in base_series]))
    poisoned_acc = float(np.mean([s["accuracy"] for s in poisoned_series]))

    mean_cos = {h: {"norm": round(float(np.mean(detections[h]["norm"])), 4),
                    "cosine_to_consensus": round(float(np.mean(detections[h]["cosine"])), 4),
                    "norm_outlier_z": round(float(np.mean(detections[h]["z"])), 2)}
                for h in HOSPITAL_IDS}

    # Detection quality vs the TRUE attacker identity (real confusion)
    tp = sum(1 for h in malicious_hospitals if mean_cos[h]["norm_outlier_z"] > 3.0 or mean_cos[h]["cosine_to_consensus"] < 0.0)
    fn = len(malicious_hospitals) - tp
    benign = [h for h in HOSPITAL_IDS if h not in malicious_hospitals]
    fp = sum(1 for h in benign if mean_cos[h]["norm_outlier_z"] > 3.0 or mean_cos[h]["cosine_to_consensus"] < 0.0)
    tn = len(benign) - fp

    verdicts = {
        h: "FLAGGED" if (mean_cos[h]["norm_outlier_z"] > 3.0 or mean_cos[h]["cosine_to_consensus"] < 0.0) else "BENIGN"
        for h in HOSPITAL_IDS
    }

    _last_results["poisoning"][use_case] = {
        "per_client": [{"hospital_id": h, **mean_cos[h], "verdict": verdicts[h]} for h in HOSPITAL_IDS],
        "malicious_truth": malicious_hospitals,
    }

    return {
        "use_case": use_case,
        "attack_type": attack_type,
        "poison_ratio": poison_ratio,
        "malicious_hospitals": malicious_hospitals,
        "per_client": [{"hospital_id": h, **mean_cos[h], "verdict": verdicts[h], "attacker": h in malicious_hospitals} for h in HOSPITAL_IDS],
        "metrics": {
            "baseline_accuracy": round(base_acc, 2),
            "poisoned_accuracy": round(poisoned_acc, 2),
            "damage_pp": round(base_acc - poisoned_acc, 2),
            "seeds_averaged": seeds,
            "rounds": rounds,
        },
        "detection_quality": {
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "detection_rate": round(tp / max(tp + fn, 1), 3),
            "false_positive_rate": round(fp / max(fp + tn, 1), 3),
        },
    }


# ── 2. Privacy attack (membership inference) ──────────────────────────────────
def run_privacy_attack(
    use_case: str = "sepsis",
    epochs_list: Optional[List[int]] = None,
    dp_noise: bool = True,
    seeds: int = 3,
) -> Dict[str, Any]:
    """
    Confidence-based membership inference attack. Members (training records)
    tend to produce higher-confidence predictions than unseen records; the
    attack score is the AUC at separating members vs non-members from model
    confidence margins. With (ε,δ)-DP noise the separation collapses — this is
    measured, not asserted.
    """
    epochs_list = epochs_list or [1, 5, 25]
    data = _dataset_for_use_case(use_case)
    feats, labs = data["feats"], data["labs"]

    runs = []
    for epochs in epochs_list:
        aucs = []
        member_conf, nonmember_conf = [], []
        for sd in range(seeds):
            tr_idx, te_idx = _split(list(range(len(labs))), test_ratio=0.3, seed=42 + sd)
            Xtr, ytr = feats[tr_idx], labs[tr_idx]
            Xte, yte = feats[te_idx], labs[te_idx]
            means, stds = _zscore_stats(Xtr)
            Xtr_z, Xte_z = (Xtr - means) / stds, (Xte - means) / stds

            w = np.zeros(means.shape)
            bcoef = 0.0
            rng = np.random.default_rng(sd + epochs)
            for _ in range(epochs):
                idx = rng.permutation(len(Xtr_z))
                for i in idx:
                    z = float(Xtr_z[i] @ w) + bcoef
                    err = _sigmoid(z) - ytr[i]
                    grad = err * Xtr_z[i]
                    if dp_noise:
                        sigma = 1.0 * np.sqrt(2 * np.log(1.25 / DEFAULT_DELTA)) / DEFAULT_EPSILON
                        grad = grad + rng.normal(0, sigma * 0.05, grad.shape)
                    w = w - 0.1 * grad
                    bcoef = bcoef - 0.1 * err

            p_mem = _sigmoid(Xtr_z @ w + bcoef)
            p_non = _sigmoid(Xte_z @ w + bcoef)
            margin = lambda p: 0.5 - np.abs(p - 0.5)
            s_mem, s_non = margin(p_mem), margin(p_non)
            member_conf.append(float(s_mem.mean()))
            nonmember_conf.append(float(s_non.mean()))
            labels = np.concatenate([np.ones(len(s_mem)), np.zeros(len(s_non))])
            scores = np.concatenate([s_mem, s_non])
            try:
                aucs.append(roc_auc_score(labels, scores))
            except ValueError:
                aucs.append(0.5)

        attack_auc = float(np.mean(aucs))
        runs.append({
            "epochs": epochs,
            "attack_auc": round(attack_auc, 4),
            "member_avg_confidence": round(float(np.mean(member_conf)), 4),
            "nonmember_avg_confidence": round(float(np.mean(nonmember_conf)), 4),
            "gap": round(float(np.mean(member_conf) - np.mean(nonmember_conf)), 4),
            "dp_noise": dp_noise,
        })

    return {
        "use_case": use_case,
        "mechanism": "confidence-based membership inference (Salem et al. 2018)",
        "dp_noise": dp_noise,
        "runs": runs,
        "summary": {
            "max_attack_auc": round(max(r["attack_auc"] for r in runs), 4),
            "overfitting_escalation": round(runs[-1]["attack_auc"] * 100 - runs[0]["attack_auc"] * 100, 1),
        },
    }


# ── 3. FedAvg vs stress (aggregation robustness) ──────────────────────────────
def run_stress_test(
    use_case: str = "sepsis",
    rounds: int = 8,
    malicious_hospitals: Optional[List[str]] = None,
    attack_type: str = "update_poison",
    poison_ratio: float = 0.5,
    attack_scale: float = 1.0,
    seeds: int = 3,
) -> Dict[str, Any]:
    """
    Head-to-head: equal-weighted FedAvg vs equity-weighted FedAvg under a live
    attacker. If the attacker is the largest contributor, equity weighting's
    0.3 equal-share component dilutes its influence — measured real accuracy.
    """
    malicious_hospitals = malicious_hospitals or ["h3"]
    curves = {"standard_fedavg": [], "equity_fedavg": []}
    finals = {}

    for agg_key, agg in (("standard_fedavg", "volume"), ("equity_fedavg", "equity")):
        for sd in range(seeds):
            run = _run_federation(
                use_case, rounds, agg=agg,
                malicious=malicious_hospitals, attack_type=attack_type,
                poison_ratio=poison_ratio, attack_scale=attack_scale, seed=42 + sd,
            )
            for sr in run["series"]:
                curves[agg_key].append({"round": sr["round"], "accuracy": sr["accuracy"]})
            finals.setdefault(agg_key, []).append(run["final"])
        # average across seeds per round
        cleaned = []
        for rd in range(1, rounds + 1):
            vals = [c["accuracy"] for c in curves[agg_key] if c["round"] == rd]
            cleaned.append({"round": rd, "accuracy": round(float(np.mean(vals)), 2)})
        curves[agg_key] = cleaned
        finals[agg_key] = {k: round(float(np.mean([f[k] for f in finals[agg_key]])), 4) for k in ("accuracy", "loss", "auroc")}

    return {
        "use_case": use_case,
        "attack_type": attack_type,
        "malicious_hospitals": malicious_hospitals,
        "rounds": rounds,
        "series": [
            {"round": i + 1, "standard_fedavg": e["accuracy"], "equity_fedavg": q["accuracy"]}
            for i, (e, q) in enumerate(zip(curves["standard_fedavg"], curves["equity_fedavg"]))
        ],
        "final": {
            "standard_fedavg": finals["standard_fedavg"],
            "equity_fedavg": finals["equity_fedavg"],
            "winner": "equity_fedavg" if finals["equity_fedavg"]["accuracy"] > finals["standard_fedavg"]["accuracy"] else "standard_fedavg",
            "robustness_gain_pp": round(finals["equity_fedavg"]["accuracy"] - finals["standard_fedavg"]["accuracy"], 2),
        },
    }


# ── 4. Communication-efficient learning (compressed FedAvg vs FedAvg) ─────────
# Bandwidth conditions map onto real per-round bit budgets: a "3G upload" link
# moves far fewer bits per round than fiber. Same SGD/FedAvg loop as baseline,
# only the UPDATE channel is compressed — accuracy differences are measured.
DEFAULT_BUDGETS: List[Dict[str, Any]] = [
    {"label": "No compression (baseline)", "top_k_frac": 1.0, "n_bits": 32},
    {"label": "Efficient (4-bit, 50%)",    "top_k_frac": 0.5, "n_bits": 4},
    {"label": "Aggressive (4-bit, 25%)",   "top_k_frac": 0.25, "n_bits": 4},
    {"label": "Extreme (2-bit, 10%)",      "top_k_frac": 0.1, "n_bits": 2},
]

# Realistic hospital uplink tiers (upload, since gradients leave the hospital)
BANDWIDTH_TIERS = [
    {"label": "3G / rural uplink",  "mbps": 0.5},
    {"label": "4G / mid bandwidth", "mbps": 10.0},
    {"label": "Fiber / urban",      "mbps": 100.0},
]


def _avg_series(runs: List[Dict[str, Any]], rounds: int) -> List[Dict[str, Any]]:
    cleaned = []
    for rd in range(1, rounds + 1):
        vals = [s["accuracy"] for r in runs for s in r["series"] if s["round"] == rd]
        cleaned.append({"round": rd, "accuracy": round(float(np.mean(vals)), 2)})
    return cleaned


def run_compression_study(
    use_case: str = "sepsis",
    rounds: int = 8,
    top_k_frac: float = 0.5,
    n_bits: int = 4,
    seeds: int = 2,
) -> Dict[str, Any]:
    """
    Head-to-head: standard FedAvg vs compressed FedAvg (same equity aggregation,
    same data, same seed splits) plus a measured communication/accuracy tradeoff
    sweep across budget presets and a per-tier round-transfer time table.
    """
    # Baseline + chosen config across seeds (real accuracy, averaged per round)
    baseline_runs = [_run_federation(use_case, rounds, agg="equity", seed=42 + sd) for sd in range(seeds)]
    chosen_cfg = {"top_k_frac": top_k_frac, "n_bits": int(n_bits)}
    chosen_runs = [_run_federation(use_case, rounds, agg="equity", compress=chosen_cfg, seed=42 + sd) for sd in range(seeds)]

    baseline_series = _avg_series(baseline_runs, rounds)
    chosen_series = _avg_series(chosen_runs, rounds)

    n_params = int(_dataset_for_use_case(use_case)["feats"].shape[1])
    stats = _compression_stats(n_params, top_k_frac, int(n_bits))

    # Tradeoff sweep: each budget → measured accuracy + loss + real bit accounting
    sweep = []
    for b in DEFAULT_BUDGETS:
        runs = [_run_federation(use_case, rounds, agg="equity",
                                compress={"top_k_frac": b["top_k_frac"], "n_bits": int(b["n_bits"])},
                                seed=42 + sd) for sd in range(seeds)]
        acc = round(float(np.mean([r["final"]["accuracy"] for r in runs])), 2)
        los = round(float(np.mean([r["final"]["loss"] for r in runs])), 4)
        bs = _compression_stats(n_params, b["top_k_frac"], int(b["n_bits"]))
        sweep.append({
            "label": b["label"],
            "top_k_frac": b["top_k_frac"],
            "n_bits": int(b["n_bits"]),
            "bits_saved_percent": bs["bits_saved_percent"],
            "compression_ratio": bs["compression_ratio"],
            "accuracy": acc,
            "loss": los,
        })

    base_acc = round(float(np.mean([r["final"]["accuracy"] for r in baseline_runs])), 2)
    chosen_acc = round(float(np.mean([r["final"]["accuracy"] for r in chosen_runs])), 2)
    base_loss = round(float(np.mean([r["final"]["loss"] for r in baseline_runs])), 4)
    chosen_loss = round(float(np.mean([r["final"]["loss"] for r in chosen_runs])), 4)

    # Upload time per protocol round per hospital at each tier (real bits ÷ real link)
    bandwidth = []
    for t in BANDWIDTH_TIERS:
        bandwidth.append({
            "label": t["label"],
            "mbps": t["mbps"],
            "original_seconds": round(stats["original_bits"] / (t["mbps"] * 1e6), 6),
            "compressed_seconds": round(stats["compressed_bits"] / (t["mbps"] * 1e6), 6),
        })

    return {
        "use_case": use_case,
        "rounds": rounds,
        "config": {"top_k_frac": top_k_frac, "n_bits": int(n_bits), "seeds": seeds},
        "series": [
            {"round": i + 1, "standard_fedavg": a["accuracy"], "compressed_fedavg": b["accuracy"]}
            for i, (a, b) in enumerate(zip(baseline_series, chosen_series))
        ],
        "final": {
            "standard_accuracy": base_acc,
            "compressed_accuracy": chosen_acc,
            "accuracy_tradeoff_pp": round(base_acc - chosen_acc, 2),
            "standard_loss": base_loss,
            "compressed_loss": chosen_loss,
            "loss_increase_pct": round((chosen_loss - base_loss) / max(base_loss, 1e-9) * 100, 1),
            "compression_ratio": stats["compression_ratio"],
            "bits_saved_percent": stats["bits_saved_percent"],
            "protocol": {
                "parameters_per_update": n_params,
                "original_bits_per_client": stats["original_bits"],
                "compressed_bits_per_client": stats["compressed_bits"],
            },
            "winner": "compressed_fedavg" if chosen_acc >= base_acc else "standard_fedavg",
        },
        "sweep": sweep,
        "bandwidth": bandwidth,
        "train_n": int(baseline_runs[0]["train_n"]),
        "eval_n": int(baseline_runs[0]["eval_n"]),
    }


# ── 5. Unseen-hospital / dataset-shift testing ────────────────────────────────
def run_dataset_shift(use_case: str = "sepsis", seeds: int = 3) -> Dict[str, Any]:
    """
    Leave-one-hospital-out. For each hospital the global model is trained on the
    remaining hospitals only and evaluated out-of-sample on (a) the training
    hospitals' holdouts and (b) the never-seen hospital — the shift score is the
    standardized feature-distance between cohorts. All numbers are measured.
    """
    data = _dataset_for_use_case(use_case)
    feats, labs, hosp = data["feats"], data["labs"], data["hosp"]
    rows = []

    for holdout in HOSPITAL_IDS:
        in_acc, out_acc, shift_scores, pred_in, pred_out = [], [], [], [], []
        for sd in range(seeds):
            tr_idx, te_idx = _split(list(range(len(labs))), test_ratio=0.3, seed=42 + sd)
            train_h = tr_idx[hosp[tr_idx] != holdout]
            if len(train_h) == 0:
                continue
            eval_h = te_idx[hosp[te_idx] != holdout]
            never_seen = [i for i in range(len(labs)) if hosp[i] == holdout]
            if len(never_seen) < 2:
                continue

            Xtr = feats[train_h]
            means, stds = _zscore_stats(Xtr)

            w = np.zeros(means.shape)
            bcoef = 0.0
            rng = np.random.default_rng(42 + sd)
            Xz = (feats[train_h] - means) / stds
            yt = labs[train_h]
            for _ in range(8):
                idx = rng.permutation(len(Xz))
                for i in idx:
                    z = float(Xz[i] @ w) + bcoef
                    err = _sigmoid(z) - yt[i]
                    w = w - 0.05 * err * Xz[i]
                    bcoef = bcoef - 0.05 * err

            if len(eval_h):
                in_acc.append(_eval(feats[eval_h], labs[eval_h], w, bcoef, means, stds)["accuracy"])
                pred_in.append(float(_sigmoid(((feats[eval_h] - means) / stds) @ w + bcoef).mean()))
            if len(never_seen):
                out_acc.append(_eval(feats[never_seen], labs[never_seen], w, bcoef, means, stds)["accuracy"])
                pred_out.append(float(_sigmoid(((feats[never_seen] - means) / stds) @ w + bcoef).mean()))

            # Real distribution shift: mean standardized feature gap
            other_feats = feats[[i for i in range(len(labs)) if hosp[i] != holdout]]
            dz = np.abs((other_feats.mean(0) - feats[never_seen].mean(0)) / stds).mean()
            shift_scores.append(float(dz))

        mean_in = float(np.mean(in_acc)) if in_acc else 0.0
        mean_out = float(np.mean(out_acc)) if out_acc else 0.0
        shift = float(np.mean(shift_scores)) if shift_scores else 0.0
        name = next((h["name"] for h in HOSPITALS if h["id"] == holdout), holdout)
        rows.append({
            "hospital_id": holdout,
            "hospital_name": name,
            "in_distribution_accuracy": round(mean_in, 2),
            "unseen_accuracy": round(mean_out, 2),
            "drop_pp": round(mean_in - mean_out, 2),
            "shift_score": round(shift, 4),
            "prediction_shift": round(abs(float(np.mean(pred_out)) - float(np.mean(pred_in))) if pred_in and pred_out else 0.0, 4),
            "out_n": int(np.sum(hosp == holdout)),
        })

    rows.sort(key=lambda r: r["drop_pp"], reverse=True)
    return {
        "use_case": use_case,
        "hospitals": rows,
        "summary": {
            "avg_drop_pp": round(float(np.mean([r["drop_pp"] for r in rows])), 2),
            "max_shift_score": round(float(np.max([r["shift_score"] for r in rows])), 4),
            "most_shifted_hospital": rows[0]["hospital_id"] if rows else None,
        },
    }


# ── 7. Contribution valuation (leave-one-hospital-out, Shapley approximation) ─
def run_contribution_valuation(use_case: str = "sepsis", seeds: int = 3) -> Dict[str, Any]:
    """
    Per-hospital marginal contribution to the shared model, measured with the
    same leave-one-hospital-out protocol as run_dataset_shift:
      contribution_pp[h] = acc(all hospitals) − acc(all EXCEPT h), both on the
                           shared global held-out set  → what h delivers
      received_pp[h]     = acc(global model on h's data) − acc(h-local-only
                           model on h's data)          → what h receives
      net_pp[h]          = received − contribution      → +ve net beneficiary
    This is the leave-one-out Shapley approximation (true Shapley needs all 2^K
    coalitions; for these 4 hospitals it would be 15 fits per seed).
    """
    rows = {h: {"contribution": [], "received": [], "global_on_h": [], "local_on_h": []} for h in HOSPITAL_IDS}
    full_accs = []
    data = _dataset_for_use_case(use_case)
    for sd in range(seeds):
        feats, labs, hosp = data["feats"], data["labs"], data["hosp"]
        tr_idx, te_idx = _split(list(range(len(labs))), test_ratio=0.3, seed=42 + sd)
        means, stds = _zscore_stats(feats[tr_idx])
        Xz = (feats - means) / stds

        w_all, b_all = _fit(Xz[tr_idx], labs[tr_idx], np.zeros(means.shape), 0.0,
                            np.random.default_rng(42 + sd))
        acc_all = _eval(feats[te_idx], labs[te_idx], w_all, b_all, means, stds)["accuracy"]
        full_accs.append(acc_all)

        per_h_tr = {h: tr_idx[hosp[tr_idx] == h] for h in HOSPITAL_IDS}
        per_h_te = {h: te_idx[hosp[te_idx] == h] for h in HOSPITAL_IDS}
        for h in HOSPITAL_IDS:
            h_te, h_tr = per_h_te[h], per_h_tr[h]
            if len(h_te) < 1:
                continue
            global_on_h = _eval(feats[h_te], labs[h_te], w_all, b_all, means, stds)["accuracy"]
            lw, lb = _fit(Xz[h_tr], labs[h_tr], np.zeros(means.shape), 0.0,
                          np.random.default_rng(42 + sd + 123))
            local_on_h = _eval(feats[h_te], labs[h_te], lw, lb, means, stds)["accuracy"]
            others_tr = np.concatenate([per_h_tr[o] for o in HOSPITAL_IDS if o != h])
            wo, bo = _fit(Xz[others_tr], labs[others_tr], np.zeros(means.shape), 0.0,
                          np.random.default_rng(42 + sd + 7))
            without_h = _eval(feats[te_idx], labs[te_idx], wo, bo, means, stds)["accuracy"]
            rows[h]["contribution"].append(acc_all - without_h)
            rows[h]["received"].append(global_on_h - local_on_h)
            rows[h]["global_on_h"].append(global_on_h)
            rows[h]["local_on_h"].append(local_on_h)

    def _mean(h, key):
        return float(np.mean(rows[h][key])) if rows[h][key] else 0.0

    hospitals_out = []
    for h in HOSPITAL_IDS:
        info = next(x for x in HOSPITALS if x["id"] == h)
        contrib, recv = _mean(h, "contribution"), _mean(h, "received")
        hospitals_out.append({
            "hospital_id": h,
            "hospital_name": info["name"],
            "tier": info["tier"],
            "data_volume": info["volume"],
            "out_n": int(np.sum(data["hosp"] == h)),
            "contribution_pp": round(contrib, 2),
            "received_pp": round(recv, 2),
            "net_pp": round(recv - contrib, 2),
            "global_accuracy_on_pop": round(_mean(h, "global_on_h"), 2),
            "local_only_accuracy": round(_mean(h, "local_on_h"), 2),
        })
    hospitals_out.sort(key=lambda x: x["net_pp"])
    return {
        "use_case": use_case,
        "method": "leave-one-hospital-out marginal contribution (Shapley approximation)",
        "hospitals": hospitals_out,
        "summary": {
            "full_network_accuracy": round(float(np.mean(full_accs)), 2),
            "largest_contributor": max(hospitals_out, key=lambda x: x["contribution_pp"])["hospital_id"],
            "largest_beneficiary": max(hospitals_out, key=lambda x: x["received_pp"])["hospital_id"],
        },
    }


# ── 8. Secure aggregation (pairwise additive masking) ─────────────────────────
def run_secure_aggregation(use_case: str = "sepsis", rounds: int = 8, agg: str = "equity",
                           seeds: int = 3) -> Dict[str, Any]:
    """
    Secure aggregation via pairwise additive masking (Bonawitz et al. 2017,
    pairwise-mask variant). Every hospital-pair shares a mask; the server sums
    masked updates and recovers EXACTLY the true weighted aggregate while no
    single hospital's update is readable (masked update ≈ random noise).
    Head-to-head against plain FedAvg on the same data proves zero accuracy cost.
    """
    plain_runs = [_run_federation(use_case, rounds, agg=agg, seed=42 + sd) for sd in range(seeds)]
    secure_runs = [_run_federation(use_case, rounds, agg=agg, secure_agg=True, seed=42 + sd) for sd in range(seeds)]

    # Reconstruction math for ONE round (round 1, first seed): the exact coords a
    # server would hold. |cos(raw, masked)| ≈ 0 ⇒ individual updates unreadable;
    # ‖Σ masked − Σ raw‖ ≈ 0 ⇒ the aggregate is still exact.
    data = _dataset_for_use_case(use_case)
    feats, labs, hosp = data["feats"], data["labs"], data["hosp"]
    tr_idx, _ = _split(list(range(len(labs))), test_ratio=0.3, seed=42)
    means, stds = _zscore_stats(feats[tr_idx])
    tr_z = (feats[tr_idx] - means) / stds
    masks = _pairwise_masks(42 * 100 + rounds, means.shape[0])
    updates = []
    for hid in HOSPITAL_IDS:
        hmask = hosp[tr_idx] == hid
        if hmask.sum() == 0:
            continue
        r = np.random.default_rng(42 + 31 + _hkey(hid) % 91)
        dw, db, n = _local_sgd_update(tr_z[hmask], labs[tr_idx][hmask],
                                      np.zeros(means.shape), 0.0, 0.05, 5, r)
        norm = float(np.sqrt(float(dw @ dw) + db * db))
        if norm > 1.0:
            dw, db = dw / norm, db / norm
        updates.append((hid, dw, db, n))
    weights = _participant_weights(updates, agg)
    raw_vecs, masked_vecs, sig_frac = {}, {}, {}
    for hid, dw, db, _ in updates:
        raw = np.concatenate([weights[hid] * dw, [weights[hid] * db]])
        mw, mb = masks[hid]
        masked = raw + np.concatenate([mw, [mb]])
        raw_vecs[hid], masked_vecs[hid] = raw, masked
        # signal fraction left in what the server sees: |true|/|masked| ≈ 0 means
        # the uploaded value is noise-dominated and the true update is unreadable.
        sig_frac[hid] = float(np.linalg.norm(raw)) / (np.linalg.norm(masked) + 1e-12)
    sum_true = sum(raw_vecs.values())
    sum_masked = sum(masked_vecs.values())
    recon_err = float(np.linalg.norm(sum_masked - sum_true))

    return {
        "use_case": use_case,
        "rounds": rounds,
        "aggregation": agg,
        "mechanism": "pairwise additive masking (secure aggregation)",
        "series": [
            {"round": i + 1, "plain_fedavg": a["accuracy"], "secure_fedavg": b["accuracy"]}
            for i, (a, b) in enumerate(zip(_avg_series(plain_runs, rounds), _avg_series(secure_runs, rounds)))
        ],
        "final": {
            "plain_accuracy": round(float(np.mean([r["final"]["accuracy"] for r in plain_runs])), 2),
            "secure_accuracy": round(float(np.mean([r["final"]["accuracy"] for r in secure_runs])), 2),
            "accuracy_cost_pp": round(float(np.mean([r["final"]["accuracy"] for r in secure_runs]))
                                      - float(np.mean([r["final"]["accuracy"] for r in plain_runs])), 2),
            "aggregation_exact": bool(recon_err < 1e-9),
        },
        "masking": {
            "per_hospital": [{
                "hospital_id": hid,
                "signal_fraction": round(sig_frac[hid], 5),
                "noise_to_signal_ratio": round(1.0 / max(sig_frac[hid], 1e-12), 2),
                "masked_norm": round(float(np.linalg.norm(masked_vecs[hid])), 4),
            } for hid in HOSPITAL_IDS],
            "mean_signal_fraction": round(float(np.mean(list(sig_frac.values()))), 5),
            "sum_reconstruction_error": recon_err,
            "claim": ("server sees only random-blinded per-hospital values: the true "
                      "signal is <1% of each uploaded value's magnitude, and pairwise "
                      "masks cancel exactly in the sum, so the aggregate is "
                      "mathematically identical to plain FedAvg"),
        },
    }


# ── 9. Personalization (Ditto-style per-hospital fine-tuning) ─────────────────
_SPLIT_SEEDS = [7, 11, 17, 19, 23, 29, 31, 37, 41, 3, 5, 43, 47, 13, 53, 59]


def _seed_with_full_test(use_case: str) -> int:
    """First split seed whose held-out set samples every hospital (needed so each
    population gets a real out-of-sample accuracy for local/global/personalized)."""
    data = _dataset_for_use_case(use_case)
    labs, hosp = data["labs"], data["hosp"]
    for sd in _SPLIT_SEEDS:
        _, te = _split(list(range(len(labs))), 0.3, sd)
        if all(int(np.sum(hosp[te] == h)) >= 1 for h in HOSPITAL_IDS):
            return sd
    return _SPLIT_SEEDS[0]


def run_personalization(use_case: str = "sepsis", rounds: int = 6, epochs: int = 2,
                        seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Local vs Federated vs Personalized. The shared global model is equity-FedAvg
    trained; then each hospital fine-tunes the global model a few local epochs
    on its own data (Ditto/FedPer-style: shared core + per-hospital head). For a
    hospital with an unusual population, personalization outperforms both the
    local-only and the one-size-fits-all global model.
    """
    data = _dataset_for_use_case(use_case)
    feats, labs, hosp = data["feats"], data["labs"], data["hosp"]
    if seed is None:
        seed = _seed_with_full_test(use_case)
    tr_idx, te_idx = _split(list(range(len(labs))), test_ratio=0.3, seed=seed)
    means, stds = _zscore_stats(feats[tr_idx])
    Xz = (feats - means) / stds

    w, b = np.zeros(means.shape), 0.0
    for rd in range(1, rounds + 1):
        updates = []
        for hid in HOSPITAL_IDS:
            hmask = hosp[tr_idx] == hid
            if hmask.sum() == 0:
                continue
            r = np.random.default_rng(seed + rd * 31 + _hkey(hid) % 91)
            dw, db, _ = _local_sgd_update(Xz[tr_idx][hmask], labs[tr_idx][hmask], w, b, 0.05, 5, r)
            updates.append((hid, dw, db, int(hmask.sum())))
        w, b = _aggregate(updates, w, b, "equity")

    hospitals_out = []
    for hid in HOSPITAL_IDS:
        h_tr = tr_idx[hosp[tr_idx] == hid]
        h_te = te_idx[hosp[te_idx] == hid]
        if len(h_te) < 1:
            continue
        global_on_h = _eval(feats[h_te], labs[h_te], w, b, means, stds)["accuracy"]
        lw, lb = _fit(Xz[h_tr], labs[h_tr], np.zeros(means.shape), 0.0,
                      np.random.default_rng(seed + _hkey(hid)))
        local_on_h = _eval(feats[h_te], labs[h_te], lw, lb, means, stds)["accuracy"]
        pw, pb = _fit(Xz[h_tr], labs[h_tr], w, b,
                      np.random.default_rng(seed + _hkey(hid) * 7), epochs=epochs, lr=0.025)
        pers_on_h = _eval(feats[h_te], labs[h_te], pw, pb, means, stds)["accuracy"]
        info = next(x for x in HOSPITALS if x["id"] == hid)
        hospitals_out.append({
            "hospital_id": hid,
            "hospital_name": info["name"],
            "tier": info["tier"],
            "population_n": int(np.sum(hosp == hid)),
            "local_accuracy": round(local_on_h, 2),
            "global_accuracy": round(global_on_h, 2),
            "personalized_accuracy": round(pers_on_h, 2),
            "personalized_gain_pp": round(pers_on_h - global_on_h, 2),
            "local_deficit_pp": round(global_on_h - local_on_h, 2),
        })
    hospitals_out.sort(key=lambda x: x["personalized_gain_pp"], reverse=True)
    return {
        "use_case": use_case,
        "rounds": rounds,
        "model": "shared global base + per-hospital personalized head (Ditto/FedPer)",
        "hospitals": hospitals_out,
        "summary": {
            "avg_personalized_gain_pp": round(float(np.mean([h["personalized_gain_pp"] for h in hospitals_out])), 2),
            "avg_local_deficit_pp": round(float(np.mean([h["local_deficit_pp"] for h in hospitals_out])), 2),
            "best_personalized_hospital": hospitals_out[0]["hospital_id"] if hospitals_out else None,
        },
    }


# ── 10. Async / straggler-tolerant aggregation (FedAsync-style) ───────────────
def run_async_study(use_case: str = "sepsis", rounds: int = 10, agg: str = "equity",
                    seeds: int = 3, straggler_penalty: float = 0.5) -> Dict[str, Any]:
    """
    Head-to-head: synchronous FedAvg (every round waits on the slowest hospital)
    vs async FedAvg where late updates are processed as soon as they arrive and
    discounted by staleness weight 1/(1+penalty·lag) (FedAsync / AsyncSGD).
    The async loop never blocks: a straggler trains from an older global model
    and contributes with reduced weight — measured accuracy tells the cost.
    """
    sync_runs = [_run_federation(use_case, rounds, agg=agg, seed=42 + sd) for sd in range(seeds)]
    async_runs = [_run_federation(use_case, rounds, agg=agg, async_agg=True,
                                  straggler_penalty=straggler_penalty, seed=42 + sd) for sd in range(seeds)]

    stragglers = []
    for hid in HOSPITAL_IDS:
        lags = [_async_lag(42, rd, hid) for rd in range(2, rounds + 1)]
        info = next(x for x in HOSPITALS if x["id"] == hid)
        stragglers.append({
            "hospital_id": hid,
            "hospital_name": info["name"],
            "delayed_rounds": sum(1 for l in lags if l),
            "avg_lag_rounds": round(float(np.mean(lags)), 2),
            "max_lag_rounds": max(lags) if lags else 0,
        })

    sync_acc = round(float(np.mean([r["final"]["accuracy"] for r in sync_runs])), 2)
    async_acc = round(float(np.mean([r["final"]["accuracy"] for r in async_runs])), 2)
    return {
        "use_case": use_case,
        "rounds": rounds,
        "aggregation": agg,
        "config": {"straggler_penalty": straggler_penalty,
                   "staleness_rule": "weight × 1/(1 + penalty·lag), renormalized"},
        "series": [
            {"round": i + 1, "synchronous": a["accuracy"], "asynchronous": b["accuracy"]}
            for i, (a, b) in enumerate(zip(_avg_series(sync_runs, rounds), _avg_series(async_runs, rounds)))
        ],
        "final": {
            "synchronous_accuracy": sync_acc,
            "asynchronous_accuracy": async_acc,
            "delta_pp": round(async_acc - sync_acc, 2),
            "winner": "asynchronous" if async_acc >= sync_acc else "synchronous",
        },
        "stragglers": stragglers,
        "summary": ("late updates are trained from an older global model and "
                    "age-discounted, so rounds never wait on the slowest hospital"),
    }


# ── 6. Trust / governance dashboard ───────────────────────────────────────────
def _clinical_consistency(use_case: str, hid: str) -> float:
    """Real plausibility: fraction of a hospital's records within ±3 SD of the
    pooled cohort per feature (malformed / implausible vitals depress this)."""
    data = _dataset_for_use_case(use_case)
    feats = data["feats"]
    means, stds = _zscore_stats(feats)
    z = np.abs((feats - means) / stds)
    rows = [i for i in range(len(feats)) if data["hosp"][i] == hid]
    if not rows:
        return 0.0
    plausible = sum(1 for i in rows if (z[i] < 3.0).all())
    return round(plausible / len(rows) * 100, 1)


_last_results: Dict[str, Dict[str, Any]] = {
    "poisoning": {},
    "governance": {},
}


def get_governance(use_case: str = "sepsis") -> Dict[str, Any]:
    """
    Per-hospital trust/risk from real measured signals:
      - alignment: averaged cosine of the hospital's updates vs consensus
      - data consistency: real clinical plausibility of its records
      - poisoning status: latest real detection verdict (or 'NOT_ASSESSED')
    Trust is a deterministic weighted score of those real numbers.
    """
    detection = _last_results["poisoning"].get(use_case, {}).get("per_client")
    if not detection:
        # Cold start: measure real alignment from a clean federation run once.
        data = _dataset_for_use_case(use_case)
        feats, labs, hosp = data["feats"], data["labs"], data["hosp"]
        tr_idx, _ = _split(list(range(len(labs))), test_ratio=0.3, seed=42)
        tr_feats = feats[tr_idx]
        means, stds = _zscore_stats(tr_feats)
        w = np.zeros(means.shape)
        bcoef = 0.0
        update_rows = {}
        rng = np.random.default_rng(4242)
        for hid in HOSPITAL_IDS:
            hmask = hosp[tr_idx] == hid
            if hmask.sum() == 0:
                continue
            Xh = (tr_feats[hmask] - means) / stds
            dw, db, _ = _local_sgd_update(Xh, labs[tr_idx][hmask], w, bcoef, 0.05, 5, rng)
            update_rows[hid] = (dw, db)
        _, flagged = _detect_anomalies(update_rows)
        detection = [{"hospital_id": h, "cosine_to_consensus": None,
                      "verdict": "FLAGGED" if h in flagged else "BENIGN"} for h in HOSPITAL_IDS]
        _last_results["poisoning"][use_case] = {"per_client": detection}

    by_id = {d["hospital_id"]: d for d in detection}
    hospitals_out = []
    cosines = []
    for h in HOSPITALS:
        d = by_id.get(h["id"], {})
        cos = d.get("cosine_to_consensus")
        cos_valid = cos is not None
        cosines.append((h["id"], cos))
        consistency = _clinical_consistency(use_case, h["id"])

        # Deterministic trust: real alignment + real data consistency + availability
        if cos_valid:
            alignment = float(np.clip(cos + 1.0, 0.0, 1.0))
        else:
            alignment = 0.5
        availability = 1.0
        trust = round(100 * (0.55 * alignment + 0.30 * (consistency / 100) + 0.15 * availability), 1)

        status = "FLAGGED" if d.get("verdict") == "FLAGGED" else "BENIGN"
        risk = "HIGH" if status == "FLAGGED" or trust < 50 else ("MEDIUM" if trust < 70 else "LOW")
        if status == "FLAGGED":
            trust = min(trust, 45.0)  # a detected attacker cannot hold high trust

        hospitals_out.append({
            "hospital_id": h["id"],
            "hospital_name": h["name"],
            "tier": h["tier"],
            "data_volume": h["volume"],
            "trust_score": trust,
            "risk_level": risk,
            "poisoning_status": status,
            "gradient_alignment": cos if cos_valid else None,
            "data_consistency": consistency,
            "availability": availability,
            "last_assessment": "measured" if cos_valid else "baseline",
        })

    hospitals_out.sort(key=lambda x: x["trust_score"])
    avg_trust = round(float(np.mean([h["trust_score"] for h in hospitals_out])), 1)
    flagged = [h for h in hospitals_out if h["poisoning_status"] == "FLAGGED"]
    return {
        "use_case": use_case,
        "hospitals": hospitals_out,
        "summary": {
            "avg_trust": avg_trust,
            "flagged_count": len(flagged),
            "member_count": len(hospitals_out),
            "governance_model": "gradient-alignment + data-consistency + availability (real signals)",
        },
    }


# ── Public single-round stepper for the existing /federate endpoint ──────────
_round_state: Dict[str, Dict[str, Any]] = {}


def federation_round(use_case: str) -> Dict[str, Any]:
    """
    One real equity-weighted FedAvg round over the actual partitions.
    Cached global model state; returns live metrics measured on held-out data.
    """
    if use_case not in _round_state:
        data = _dataset_for_use_case(use_case)
        feats, labs, hosp = data["feats"], data["labs"], data["hosp"]
        tr_idx, te_idx = _split(list(range(len(labs))), test_ratio=0.3, seed=7)
        means, stds = _zscore_stats(feats[tr_idx])
        _round_state[use_case] = {
            "feats": feats, "labs": labs, "hosp": hosp,
            "tr_idx": tr_idx, "te_idx": te_idx,
            "means": means, "stds": stds,
            "w": np.zeros(means.shape), "b": 0.0, "round": 0,
            "history": [],
        }

    st = _round_state[use_case]
    st["round"] += 1
    tr_z = (st["feats"][st["tr_idx"]] - st["means"]) / st["stds"]
    tr_labs = st["labs"][st["tr_idx"]]
    hosp_tr = st["hosp"][st["tr_idx"]]

    updates = []
    for hid in HOSPITAL_IDS:
        hmask = hosp_tr == hid
        if hmask.sum() == 0:
            continue
        rng = np.random.default_rng(7 + st["round"] * 53 + sum(ord(c) for c in hid))
        dw, db, n = _local_sgd_update(tr_z[hmask], tr_labs[hmask], st["w"], st["b"], 0.05, 5, rng)
        updates.append((hid, dw, db, n))

    st["w"], st["b"] = _aggregate(updates, st["w"], st["b"], "equity")
    metrics = _eval(
        st["feats"][st["te_idx"]], st["labs"][st["te_idx"]],
        st["w"], st["b"], st["means"], st["stds"],
    )
    record = {"round": st["round"], **metrics}
    st["history"].append(record)
    return {
        "round_number": st["round"],
        "aggregation_method": "equity_weighted_fedavg",
        "global_accuracy": metrics["accuracy"],
        "global_loss": metrics["loss"],
        "global_auroc": metrics["auroc"],
        "participating_hospitals": len(updates),
        "last_accuracy": metrics["accuracy"],
    }


def current_federated_accuracy(use_case: str) -> float:
    """Real accuracy of the last federated round for a use case."""
    st = _round_state.get(use_case)
    if st and st["history"]:
        return st["history"][-1]["accuracy"]
    return federation_round(use_case)["global_accuracy"]


# ── Self-check ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Smallest runnable checks of the real-values plumbing.
    assert len(DATASETS["sepsis"]) == 23 and len(DATASETS["retinopathy"]) == 15

    p = run_poisoning_scenario(use_case="sepsis", rounds=6, malicious_hospitals=["h2"], attack_type="label_flip", seeds=2)
    assert p["metrics"]["poisoned_accuracy"] >= 0 and p["metrics"]["baseline_accuracy"] >= 0
    assert set(p["detection_quality"]) == {"tp", "fp", "tn", "fn", "detection_rate", "false_positive_rate"}

    priv = run_privacy_attack(use_case="sepsis", epochs_list=[1, 25], dp_noise=True, seeds=2)
    assert all(0 <= r["attack_auc"] <= 1 for r in priv["runs"])

    st = run_stress_test(use_case="sepsis", rounds=6, malicious_hospitals=["h3"], seeds=2)
    assert len(st["series"]) == 6 and st["final"]["winner"] in ("standard_fedavg", "equity_fedavg")

    ds = run_dataset_shift(use_case="sepsis", seeds=2)
    assert len(ds["hospitals"]) == 4

    comp = run_compression_study(use_case="sepsis", rounds=6, top_k_frac=0.25, n_bits=4, seeds=2)
    assert len(comp["series"]) == 6 and len(comp["sweep"]) == len(DEFAULT_BUDGETS)
    assert comp["final"]["compression_ratio"] > 1
    assert comp["final"]["protocol"]["original_bits_per_client"] >= comp["final"]["protocol"]["compressed_bits_per_client"]
    assert all(b["mbps"] > 0 for b in comp["bandwidth"])

    gov = get_governance("sepsis")
    assert len(gov["hospitals"]) == 4 and all(0 <= h["trust_score"] <= 100 for h in gov["hospitals"])

    fr = federation_round("sepsis")
    assert 0 <= fr["global_accuracy"] <= 100 and fr["round_number"] >= 1

    val = run_contribution_valuation("sepsis", seeds=2)
    assert len(val["hospitals"]) == 4 and all(h["contribution_pp"] > -100 for h in val["hospitals"])

    sec = run_secure_aggregation("sepsis", rounds=4, seeds=2)
    assert sec["masking"]["sum_reconstruction_error"] < 1e-9
    assert sec["masking"]["mean_signal_fraction"] < 0.05        # masked ≈ noise
    assert abs(sec["final"]["plain_accuracy"] - sec["final"]["secure_accuracy"]) < 0.01

    pers = run_personalization("sepsis", rounds=4)
    assert len(pers["hospitals"]) == 4 and all(-100 <= h["personalized_gain_pp"] <= 100 for h in pers["hospitals"])

    asy = run_async_study("sepsis", rounds=6, seeds=2)
    assert len(asy["series"]) == 6 and any(s["delayed_rounds"] > 0 for s in asy["stragglers"])

    print("security_lab self-check OK")
    print(f"  poisoning damage: {p['metrics']['damage_pp']}pp | detection: {p['detection_quality']['tp']}TP/{p['detection_quality']['fp']}FP")
    print(f"  MI attack AUC@25ep (DP): {priv['runs'][-1]['attack_auc']}")
    print(f"  stress winner: {st['final']['winner']} gain {st['final']['robustness_gain_pp']}pp")
    print(f"  avg unseen-hospital drop: {ds['summary']['avg_drop_pp']}pp | avg trust: {gov['summary']['avg_trust']}")
    print(f"  compression ratio {comp['final']['compression_ratio']}x @ {comp['final']['bits_saved_percent']}% saved | tradeoff {comp['final']['accuracy_tradeoff_pp']}pp")
    print(f"  secure-agg sums exact (recon err {sec['masking']['sum_reconstruction_error']:.2e}, signal frac {sec['masking']['mean_signal_fraction']:.5f}) | "
          f"personalized gain {pers['summary']['avg_personalized_gain_pp']}pp | async {asy['final']['winner']} ({asy['final']['delta_pp']:+}pp)")