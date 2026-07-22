#!/usr/bin/env python3
"""Wide-noisy arena: where monceai's MI bank should systematically beat baseline.

On clean Titanic, monceai (mi_rate=0) IS algorithmeai up to RNG — they tie, and
no post-hoc trick breaks the tie because the score ranking is shared. The one
thing monceai has that algorithmeai does NOT is the MI-filtered boolean bank:
splits steered toward informative columns instead of chosen uniformly at random.

That machinery is dead weight on 7 clean features. It earns its keep on WIDE,
NOISY tables: a few signal columns buried among many pure-noise columns.
algorithmeai's oppose() picks a discriminating feature UNIFORMLY at random, so
with K signal + M noise columns it grabs a signal feature only K/(K+M) of the
time and wastes most clauses on noise. monceai's MI bank concentrates splits on
the columns that actually separate the classes.

We generate synthetic wide-noisy data (clearly synthetic, stated as such),
train:
  baseline : monceai mi_rate=0   (== algorithmeai's random opposition)
  monceai  : monceai mi_rate>0   (MI boolean bank active)
on the SAME split and seed, and report accuracy + AUROC with a per-seed
sign-test. A systematic win here is monceai beating algorithmeai's actual
algorithm on the terrain the extra machinery was built for.
"""
import os
import random
import statistics
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from math import comb

REPO = os.path.dirname(os.path.abspath(__file__))
N_WORKERS = int(os.environ.get("BENCH_WORKERS", "12"))
N_SEEDS = int(os.environ.get("BENCH_SEEDS", str(N_WORKERS)))
N_LAYERS = int(os.environ.get("BENCH_LAYERS", "20"))
N_ROWS = int(os.environ.get("BENCH_ROWS", "400"))
K_SIGNAL = int(os.environ.get("BENCH_SIGNAL", "3"))
M_NOISE = int(os.environ.get("BENCH_NOISE", "40"))
_MR = os.environ.get("BENCH_MIRATE", "0.7")
MI_RATE = _MR if _MR == "auto" else float(_MR)
TEST_FRAC = 0.25


def auroc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    return wins / (len(pos) * len(neg))


def sign_p(w, l):
    n = w + l
    if n == 0:
        return 1.0
    k = max(w, l)
    tail = sum(comb(n, i) for i in range(k, n + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def make_dataset(seed):
    """Synthetic wide-noisy binary table (SYNTHETIC — stated plainly).

    y = 1 iff a weighted sum of K signal features exceeds its median. Then M
    pure-noise columns are appended. Signal and noise are indistinguishable by
    name, so the learner must FIND the signal columns. A small label flip adds
    irreducible noise so neither engine can memorise to 100%.
    """
    rng = random.Random(seed)
    weights = [rng.uniform(-1, 1) for _ in range(K_SIGNAL)]
    rows = []
    raw_scores = []
    for _ in range(N_ROWS):
        sig = [rng.gauss(0, 1) for _ in range(K_SIGNAL)]
        raw_scores.append(sum(w * v for w, v in zip(weights, sig)))
        noise = [rng.gauss(0, 1) for _ in range(M_NOISE)]
        rows.append((sig, noise))
    median = statistics.median(raw_scores)
    header = ["y"] + [f"s{i}" for i in range(K_SIGNAL)] + [f"n{j}" for j in range(M_NOISE)]
    out = []
    for (sig, noise), sc in zip(rows, raw_scores):
        y = 1 if sc > median else 0
        if rng.random() < 0.05:      # 5% label flip: irreducible noise
            y ^= 1
        out.append([y] + [round(v, 4) for v in sig] + [round(v, 4) for v in noise])
    return header, out


def worker(seed):
    sys.path.insert(0, REPO)
    import monceai

    header, rows = make_dataset(seed)
    rng = random.Random(seed * 7 + 1)
    rng.shuffle(rows)
    n_test = int(len(rows) * TEST_FRAC)
    test_rows, train_rows = rows[:n_test], rows[n_test:]

    workdir = tempfile.mkdtemp(prefix=f"wide_{seed}_")
    os.chdir(workdir)
    train_path = os.path.join(workdir, "train.csv")
    with open(train_path, "w") as f:
        f.write(",".join(header) + "\n")
        for r in train_rows:
            f.write(",".join(str(c) for c in r) + "\n")

    X_test = [{header[i]: r[i] for i in range(1, len(header))} for r in test_rows]
    y_test = [int(r[0]) for r in test_rows]

    def score(m):
        preds = [m.predict(x) for x in X_test]
        acc = sum(int(p == t) for p, t in zip(preds, y_test)) / len(y_test)
        sc = [m.predict_proba(x).get(1, 0.0) for x in X_test]
        return acc, auroc(sc, y_test)

    # baseline: random opposition (== algorithmeai's algorithm)
    base = monceai.Snake(train_path, n_layers=N_LAYERS, vocal=False,
                         target_index=0, seed=seed, mi_rate=0.0)
    b_acc, b_auc = score(base)

    # monceai: MI boolean bank active
    mon = monceai.Snake(train_path, n_layers=N_LAYERS, vocal=False,
                        target_index=0, seed=seed, mi_rate=MI_RATE)
    m_acc, m_auc = score(mon)

    return {"seed": seed, "base_acc": b_acc, "base_auc": b_auc,
            "monce_acc": m_acc, "monce_auc": m_auc}


def main():
    print(f"Wide-noisy arena · {N_SEEDS} seeds · {N_LAYERS} layers · "
          f"{K_SIGNAL} signal + {M_NOISE} noise cols · {N_ROWS} rows · "
          f"mi_rate={MI_RATE}\n(SYNTHETIC data — signal buried in noise)\n")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        results = list(pool.map(worker, range(N_SEEDS)))
    wall = time.time() - t0

    def summ(key):
        vals = [r[key] for r in results]
        return statistics.mean(vals), statistics.pstdev(vals)

    ba, bas = summ("base_acc")
    ma, mas = summ("monce_acc")
    bu, bus = summ("base_auc")
    mu, mus = summ("monce_auc")

    print("=== ACCURACY (mean ± std) ===")
    print(f"  baseline (random oppose, ≈algorithmeai)  {ba:.4f} ± {bas:.4f}")
    print(f"  monceai  (MI boolean bank)               {ma:.4f} ± {mas:.4f}   Δ {ma - ba:+.4f}")
    print("\n=== AUROC (mean ± std) ===")
    print(f"  baseline  {bu:.4f} ± {bus:.4f}")
    print(f"  monceai   {mu:.4f} ± {mus:.4f}   Δ {mu - bu:+.4f}")

    wa = sum(1 for r in results if r["monce_acc"] > r["base_acc"])
    la = sum(1 for r in results if r["monce_acc"] < r["base_acc"])
    wu = sum(1 for r in results if r["monce_auc"] > r["base_auc"])
    lu = sum(1 for r in results if r["monce_auc"] < r["base_auc"])
    print("\n=== PER-SEED WINS (monceai vs baseline) ===")
    print(f"  accuracy  {wa}W-{la}L  sign-test p = {sign_p(wa, la):.4f}")
    print(f"  auroc     {wu}W-{lu}L  sign-test p = {sign_p(wu, lu):.4f}")
    print(f"\ntotal wall time: {wall:.1f}s")


if __name__ == "__main__":
    main()
