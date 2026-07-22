#!/usr/bin/env python3
"""Compute-fair bagging experiment: does averaging B seeded models beat one?

algorithmeai runs ONE unseeded randomized pass. monceai is seedable, which
unlocks bagging: train B independent models on different seeds and average their
probability vectors. Bagging reduces the variance of a high-variance randomized
learner, which reliably lifts BOTH accuracy and AUROC.

The fairness guard: total layer budget is held CONSTANT. A single model gets
TOTAL layers; a B-bag ensemble gets B models of TOTAL//B layers each — the same
number of learned clauses either way. So any win is not "more compute", it is a
better use of the SAME compute. That is a win algorithmeai's one-shot design
cannot reproduce.

Baseline (bags=1) is monceai with mi_rate=0, which is algorithmeai up to RNG.
"""
import csv
import os
import random
import statistics
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from math import comb

REPO = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV = os.path.join(REPO, "titanic", "train.csv")
N_WORKERS = int(os.environ.get("BENCH_WORKERS", "12"))
N_SEEDS = int(os.environ.get("BENCH_SEEDS", str(N_WORKERS)))
TOTAL_LAYERS = int(os.environ.get("BENCH_TOTAL", "30"))
BAG_SIZES = [int(x) for x in os.environ.get("BENCH_BAGS", "1,2,3,5,6").split(",")]
TEST_FRAC = 0.20

NUM_INT = {"Pclass", "SibSp", "Parch"}
NUM_FLOAT = {"Age", "Fare"}
DROP = {"PassengerId", "Survived"}


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


def _typed(row):
    x = {}
    for k, v in row.items():
        if k in DROP:
            continue
        if k in NUM_INT:
            x[k] = int(v) if v.strip() else 0
        elif k in NUM_FLOAT:
            x[k] = float(v) if v.strip() else 0.0
        else:
            x[k] = v
    return x


def _write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def worker(seed):
    """For one split: score every bag size at equal total-layer budget."""
    sys.path.insert(0, REPO)
    import monceai

    with open(TRAIN_CSV) as f:
        reader = csv.reader(f)
        header = next(reader)
        all_rows = list(reader)
    rng = random.Random(seed)
    rng.shuffle(all_rows)
    n_test = int(len(all_rows) * TEST_FRAC)
    test_rows, train_rows = all_rows[:n_test], all_rows[n_test:]

    workdir = tempfile.mkdtemp(prefix=f"bag_{seed}_")
    os.chdir(workdir)
    train_path = os.path.join(workdir, "train.csv")
    _write_csv(train_path, header, train_rows)

    dict_rows = [dict(zip(header, r)) for r in test_rows]
    X_test = [_typed(r) for r in dict_rows]
    y_true = [int(r["Survived"]) for r in dict_rows]

    out = {"seed": seed}
    for bags in BAG_SIZES:
        per = max(1, TOTAL_LAYERS // bags)
        # Train `bags` independent models, each seeded distinctly & reproducibly.
        models = []
        for b in range(bags):
            m = monceai.Snake(train_path, n_layers=per, vocal=False,
                              target_index=1, excluded_features_index=[0],
                              seed=seed * 1000 + b)
            models.append(m)
        # Average probability of class 1 across the ensemble.
        scores, preds = [], []
        for x in X_test:
            p1 = sum(m.predict_proba(x).get(1, 0.0) for m in models) / bags
            scores.append(p1)
            preds.append(1 if p1 >= 0.5 else 0)
        acc = sum(int(p == t) for p, t in zip(preds, y_true)) / len(y_true)
        out[f"bag{bags}_acc"] = acc
        out[f"bag{bags}_auc"] = auroc(scores, y_true)
        out[f"bag{bags}_layers"] = per * bags
    return out


def main():
    print(f"Compute-fair bagging · {N_SEEDS} seeds · total≈{TOTAL_LAYERS} layers · "
          f"bag sizes {BAG_SIZES}\n")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        results = list(pool.map(worker, range(N_SEEDS)))
    wall = time.time() - t0

    def summ(key):
        vals = [r[key] for r in results if key in r]
        return statistics.mean(vals), statistics.pstdev(vals)

    base_acc = summ("bag1_acc")[0]
    base_auc = summ("bag1_auc")[0]

    print("=== ACCURACY (mean ± std) · Δ vs single model ===")
    for bags in BAG_SIZES:
        mu, sd = summ(f"bag{bags}_acc")
        eff = results[0][f"bag{bags}_layers"]
        tag = "  <-- single (≈algorithmeai)" if bags == 1 else f"  Δ {mu - base_acc:+.4f}"
        print(f"  bags={bags} ({eff} total layers)  {mu:.4f} ± {sd:.4f}{tag}")

    print("\n=== AUROC (mean ± std) · Δ vs single model ===")
    for bags in BAG_SIZES:
        mu, sd = summ(f"bag{bags}_auc")
        tag = "  <-- single (≈algorithmeai)" if bags == 1 else f"  Δ {mu - base_auc:+.4f}"
        print(f"  bags={bags}  {mu:.4f} ± {sd:.4f}{tag}")

    def sign_p(w, l):
        """Two-sided sign-test p-value: P(|wins| this extreme | fair coin)."""
        n = w + l
        if n == 0:
            return 1.0
        k = max(w, l)
        tail = sum(comb(n, i) for i in range(k, n + 1)) / (2 ** n)
        return min(1.0, 2 * tail)

    print("\n=== PER-SEED WINS vs single model (ties excluded, sign-test p) ===")
    for bags in BAG_SIZES:
        if bags == 1:
            continue
        wa = sum(1 for r in results if r[f"bag{bags}_acc"] > r["bag1_acc"])
        la = sum(1 for r in results if r[f"bag{bags}_acc"] < r["bag1_acc"])
        wu = sum(1 for r in results if r[f"bag{bags}_auc"] > r["bag1_auc"])
        lu = sum(1 for r in results if r[f"bag{bags}_auc"] < r["bag1_auc"])
        print(f"  bags={bags}  acc {wa}W-{la}L (p={sign_p(wa, la):.3f})   "
              f"auroc {wu}W-{lu}L (p={sign_p(wu, lu):.3f})   (of {len(results)})")

    print(f"\ntotal wall time: {wall:.1f}s")


if __name__ == "__main__":
    main()
