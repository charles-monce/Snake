#!/usr/bin/env python3
"""Decision-threshold experiment: the one lever that moves accuracy at fixed AUROC.

Every prior experiment tied on AUROC because monceai and algorithmeai share the
clause engine and therefore produce the SAME score *ranking*. AUROC only sees
ranking. Accuracy, though, is decided by WHERE you cut that ranking — and
algorithmeai hardcodes the cut at argmax (threshold 0.5 for a binary target).

Titanic is imbalanced (~38% positive), so 0.5 is unlikely to be the
accuracy-optimal cut. This harness trains ONE algorithmeai model per seed, then
on the SAME p1 score stream compares:

  argmax0.5 : algorithmeai's native cut (predict 1 iff p1 >= 0.5)
  tuned     : the threshold that maximises accuracy on the TRAINING scores,
              then applied unseen to test (a standard, honest, leak-free tune)

Same model, same lookalikes, same scores, same AUROC — only the cut differs.
A systematic accuracy win here is a real monceai-only improvement, because it
needs a training-scored threshold that algorithmeai never computes.
"""
import csv
import os
import random
import statistics
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from math import comb

REPO = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV = os.path.join(REPO, "titanic", "train.csv")
N_WORKERS = int(os.environ.get("BENCH_WORKERS", "12"))
N_SEEDS = int(os.environ.get("BENCH_SEEDS", str(N_WORKERS)))
N_LAYERS = int(os.environ.get("BENCH_LAYERS", "30"))
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


def best_threshold(scores, labels):
    """Threshold maximising accuracy on (scores, labels). Leak-free when the
    inputs are TRAINING data. Ties broken toward 0.5 to stay conservative."""
    cuts = sorted(set(scores))
    # candidate cuts = midpoints between adjacent distinct scores, plus ends
    cands = [0.5]
    for i in range(len(cuts)):
        cands.append(cuts[i] - 1e-9)
        if i + 1 < len(cuts):
            cands.append((cuts[i] + cuts[i + 1]) / 2)
    best_t, best_acc = 0.5, -1.0
    n = len(labels)
    for t in cands:
        acc = sum(int((s >= t) == bool(y)) for s, y in zip(scores, labels)) / n
        # prefer higher acc; on tie prefer the cut closest to 0.5
        if acc > best_acc or (acc == best_acc and abs(t - 0.5) < abs(best_t - 0.5)):
            best_acc, best_t = acc, t
    return best_t


def worker(seed):
    sys.path.insert(0, REPO)
    import algorithmeai

    with open(TRAIN_CSV) as f:
        reader = csv.reader(f)
        header = next(reader)
        all_rows = list(reader)
    rng = random.Random(seed)
    rng.shuffle(all_rows)
    n_test = int(len(all_rows) * TEST_FRAC)
    test_rows, train_rows = all_rows[:n_test], all_rows[n_test:]

    # The model MEMORISES its training rows, so a threshold tuned on training
    # scores never moves off 0.5. We carve a validation fold OUT of train: the
    # model is fit on `fit_rows` only, the cut is tuned on held-out `val_rows`,
    # and evaluated on the untouched test set. Leak-free on both counts.
    n_val = int(len(train_rows) * 0.25)
    val_rows, fit_rows = train_rows[:n_val], train_rows[n_val:]

    workdir = tempfile.mkdtemp(prefix=f"thr_{seed}_")
    os.chdir(workdir)
    train_path = os.path.join(workdir, "train.csv")
    _write_csv(train_path, header, fit_rows)

    # test set
    dtest = [dict(zip(header, r)) for r in test_rows]
    X_test = [_typed(r) for r in dtest]
    y_test = [int(r["Survived"]) for r in dtest]
    # validation fold (held out from fitting) — where the threshold is tuned
    dval = [dict(zip(header, r)) for r in val_rows]
    X_val = [_typed(r) for r in dval]
    y_val = [int(r["Survived"]) for r in dval]

    a = algorithmeai.Snake(train_path, n_layers=N_LAYERS, vocal=False,
                           target_index=1, excluded_features_index=[0])

    val_scores = [a.get_probability(x).get(1, 0.0) for x in X_val]
    test_scores = [a.get_probability(x).get(1, 0.0) for x in X_test]

    t_star = best_threshold(val_scores, y_val)

    acc_05 = sum(int((s >= 0.5) == bool(y)) for s, y in zip(test_scores, y_test)) / len(y_test)
    acc_tuned = sum(int((s >= t_star) == bool(y)) for s, y in zip(test_scores, y_test)) / len(y_test)

    return {
        "seed": seed,
        "t_star": t_star,
        "argmax_acc": acc_05,
        "tuned_acc": acc_tuned,
        "auc": auroc(test_scores, y_test),  # identical for both cuts
    }


def sign_p(w, l):
    n = w + l
    if n == 0:
        return 1.0
    k = max(w, l)
    tail = sum(comb(n, i) for i in range(k, n + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def main():
    print(f"Threshold experiment · {N_SEEDS} seeds · {N_LAYERS} layers · "
          f"same scores, argmax0.5 vs training-tuned cut\n")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        results = list(pool.map(worker, range(N_SEEDS)))
    wall = time.time() - t0

    def summ(key):
        vals = [r[key] for r in results]
        return statistics.mean(vals), statistics.pstdev(vals)

    am, asd = summ("argmax_acc")
    tm, tsd = summ("tuned_acc")
    auc, aucsd = summ("auc")
    ts_m, ts_sd = summ("t_star")

    print("=== ACCURACY (mean ± std) ===")
    print(f"  argmax @0.5   {am:.4f} ± {asd:.4f}   <-- algorithmeai's cut")
    print(f"  tuned cut     {tm:.4f} ± {tsd:.4f}   Δ {tm - am:+.4f}")
    print(f"  (AUROC, unchanged by cut: {auc:.4f} ± {aucsd:.4f})")
    print(f"  mean tuned threshold: {ts_m:.3f} ± {ts_sd:.3f}")

    w = sum(1 for r in results if r["tuned_acc"] > r["argmax_acc"])
    l = sum(1 for r in results if r["tuned_acc"] < r["argmax_acc"])
    print(f"\n=== PER-SEED WINS (tuned vs argmax) ===")
    print(f"  {w}W-{l}L of {len(results)}   sign-test p = {sign_p(w, l):.4f}")
    print(f"\ntotal wall time: {wall:.1f}s")


if __name__ == "__main__":
    main()
