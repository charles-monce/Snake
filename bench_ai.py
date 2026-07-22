#!/usr/bin/env python3
"""Fire-and-forget concurrent benchmark: algorithmeai vs monceai on Titanic.

Spawns N workers in a process pool. Each worker:
  * runs in its own temp cwd (algorithmeai auto-writes snakeclassifier.json)
  * splits titanic/train.csv 80/20 with its own seed
  * trains both engines on the same split
  * scores held-out accuracy for each
Then we aggregate mean/std/spread across workers.
"""
import csv
import os
import random
import statistics
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor

REPO = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV = os.path.join(REPO, "titanic", "train.csv")
N_WORKERS = int(os.environ.get("BENCH_WORKERS", "10"))
N_LAYERS = int(os.environ.get("BENCH_LAYERS", "30"))
TEST_FRAC = 0.20

# Titanic column types for building typed query dicts (target = Survived).
NUM_INT = {"Pclass", "SibSp", "Parch"}
NUM_FLOAT = {"Age", "Fare"}
DROP = {"PassengerId", "Survived"}


def auroc(scores, labels):
    """AUROC via the Mann-Whitney U statistic, with tie handling. Stdlib only.

    Threshold-free: measures P(score of a random positive > score of a random
    negative). 0.5 = coin flip, 1.0 = perfect ranking. Ties contribute 0.5.
    """
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    # Rank-sum: sum over positive/negative pairs of (1 if p>n, 0.5 if tie).
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    return wins / (len(pos) * len(neg))


def _typed(row):
    """Cast a csv.DictReader row into a feature dict (Survived excluded)."""
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
    """One independent train/test run of both engines. Returns a result dict."""
    sys.path.insert(0, REPO)
    import algorithmeai
    import monceai

    # Read all labeled rows, split by seed.
    with open(TRAIN_CSV) as f:
        reader = csv.reader(f)
        header = next(reader)
        all_rows = list(reader)
    rng = random.Random(seed)
    rng.shuffle(all_rows)
    n_test = int(len(all_rows) * TEST_FRAC)
    test_rows, train_rows = all_rows[:n_test], all_rows[n_test:]

    # Each worker isolates its cwd so algorithmeai's auto-saved json can't race.
    workdir = tempfile.mkdtemp(prefix=f"bench_{seed}_")
    os.chdir(workdir)
    train_path = os.path.join(workdir, "train.csv")
    _write_csv(train_path, header, train_rows)

    # Build typed test set + truth (Survived is column index 1).
    dict_rows = [dict(zip(header, r)) for r in test_rows]
    X_test = [_typed(r) for r in dict_rows]
    y_true = [int(r["Survived"]) for r in dict_rows]

    out = {"seed": seed, "n_train": len(train_rows), "n_test": len(test_rows)}

    # --- algorithmeai (target_index=1 Survived, exclude PassengerId idx 0) ---
    t0 = time.time()
    a = algorithmeai.Snake(train_path, n_layers=N_LAYERS, vocal=False,
                           target_index=1, excluded_features_index=[0])
    out["algo_train_s"] = round(time.time() - t0, 2)
    a_pred = [a.get_prediction(x) for x in X_test]
    out["algo_acc"] = sum(int(p == t) for p, t in zip(a_pred, y_true)) / len(y_true)
    a_score = [a.get_probability(x).get(1, 0.0) for x in X_test]
    out["algo_auroc"] = auroc(a_score, y_true)

    # --- monceai (seeded => reproducible) ------------------------------------
    t0 = time.time()
    m = monceai.Snake(train_path, n_layers=N_LAYERS, vocal=False,
                      target_index=1, excluded_features_index=[0], seed=seed)
    out["monce_train_s"] = round(time.time() - t0, 2)
    m_pred = [m.predict(x) for x in X_test]
    out["monce_acc"] = sum(int(p == t) for p, t in zip(m_pred, y_true)) / len(y_true)
    m_score = [m.predict_proba(x).get(1, 0.0) for x in X_test]
    out["monce_auroc"] = auroc(m_score, y_true)

    out["agreement"] = sum(int(pa == pm) for pa, pm in zip(a_pred, m_pred)) / len(y_true)
    return out


def main():
    print(f"Firing {N_WORKERS} concurrent workers · {N_LAYERS} layers · "
          f"{int((1-TEST_FRAC)*100)}/{int(TEST_FRAC*100)} split\n")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        results = list(pool.map(worker, range(N_WORKERS)))
    wall = time.time() - t0

    print(f"{'seed':>4} {'algo_acc':>9} {'monce_acc':>10} "
          f"{'algo_auc':>9} {'monce_auc':>10} {'agree':>7}")
    for r in sorted(results, key=lambda x: x["seed"]):
        print(f"{r['seed']:>4} {r['algo_acc']:>9.3f} {r['monce_acc']:>10.3f} "
              f"{r['algo_auroc']:>9.3f} {r['monce_auroc']:>10.3f} {r['agreement']:>7.3f}")

    def summ(key):
        vals = [r[key] for r in results]
        return statistics.mean(vals), statistics.pstdev(vals), min(vals), max(vals)

    print("\n=== SUMMARY (mean ± std, [min, max]) ===")
    for label, key in [("algorithmeai acc  ", "algo_acc"),
                       ("monceai      acc  ", "monce_acc"),
                       ("algorithmeai AUROC", "algo_auroc"),
                       ("monceai      AUROC", "monce_auroc"),
                       ("pred agreement    ", "agreement")]:
        mu, sd, lo, hi = summ(key)
        print(f"{label}: {mu:.3f} ± {sd:.3f}   [{lo:.3f}, {hi:.3f}]")

    aa, _, _, _ = summ("algo_acc")
    ma, _, _, _ = summ("monce_acc")
    au, _, _, _ = summ("algo_auroc")
    mu_, _, _, _ = summ("monce_auroc")
    print(f"\nΔ accuracy (monceai - algorithmeai): {ma - aa:+.3f}")
    print(f"Δ AUROC    (monceai - algorithmeai): {mu_ - au:+.3f}")
    print(f"total wall time: {wall:.1f}s")


if __name__ == "__main__":
    main()
