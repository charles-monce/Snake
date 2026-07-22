#!/usr/bin/env python3
"""Decisive inference-rule experiment: monceai's decision vs algorithmeai's.

The two engines share the SAME randomized clause engine — with mi_rate=0 they
are the same model up to RNG, so end-to-end they tie. This harness removes that
confound entirely:

  * train ONE algorithmeai model per seed on the Titanic split
  * for every test point, pull its lookalikes ONCE (identical learned structure)
  * score those SAME lookalikes with several decision rules

The clause engine, the training, the seed — all identical. The ONLY thing that
varies is how lookalike votes become a probability. If a rule beats the native
plurality vote on the same lookalikes, that is a genuine, seed-noise-free win we
can then bake into monceai.predict_proba as the default.

Rules under test (all O(#lookalikes) at inference, zero deps):
  plurality  : algorithmeai's native rule (uniform vote, uniform fallback)
  baserate   : plurality, but fall back to the class prior when no lookalikes
  laplace-A  : Laplace smoothing of votes toward the prior, strength A
  spec       : votes weighted by condition specificity (#clauses ANDed)
  spec+lap   : specificity weighting AND Laplace smoothing (the candidate)
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

REPO = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV = os.path.join(REPO, "titanic", "train.csv")
N_WORKERS = int(os.environ.get("BENCH_WORKERS", "12"))
N_LAYERS = int(os.environ.get("BENCH_LAYERS", "30"))
LAPLACE = float(os.environ.get("BENCH_LAPLACE", "1.0"))
TEST_FRAC = 0.20

NUM_INT = {"Pclass", "SibSp", "Parch"}
NUM_FLOAT = {"Age", "Fare"}
DROP = {"PassengerId", "Survived"}


def auroc(scores, labels):
    """AUROC via Mann-Whitney U with tie handling. Stdlib only."""
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


# ---- decision rules: (lookalikes, prior) -> {class: proba} ----------------- #
# A lookalike is [row_index, class, condition]; condition is a list of clause
# indices (its length = how many clauses had to be simultaneously false = how
# specific the match is). `prior` is {class: training base rate}.

def rule_plurality(looks, prior):
    classes = sorted(prior)
    if not looks:
        return {c: 1.0 / len(classes) for c in classes}  # uniform fallback
    n = len(looks)
    return {c: sum(1 for _, t, _ in looks if t == c) / n for c in classes}


def rule_baserate(looks, prior):
    if not looks:
        return dict(prior)  # informed fallback: the class prior
    return rule_plurality(looks, prior)


def rule_laplace(looks, prior, alpha):
    classes = sorted(prior)
    if not looks:
        return dict(prior)
    counts = Counter(t for _, t, _ in looks)
    n = len(looks)
    # proba(c) = (votes_c + alpha*prior_c) / (n + alpha): shrink toward prior.
    return {c: (counts.get(c, 0) + alpha * prior[c]) / (n + alpha) for c in classes}


def rule_spec(looks, prior):
    classes = sorted(prior)
    if not looks:
        return dict(prior)
    # weight each vote by specificity = #clauses in its matching condition.
    tot = 0.0
    acc = {c: 0.0 for c in classes}
    for _, t, cond in looks:
        w = 1.0 + len(cond)
        acc[t] += w
        tot += w
    return {c: acc[c] / tot for c in classes}


def rule_spec_laplace(looks, prior, alpha):
    classes = sorted(prior)
    if not looks:
        return dict(prior)
    acc = {c: 0.0 for c in classes}
    tot = 0.0
    for _, t, cond in looks:
        w = 1.0 + len(cond)
        acc[t] += w
        tot += w
    return {c: (acc[c] + alpha * prior[c]) / (tot + alpha) for c in classes}


RULES = [
    ("plurality", lambda lk, pr: rule_plurality(lk, pr)),
    ("baserate", lambda lk, pr: rule_baserate(lk, pr)),
    (f"laplace{LAPLACE:g}", lambda lk, pr: rule_laplace(lk, pr, LAPLACE)),
    ("spec", lambda lk, pr: rule_spec(lk, pr)),
    (f"spec+lap{LAPLACE:g}", lambda lk, pr: rule_spec_laplace(lk, pr, LAPLACE)),
]


def _argmax_class(proba):
    best = max(proba.values())
    return next(c for c in proba if proba[c] == best)


def worker(seed):
    """Train one algorithmeai model; score the SAME lookalikes every rule way."""
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

    workdir = tempfile.mkdtemp(prefix=f"infer_{seed}_")
    os.chdir(workdir)
    train_path = os.path.join(workdir, "train.csv")
    _write_csv(train_path, header, train_rows)

    dict_rows = [dict(zip(header, r)) for r in test_rows]
    X_test = [_typed(r) for r in dict_rows]
    y_true = [int(r["Survived"]) for r in dict_rows]

    a = algorithmeai.Snake(train_path, n_layers=N_LAYERS, vocal=False,
                           target_index=1, excluded_features_index=[0])

    # Class prior from the model's own training targets (base rate).
    counts = Counter(a.targets)
    total = sum(counts.values())
    prior = {c: counts[c] / total for c in sorted(counts)}

    # Pull each test point's lookalikes ONCE — the shared learned structure.
    all_looks = [a.get_lookalikes(x) for x in X_test]

    out = {"seed": seed}
    for name, fn in RULES:
        preds, scores = [], []
        for looks in all_looks:
            proba = fn(looks, prior)
            preds.append(_argmax_class(proba))
            scores.append(proba.get(1, 0.0))
        acc = sum(int(p == t) for p, t in zip(preds, y_true)) / len(y_true)
        out[f"{name}_acc"] = acc
        out[f"{name}_auc"] = auroc(scores, y_true)
    return out


def main():
    print(f"Inference-rule experiment · {N_WORKERS} seeds · {N_LAYERS} layers · "
          f"Laplace={LAPLACE:g} · identical lookalikes per point\n")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        results = list(pool.map(worker, range(N_WORKERS)))
    wall = time.time() - t0

    names = [n for n, _ in RULES]

    def summ(key):
        vals = [r[key] for r in results if key in r]
        return statistics.mean(vals), statistics.pstdev(vals)

    print("=== ACCURACY (mean ± std over seeds) ===")
    base_acc = summ("plurality_acc")[0]
    for name in names:
        mu, sd = summ(f"{name}_acc")
        delta = mu - base_acc
        flag = "  <-- baseline" if name == "plurality" else f"  Δ {delta:+.4f}"
        print(f"  {name:<14} {mu:.4f} ± {sd:.4f}{flag}")

    print("\n=== AUROC (mean ± std over seeds) ===")
    base_auc = summ("plurality_auc")[0]
    for name in names:
        mu, sd = summ(f"{name}_auc")
        delta = mu - base_auc
        flag = "  <-- baseline" if name == "plurality" else f"  Δ {delta:+.4f}"
        print(f"  {name:<14} {mu:.4f} ± {sd:.4f}{flag}")

    # Per-seed win counts vs plurality: does the candidate win *systematically*?
    print("\n=== PER-SEED WINS vs plurality (acc / auroc, ties excluded) ===")
    for name in names:
        if name == "plurality":
            continue
        wa = sum(1 for r in results if r[f"{name}_acc"] > r["plurality_acc"])
        la = sum(1 for r in results if r[f"{name}_acc"] < r["plurality_acc"])
        wu = sum(1 for r in results if r[f"{name}_auc"] > r["plurality_auc"])
        lu = sum(1 for r in results if r[f"{name}_auc"] < r["plurality_auc"])
        print(f"  {name:<14} acc {wa}W-{la}L   auroc {wu}W-{lu}L   (of {len(results)})")

    print(f"\ntotal wall time: {wall:.1f}s")


if __name__ == "__main__":
    main()
