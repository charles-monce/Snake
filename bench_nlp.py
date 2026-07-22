#!/usr/bin/env python3
"""NLP arena: does the MI boolean bank beat baseline on bag-of-words text?

The user asked for NLP explicitly ("work on NLP"), expecting weakness. This
tests whether monceai's MI-filtered token bank beats algorithmeai's random
token opposition on a text-classification task.

Text is where the MI bank *should* shine: a document has a huge token
vocabulary, most tokens are noise, and only a handful carry sentiment. Random
opposition (algorithmeai) grabs any discriminating token — usually a
content-free one that happens to differ. The MI bank ranks tokens by
I(token present; label) and steers clauses toward the sentiment-bearing ones.

SYNTHETIC corpus (stated plainly): each document is a bag of words drawn from a
shared neutral vocabulary plus, for its class, a few class-specific keywords
mixed in at low rate. Realistic in structure (sparse signal in a sea of neutral
tokens), fully synthetic in content.
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
N_DOCS = int(os.environ.get("BENCH_DOCS", "400"))
MI_RATE = float(os.environ.get("BENCH_MIRATE", "0.7"))
TEST_FRAC = 0.25

NEUTRAL = [f"w{i}" for i in range(120)]           # shared neutral vocabulary
POS = ["great", "love", "excellent", "brilliant", "wonderful", "best"]
NEG = ["awful", "hate", "terrible", "worst", "boring", "poor"]


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


def make_corpus(seed):
    rng = random.Random(seed)
    docs = []
    for _ in range(N_DOCS):
        y = rng.randint(0, 1)
        length = rng.randint(18, 30)
        words = [rng.choice(NEUTRAL) for _ in range(length)]
        # sprinkle a few class keywords (sparse signal, some cross-leak)
        kw = POS if y == 1 else NEG
        for _ in range(rng.randint(1, 3)):
            words[rng.randrange(length)] = rng.choice(kw)
        if rng.random() < 0.10:                     # 10% cross-leak keyword
            other = NEG if y == 1 else POS
            words[rng.randrange(length)] = rng.choice(other)
        rng.shuffle(words)
        docs.append((y, " ".join(words)))
    return docs


def worker(seed):
    sys.path.insert(0, REPO)
    import monceai

    docs = make_corpus(seed)
    rng = random.Random(seed * 7 + 1)
    rng.shuffle(docs)
    n_test = int(len(docs) * TEST_FRAC)
    test, train = docs[:n_test], docs[n_test:]

    workdir = tempfile.mkdtemp(prefix=f"nlp_{seed}_")
    os.chdir(workdir)
    train_path = os.path.join(workdir, "train.csv")
    with open(train_path, "w") as f:
        f.write("label,text\n")
        for y, t in train:
            f.write(f"{y},{t}\n")

    X_test = [{"text": t} for _, t in test]
    y_test = [y for y, _ in test]

    def score(m):
        preds = [m.predict(x) for x in X_test]
        acc = sum(int(p == t) for p, t in zip(preds, y_test)) / len(y_test)
        sc = [m.predict_proba(x).get(1, 0.0) for x in X_test]
        return acc, auroc(sc, y_test)

    base = monceai.Snake(train_path, n_layers=N_LAYERS, vocal=False,
                         target_index=0, seed=seed, mi_rate=0.0)
    b_acc, b_auc = score(base)
    mon = monceai.Snake(train_path, n_layers=N_LAYERS, vocal=False,
                        target_index=0, seed=seed, mi_rate=MI_RATE)
    m_acc, m_auc = score(mon)
    return {"seed": seed, "base_acc": b_acc, "base_auc": b_auc,
            "monce_acc": m_acc, "monce_auc": m_auc}


def main():
    print(f"NLP arena · {N_SEEDS} seeds · {N_LAYERS} layers · {N_DOCS} docs · "
          f"bag-of-words sentiment · mi_rate={MI_RATE}\n(SYNTHETIC corpus)\n")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        results = list(pool.map(worker, range(N_SEEDS)))
    wall = time.time() - t0

    def summ(key):
        vals = [r[key] for r in results]
        return statistics.mean(vals), statistics.pstdev(vals)

    ba, bas = summ("base_acc"); ma, mas = summ("monce_acc")
    bu, bus = summ("base_auc"); mu, mus = summ("monce_auc")
    print("=== ACCURACY (mean ± std) ===")
    print(f"  baseline (random token oppose, ≈algorithmeai)  {ba:.4f} ± {bas:.4f}")
    print(f"  monceai  (MI token bank)                       {ma:.4f} ± {mas:.4f}   Δ {ma - ba:+.4f}")
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
