# monceai 🐍

**The single script that encodes everything.**

A dependency-free, `O(m·n²)`-budgeted, *explainable* oracle for CSV data.
One file. One class. One idea, followed all the way down.

**Author:** Charles Dana · [Algorithme.ai](https://algorithme.ai)
**Article / landing page:** https://charles-monce.github.io/Snake/

> This is the article rewrite of the original
> [`AlgorithmeAi/Snake`](https://github.com/AlgorithmeAi/Snake) — the same idea,
> told cleanly end to end in a single readable file, plus a controlled study of
> where mutual-information steering earns its keep. **`algorithmeai.py` is
> untouched;** everything new lives in `monceai.py` and the `bench_*.py` scripts.

---

## What it is

Give it a table. It learns to predict one column from the others, and — unlike
almost every other model — it can *tell you why*, in plain sentences, by
pointing at the training rows that look like your query and the exact logical
conditions that bind them together. No dependencies; `pandas` is touched in
exactly one optional method (parallel batch inference) and nowhere else.

## The result that motivated the rewrite

The original engine chooses which feature to split on **uniformly at random**.
monceai adds a **mutual-information-filtered boolean bank**: every candidate
predicate is scored by `I(predicate; target)` and clause construction is steered
toward the columns that actually carry signal.

We measured, honestly, *when that helps* — paired same-seed, same-budget
comparisons across many seeds, with a two-sided sign-test:

| arena | Δ accuracy | Δ AUROC | per-seed | verdict |
|---|---|---|---|---|
| **wide-noisy** (3 signal + 40 noise cols) | **+0.088** | **+0.059** | **40W–0L** (p≈0) | steering **wins** |
| clean tabular (Titanic) | +0.003 | +0.004 | tie | identical to original |
| NLP (single text column) | −0.013 | −0.012 | 1W–22L | steering **loses** |

The full crossover law (MI-steering helps once informative columns are a small
minority) and three published **negative results** (bagging, vote-rules,
threshold-tuning — all nulls) are in the [landing page](https://charles-monce.github.io/Snake/).

Because the win is conditional, the default is `mi_rate="auto"`: monceai reads
the table's shape at train time and engages steering only when it will help — so
it **beats the original on wide/noisy data and stays identical on clean data.**

## Use it as a library

```python
from monceai import Snake

model = Snake("train.csv", target_index=1, n_layers=100)   # mi_rate="auto"
model.predict({"Age": 22, "Sex": "male"})
model.predict_proba({"Age": 22, "Sex": "male"})
print(model.audit({"Age": 22, "Sex": "male"}))   # plain-English reasons
model.explain()                                    # global feature importance
model.save("model.json")
model.to_py("frozen.py")                           # standalone zero-dep inference
```

## Use it as a program

```bash
python monceai.py train  data.csv --target 1 --layers 100 -o model.json
python monceai.py predict model.json --csv new.csv
python monceai.py audit   model.json --row '{"Age": 22, "Sex": "male"}'
python monceai.py explain model.json
python monceai.py selftest
```

## Reproduce the study

```bash
python monceai.py selftest        # proves the engine, zero deps
BENCH_SEEDS=40 python bench_wide.py   # the headline wide-noisy win
python bench_nlp.py               # the honest NLP loss
python bench_bag.py               # null: compute-fair bagging
python bench_thresh.py            # null: tuned decision threshold
python bench_ai.py                # head-to-head vs algorithmeai on Titanic
```

## License

MIT
