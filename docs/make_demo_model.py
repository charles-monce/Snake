#!/usr/bin/env python3
"""
Build the tiny REAL monceai model that powers the "magic box" on the landing
page. Synthetic, human-meaningful, text outcome — trained by the actual engine,
then exported to a compact JSON the in-browser JS reader replays verbatim.

    python docs/make_demo_model.py

Writes docs/demo_model.json. No external data, deterministic under the seed.
"""
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # import monceai from repo root
from monceai import Snake  # noqa: E402


# ---------------------------------------------------------------------------
#  A small synthetic world: recommend a TRIP TYPE from a traveller's profile.
#  Text outcome (Beach Escape / Mountain Trek / City Explorer), mixed columns,
#  a text preferences field so token-presence literals show off in the audit.
# ---------------------------------------------------------------------------
CLIMATES = ["warm", "mild", "cold"]
WANTS = {
    "Beach Escape":   ["sunbathing", "swimming", "cocktails", "snorkeling", "relaxing"],
    "Mountain Trek":  ["hiking", "climbing", "camping", "wildlife", "waterfalls"],
    "City Explorer":  ["museums", "nightlife", "food", "shopping", "architecture"],
}


def label_of(budget, days, climate, group, wants_text):
    """The ground-truth rule the engine has to recover from the columns."""
    score = {"Beach Escape": 0.0, "Mountain Trek": 0.0, "City Explorer": 0.0}
    # climate is a strong steer
    score["Beach Escape"]  += {"warm": 2.0, "mild": 0.3, "cold": -1.0}[climate]
    score["Mountain Trek"] += {"cold": 1.6, "mild": 0.8, "warm": -0.5}[climate]
    score["City Explorer"] += 0.4  # climate-agnostic
    # activity words in the free-text field
    for kind, words in WANTS.items():
        score[kind] += 1.3 * sum(w in wants_text for w in words)
    # budget & trip length
    if budget >= 3000:
        score["City Explorer"] += 0.8
    if days >= 8:
        score["Mountain Trek"] += 0.7
    if group >= 4:
        score["Beach Escape"] += 0.6
    return max(score, key=score.get)


def make_rows(n, rng):
    rows = []
    for _ in range(n):
        climate = rng.choice(CLIMATES)
        budget = rng.choice([800, 1200, 1800, 2500, 3200, 4500])
        days = rng.choice([3, 4, 5, 7, 9, 12])
        group = rng.choice([1, 2, 3, 4, 6])
        kind = rng.choice(list(WANTS))
        # pick 2-3 activity words, mostly from the intended kind + some spillover
        picks = rng.sample(WANTS[kind], k=rng.choice([2, 3]))
        if rng.random() < 0.3:
            other = rng.choice([k for k in WANTS if k != kind])
            picks.append(rng.choice(WANTS[other]))
        rng.shuffle(picks)
        wants_text = " ".join(picks)
        label = label_of(budget, days, climate, group, wants_text)
        # 8% label noise so it isn't trivially separable
        if rng.random() < 0.08:
            label = rng.choice(list(WANTS))
        rows.append([label, budget, days, climate, group, wants_text])
    return rows


def main():
    rng = random.Random(7)
    header = ["trip_type", "budget_usd", "days", "climate", "group_size", "wants"]
    rows = make_rows(150, rng)

    csv_path = os.path.join(HERE, "_demo_train.csv")
    with open(csv_path, "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(c) for c in r) + "\n")

    # Train the REAL engine. Small layer count keeps the JSON light enough to
    # embed; seed makes it reproducible. mi_rate stays "auto" (this narrow
    # 5-column table => classic engine, exactly what ships by default).
    model = Snake(csv_path, n_layers=14, vocal=True, target_index=0, seed=7)

    # Sanity: train accuracy
    hits = 0
    for r in rows:
        X = {"budget_usd": r[1], "days": r[2], "climate": r[3],
             "group_size": r[4], "wants": r[5]}
        if model.predict(X) == r[0]:
            hits += 1
    print(f"# demo train accuracy: {hits}/{len(rows)} = {hits/len(rows):.3f}")

    out = os.path.join(HERE, "demo_model.json")
    blob = model.to_json()
    # Trim what the browser reader never touches, to shrink the payload.
    for k in ("log", "raw_targets", "literal_bank"):
        blob.pop(k, None)
    with open(out, "w") as f:
        json.dump(blob, f, separators=(",", ":"))
    size = os.path.getsize(out)
    print(f"# wrote {out} · {size/1024:.1f} KB · "
          f"{len(blob['clauses'])} clauses · {len(blob['population'])} rows")
    os.remove(csv_path)


if __name__ == "__main__":
    main()
