#!/usr/bin/env python3
"""
================================================================================

    monceai.py  ·  Snake  ·  Author: Charles Dana  ·  Algorithme.ai

    The single script that encodes everything.

    A dependency-free, O(m·n^2)-budgeted, explainable oracle for CSV data.
    One file. One class. One idea, followed all the way down.

================================================================================

WHAT THIS IS
------------
Give it a table. It learns to predict one column from the others, and — unlike
almost every other model — it can *tell you why*, in plain sentences, by
pointing at the training rows that look like your query and the exact logical
conditions that bind them together.

It handles four flavours of problem with the same machinery:

    * binary          (0/1, True/False, Yes/No)
    * multiclass      (integers, floats-as-labels, or free text)
    * regression      (a continuous number, via ordered bins + averaging)
    * multi-target    (any label alphabet, deterministically ordered)

It needs nothing but the Python standard library. `pandas` is touched in
exactly one optional method (parallel batch inference) and nowhere else.

THE ONE IDEA
------------
Everything below is built from a single primitive: the *literal*, a yes/no
question about one feature ("is Age > 37?", "does Name contain 'Mrs'?").

    literal      a single test on one column        -> True / False
    clause       an OR of literals                   -> True if any fires
    condition    an AND of clauses (by negation)     -> the reason for a match
    layer        one randomized pass building clauses that separate the classes
    model        many layers stacked into a lookalike table

To classify a new point we find its *lookalikes*: training rows whose learned
condition it satisfies. The class distribution of those lookalikes is the
prediction, and the conditions themselves are the audit trail.

WHY O(m·n^2)  (n = rows, m = feature columns)
---------------------------------------------
    oppose()           picks a discriminating feature + threshold        O(m)
    construct_clause() grows literals until every "T" row is covered,
                       then minimises; a minimised clause is O(m) long,
                       each growth/prune step rescans <= n rows           O(m·n)
    construct_sat()    repeats until every "F" row is covered: O(n) clauses  O(m·n^2)
    construct_layer()  does that once per class; the F/T sets partition
                       the population, so a whole layer is still          O(m·n^2)

A model of L layers is O(L·m·n^2). The bound holds *because* clauses are
minimised to O(m) literals — the pruning loop in construct_clause() is what
keeps the promise honest. That is the entire performance contract.

USAGE
-----
    As a library:
        from monceai import Snake
        model = Snake("train.csv", target_index=1, n_layers=100)
        print(model.predict({"Age": 22, "Sex": "male", ...}))
        print(model.audit({"Age": 22, "Sex": "male", ...}))

    As a program:
        python monceai.py train  data.csv --target 1 --layers 100 -o model.json
        python monceai.py predict model.json --csv new.csv
        python monceai.py audit   model.json --row '{"Age": 22, "Sex": "male"}'
        python monceai.py explain model.json          # global feature importance
        python monceai.py selftest                     # prove it works, no deps

================================================================================
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from math import log
from random import Random
from time import time


# ============================================================================ #
#  SECTION 0 — Small pure helpers                                              #
# ============================================================================ #

def to_float(text):
    """Parse a float, degrading to 0.0 on anything unparseable.

    CSVs are dirty. Rather than crash on a stray "" or "N/A", we treat a
    malformed numeric cell as 0.0 — the same forgiving contract the original
    Snake made, so old datasets behave identically.
    """
    try:
        return float(text)
    except (ValueError, TypeError):
        return 0.0


def mean(values):
    """Arithmetic mean of an iterable; 0.0 on empty. Stdlib only."""
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def is_numeric_universe(chars):
    """True iff every character could belong to a float literal.

    Used to auto-detect whether a column is numeric ("N") or text ("T") from
    the union of all characters seen in it — no per-cell float() attempts.
    """
    return all(c in "+-.0123456789e" for c in chars)


def entropy(labels):
    """Shannon entropy H(Y) in bits — the information content of the target."""
    n = len(labels)
    if n == 0:
        return 0.0
    counts = Counter(labels)
    return -sum((c / n) * log(c / n, 2) for c in counts.values())


def mutual_information(bits, labels):
    """I(B; Y) in bits: how much a boolean predicate B reveals about target Y.

    The single ruler by which every candidate boolean is judged. >= 0, <= H(Y).
    """
    n = len(labels)
    if n == 0:
        return 0.0
    hy = entropy(labels)
    ones = [labels[i] for i in range(n) if bits[i]]
    zeros = [labels[i] for i in range(n) if not bits[i]]
    p1 = len(ones) / n
    # Conditional entropy H(Y | B) = weighted entropy of each branch.
    h_cond = p1 * entropy(ones) + (1 - p1) * entropy(zeros)
    return max(0.0, hy - h_cond)


# ============================================================================ #
#  SECTION 1 — The model                                                       #
# ============================================================================ #

class Snake:
    """An explainable, dependency-free oracle over tabular data.

    Construct from a ``.csv`` path to train, or a ``.json`` path to load a
    previously saved model. The public surface is deliberately small:

        predict(X)           -> class label, or float estimate in regression
        predict_proba(X)     -> {class: probability}     (classification only)
        lookalikes(X)        -> [(index, target, condition), ...]
        audit(X)             -> a human-readable explanation string
        explain(X=None)      -> ranked feature importance (global or local)
        augment(X)           -> X plus all of the above, as a dict
        predict_frame(df)    -> pandas Series, computed in parallel (optional)
        validate(rows)       -> prune the model against held-out labelled rows
        save(path)/to_json() -> persist; the single source of truth for the
                                on-disk encoding, extended but back-compatible

    ``mode`` is ``"C"`` (classification) or ``"R"`` (regression). Regression
    buckets the target into ordered bins so the identical clause machinery
    applies, then reconstructs a continuous number by averaging the bin-centres
    of a query's lookalikes.
    """

    BANNER = (
        "################################################################\n"
        "#                                                              #\n"
        "#    Algorithme.ai : Snake  (monceai)   Author: Charles Dana   #\n"
        "#                                                              #\n"
        "#    A multiclass & regression .csv oracle  -  O(m n^2)        #\n"
        "#                                                              #\n"
        "################################################################\n"
    )

    # ---- construction ------------------------------------------------------ #

    def __init__(self, source=None, n_layers=100, vocal=True, target_index=0,
                 excluded_features_index=None, mode="C", regression_bins=16,
                 seed=None, mi_top_k=200, mi_sample=400, mi_greed=2.0,
                 mi_rate="auto"):
        self.log = self.BANNER
        # Core state (kept JSON-round-trippable).
        self.population = []      # list of feature dicts (the training rows)
        self.header = []          # [target_name, *feature_names]
        self.target = ""          # the target column name
        self.targets = []         # discrete labels driving the clause engine
        self.datatypes = []       # per-column tag: B/I/N/T (target first)
        self.clauses = []         # every clause ever built (OR of literals)
        self.lookalikes = {}      # {row_index_str: [condition, ...]}
        self.n_layers = n_layers
        self.vocal = vocal
        # v2 additions.
        self.mode = mode
        self.regression_bins = regression_bins
        self.seed = seed
        self.raw_targets = []     # original numeric targets before bucketing
        self.bin_centres = {}     # {bin_id: representative value} for regression
        # MI-filtered boolean bank: literals ranked by mutual information with
        # the target. The unified representation — text, numbers, booleans, any
        # series — all reduce to boolean predicates, all judged by one ruler.
        self.literal_bank = []
        self.mi_top_k = mi_top_k    # keep this many highest-MI predicates
        self.mi_sample = mi_sample  # rows sampled when scoring predicates
        self.mi_greed = mi_greed    # MI-sampling sharpness (0=uniform, ∞=argmax)
        # mi_rate: fraction of literals drawn from the MI bank. "auto" (the
        # default) decides from the table's shape at train time — see
        # _auto_mi_rate(). A float pins it explicitly. The auto rule is what
        # lets monceai beat the classic random-opposition engine on wide, noisy
        # tables while staying byte-identical to it on clean, narrow ones.
        self.mi_rate_setting = mi_rate
        self.mi_rate = 0.0 if mi_rate == "auto" else float(mi_rate)
        self._rng = Random(seed)  # seedable => reproducible models

        if source is None:
            return
        if source.endswith(".csv") or ".csv" in source:
            self._train(source, target_index, excluded_features_index or [])
        elif source.endswith(".json") or ".json" in source:
            self.load(source)
        else:
            self.say("# monceai: provide a .csv (to train) or .json (to load)")

    # ---- logging ----------------------------------------------------------- #

    def say(self, text):
        """Print if vocal, and always append to the persisted log."""
        if self.vocal:
            print(text)
        self.log += str(text) + "\n"

    # ---- training ---------------------------------------------------------- #

    def _train(self, csv_path, target_index, excluded):
        kind = "regression" if self.mode == "R" else "classification"
        self.say(f"# monceai: {kind} · {self.n_layers} layers · {csv_path}")

        header, rows = self._read_csv(csv_path)
        excl = set(excluded) | {target_index}
        target_name = header[target_index]
        feature_names = [header[i] for i in range(len(header)) if i not in excl]
        source_index = [target_index] + [i for i in range(len(header)) if i not in excl]
        self.header = [target_name] + feature_names
        self.target = target_name
        self.say(f"# features: {feature_names}")

        raw = [rows[r][target_index] for r in range(len(rows))]
        self.datatypes = [self._infer_target(raw)]

        # Detect each feature column's type from its character universe.
        for t in range(1, len(self.header)):
            src = source_index[t]
            values = [rows[r][src] for r in range(len(rows))]
            dtt = "N" if is_numeric_universe(set("".join(values))) else "T"
            self.say(f"#   [{header[src]}] {'numeric' if dtt == 'N' else 'text'} field")
            self.datatypes.append(dtt)

        occ = {t: self.targets.count(t) for t in sorted(set(self.targets))}
        self.say(f"# occurrence vector: {occ}")

        self.target = self.header[0]
        self.population = self._build_population(header, rows, source_index)
        self.lookalikes = {str(i): [] for i in range(len(self.population))}
        self.clauses = []

        # Decide the MI-bank blend rate. "auto" reads the table's shape (see
        # _auto_mi_rate); an explicit float overrides. The bank is only built
        # when it will actually be used — on clean/narrow tables the rate is 0
        # and monceai runs the classic random-opposition engine unchanged.
        if self.mi_rate_setting == "auto":
            self.mi_rate = self._auto_mi_rate()
        if self.mi_rate > 0:
            self._build_literal_bank()

        started = time()
        for i in range(self.n_layers):
            self._build_layer()
            eta = round((time() - started) * (self.n_layers - i - 1) / (i + 1), 2)
            self.say(f"# layer {i + 1}/{self.n_layers} · clauses={len(self.clauses)} · eta {eta}s")
        self.say(f"# done · {len(self.clauses)} clauses over {len(self.population)} rows")

    def _infer_target(self, raw):
        """Populate self.targets from raw target cells; return the column type.

        Classification mirrors the classic branch-by-branch detection.
        Regression buckets floats into ordered bins and records their centres.
        """
        if self.mode == "R":
            self.raw_targets = [to_float(x) for x in raw]
            self._bucketize()
            lo, hi = min(self.raw_targets), max(self.raw_targets)
            self.say(f"# regression target range [{lo:.4g}, {hi:.4g}] in {len(self.bin_centres)} bins")
            return "N"  # target cell parses as a float in the population

        uniq = sorted(set(raw))
        universe = set("".join(raw))
        if uniq == ["0", "1"]:
            self.targets = [int(x) for x in raw]
            return "B"
        if uniq == ["False", "True"] or uniq == ["FALSE", "TRUE"]:
            self.targets = [int("T" in x or "t" in x) for x in raw]
            return "B"
        if all(c in "0123456789" for c in universe):
            self.targets = [int("0" + x) for x in raw]
            return "I"
        if is_numeric_universe(universe):
            self.targets = [to_float(x) for x in raw]
            return "N"
        self.targets = list(raw)
        return "T"

    def _bucketize(self):
        """Equal-width binning of raw regression targets into ordered ids.

        self.targets becomes the per-row bin id (the discrete label the clause
        engine works on); self.bin_centres[id] is the mean of the raw values in
        that bin, used later to turn a predicted bin back into a real number.
        """
        lo, hi = min(self.raw_targets), max(self.raw_targets)
        k = max(1, int(self.regression_bins))
        width = (hi - lo) / k if hi > lo else 1.0
        self.targets = []
        for v in self.raw_targets:
            b = int((v - lo) / width) if width else 0
            self.targets.append(min(max(b, 0), k - 1))
        self.bin_centres = {}
        for b in set(self.targets):
            members = [self.raw_targets[i] for i in range(len(self.targets)) if self.targets[i] == b]
            self.bin_centres[b] = mean(members)

    # ---- CSV parsing ------------------------------------------------------- #

    def _split_line(self, line):
        """Split one CSV line into fields, honouring double-quoted commas."""
        line = line.replace("\n", "").replace("\r", "")
        if '"' not in line:
            return line.split(",")
        fields, buf, quoted = [], "", False
        for c in line:
            if c == '"':
                quoted = not quoted
            elif c == "," and not quoted:
                fields.append(buf)
                buf = ""
            else:
                buf += c
        fields.append(buf)
        return fields

    def _read_csv(self, path):
        """Return (header_fields, list_of_row_fields)."""
        with open(path, "r") as f:
            lines = f.readlines()
        header = self._split_line(lines[0])
        rows = [self._split_line(lines[i]) for i in range(1, len(lines))]
        return header, rows

    def _build_population(self, data_header, rows, source_index):
        """Typed feature dicts, dropping later rows that duplicate an earlier one.

        Duplicate detection uses the feature fingerprint only (target excluded),
        so two rows that agree on every feature but disagree on the label — an
        unlearnable contradiction — collapse to the first seen.
        """
        population, seen = [], set()
        for row in rows:
            item, fingerprint = {}, ""
            for i in range(len(self.header)):
                h, dtt, src = self.header[i], self.datatypes[i], source_index[i]
                cell = row[src] if src < len(row) else ""
                if dtt in "IB":
                    item[h] = int(cell) if cell.strip() else 0
                elif dtt == "N":
                    item[h] = to_float(cell)
                else:
                    item[h] = str(cell)
                if i > 0:
                    fingerprint += str(item[h]) + "\x1f"
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            population.append(item)
        return population

    # ---- the literal / clause engine (the heart) --------------------------- #

    def _pick(self, seq):
        """Seeded random choice — the sole source of nondeterminism, tamed."""
        return self._rng.choice(seq)

    # ---- the MI-filtered boolean bank -------------------------------------- #

    def _candidate_literals(self):
        """Generate the raw predicate universe — eyes on everything.

        Every column becomes a spray of boolean questions: numeric columns emit
        ~50 quantile thresholds, text columns emit token-presence tests over the
        commonest tokens plus structural buckets (length, word/comma/sentence
        counts, distinct chars). All in the standard ``[index, value, negated,
        tag]`` form the clause engine already understands.
        """
        cands = []
        n = len(self.population)
        for index in range(1, len(self.header)):
            h, dtt = self.header[index], self.datatypes[index]
            col = [row[h] for row in self.population]
            if dtt in "NIB":
                values = sorted(set(float(v) for v in col))
                if len(values) < 2:
                    continue
                # ~50 evenly-spaced quantile thresholds "x > t".
                k = min(50, len(values) - 1)
                for j in range(1, k + 1):
                    a = values[min(len(values) - 1, j * len(values) // (k + 1))]
                    b = values[min(len(values) - 1, j * len(values) // (k + 1) - 1)]
                    cands.append([index, (a + b) / 2, False, "N"])
            else:
                # Token presence over the ~50 most common whitespace/-/,/: tokens.
                freq = Counter()
                for cell in col:
                    for sep in (" ", "/", ":", "-", ","):
                        for tok in str(cell).split(sep):
                            tok = tok.split("'")[0].split('"')[0]
                            if 1 <= len(tok) <= 20:
                                freq[tok] += 1
                for tok, _ in freq.most_common(50):
                    cands.append([index, tok, False, "T"])
                # Structural buckets: length / words / sentences / distinct chars.
                for tag, fn in (("TN", len),
                                ("TWS", lambda s: len(s.split(" "))),
                                ("TSS", lambda s: len(s.split("."))),
                                ("TLN", lambda s: len(set(s)))):
                    sizes = sorted(set(fn(str(v)) for v in col))
                    if len(sizes) < 2:
                        continue
                    for j in range(1, min(8, len(sizes))):
                        thr = (sizes[j] + sizes[j - 1]) / 2
                        cands.append([index, thr, False, tag])
        return cands

    def _auto_mi_rate(self):
        """Choose the MI-bank blend rate from the table's shape, at train time.

        Empirically (see bench_wide.py / the article's sweep) the MI bank beats
        the classic random-opposition engine only when informative columns are a
        small minority of many columns — signal diluted in noise. When splitting
        uniformly at random already lands on signal often (few columns, or most
        columns informative), the bank adds nothing and can hurt.

        So we cheaply estimate, on a sample, what fraction of feature columns
        carry any mutual information with the target. Few columns, or a rich
        fraction informative -> stay at 0 (identical to the classic engine).
        Many columns with signal diluted below ~1-in-3 -> switch the bank on.
        Purely a function of the data; deterministic under the seed.
        """
        m = len(self.header) - 1                 # feature columns
        if m < 8 or not self.population:
            return 0.0                           # too narrow to be worth steering
        n = len(self.population)
        idx = list(range(n))
        if n > self.mi_sample:
            idx = self._rng.sample(idx, self.mi_sample)
        labels = [self.targets[i] for i in idx]
        if entropy(labels) <= 0.0:
            return 0.0                           # single-class sample: no signal
        sample = [self.population[i] for i in idx]

        # Best-MI predicate per column -> is this column informative at all?
        best_mi = {c: 0.0 for c in range(1, len(self.header))}
        for lit in self._candidate_literals():
            col = lit[0]
            bits = [self.apply_literal(x, lit) is True for x in sample]
            ones = sum(bits)
            if ones == 0 or ones == len(bits):
                continue
            mi = mutual_information(bits, labels)
            if mi > best_mi[col]:
                best_mi[col] = mi
        informative = sum(1 for c in best_mi if best_mi[c] > 0.02)  # >0.02 bits
        fraction = informative / m
        # Diluted signal (informative minority) in a wide table -> engage bank.
        rate = 0.7 if fraction <= 0.34 else 0.0
        self.say(f"# auto mi_rate · {m} feature cols · {informative} informative "
                 f"({100 * fraction:.0f}%) · mi_rate={rate}")
        return rate

    def _build_literal_bank(self):
        """Score every candidate predicate by mutual information; keep the top K.

        Turns "eyes on everything" into "attention on what matters": predicates
        are scored by I(predicate; target) on a sample and the highest-MI ones
        kept as ``self.literal_bank``. Signal survives; noise is discarded.
        """
        if not self.population:
            return
        n = len(self.population)
        idx = list(range(n))
        if n > self.mi_sample:
            idx = self._rng.sample(idx, self.mi_sample)
        sample = [self.population[i] for i in idx]
        labels = [self.targets[i] for i in idx]
        base_h = entropy(labels)

        scored = []
        for lit in self._candidate_literals():
            bits = [self.apply_literal(x, lit) is True for x in sample]
            ones = sum(bits)
            if ones == 0 or ones == len(bits):
                continue  # a predicate that never splits carries no information
            mi = mutual_information(bits, labels)
            if mi > 0.0:
                scored.append((mi, lit))
        scored.sort(key=lambda ml: ml[0], reverse=True)
        self.literal_bank = [(round(mi, 6), lit) for mi, lit in scored[:self.mi_top_k]]
        kept = len(self.literal_bank)
        top = self.literal_bank[0][0] if kept else 0.0
        self.say(f"# boolean bank · H(Y)={base_h:.3f} bits · kept {kept} MI-filtered "
                 f"predicates · top MI={top:.3f}")

    def _bank_literal(self, T, F):
        """Sample a high-MI bank predicate that separates T from F.

        The lookalike model is an ensemble, and ensembles need diverse members.
        Greedy argmax-MI collapses that diversity (identical clauses, worse
        accuracy), so we sample from all separating predicates weighted by MI
        (sharpness mi_greed). Oriented True-on-T/False-on-F; None if nothing
        separates (caller falls back to classic random opposition).
        """
        pool = []
        for mi, lit in self.literal_bank:
            index, value, _negated, tag = lit
            tv = self.apply_literal(T, lit) is True
            fv = self.apply_literal(F, lit) is True
            if tv == fv:
                continue
            oriented = [index, value, not tv, tag]  # flip to be true on T
            if (self.apply_literal(T, oriented) is True and
                    self.apply_literal(F, oriented) is False):
                pool.append((mi, oriented))
        if not pool:
            return None
        # MI-weighted sampling (mi_greed sharpens the preference for signal).
        weights = [(mi + 1e-9) ** self.mi_greed for mi, _ in pool]
        total = sum(weights)
        r = self._rng.random() * total
        upto = 0.0
        for (mi, lit), w in zip(pool, weights):
            upto += w
            if upto >= r:
                return lit
        return pool[-1][1]

    def oppose(self, T, F):
        """Build a literal that is True on row T and False on row F.

        A literal is ``[feature_index, value, negated, tag]``. The engine first
        reaches into the MI-filtered boolean bank for the highest-signal
        predicate that separates T from F — that is what makes monceai learn
        rather than guess. Only if the bank offers nothing does it fall back to
        the classic opposition: numeric split at the midpoint, text opposed
        structurally (length, distinct chars, split counts) or lexically (a
        discriminating token). Same literal objects either way.
        """
        # Primary path (opt-in via mi_rate): an MI-sampled separating predicate.
        # Blended with classic instance-specific opposition to keep diversity.
        if self._rng.random() < self.mi_rate:
            banked = self._bank_literal(T, F)
            if banked is not None:
                return banked

        candidates = [i for i in range(1, len(self.header))
                      if T[self.header[i]] != F[self.header[i]]]
        index = self._pick(candidates)
        h = self.header[index]

        if self.datatypes[index] == "N":
            return [index, (F[h] + T[h]) / 2, T[h] > F[h], "N"]

        # --- text feature ---
        if self._pick([True, False]):  # try a structural opposition first
            options = []
            if len(F[h]) != len(T[h]):
                options.append("TN")
            if len(set(F[h])) != len(set(T[h])):
                options.append("TLN")
            if [c for c in set(F[h]) if c not in T[h]]:
                options.append("FA")
            if [c for c in set(T[h]) if c not in F[h]]:
                options.append("TA")
            if len(F[h].split(" ")) != len(T[h].split(" ")):
                options.append("TWS")
            if len(F[h].split(",")) != len(T[h].split(",")):
                options.append("TPS")
            if len(F[h].split(".")) != len(T[h].split(".")):
                options.append("TSS")
            if options:
                tag = self._pick(options)
                if tag == "TN":
                    return [index, (len(F[h]) + len(T[h])) / 2, len(T[h]) > len(F[h]), "TN"]
                if tag == "TLN":
                    return [index, (len(set(F[h])) + len(set(T[h]))) / 2, len(set(T[h])) > len(set(F[h])), "TLN"]
                if tag == "FA":
                    return [index, self._pick([c for c in set(F[h]) if c not in T[h]]), True, "T"]
                if tag == "TA":
                    return [index, self._pick([c for c in set(T[h]) if c not in F[h]]), False, "T"]
                if tag == "TWS":
                    return [index, (len(F[h].split(" ")) + len(T[h].split(" "))) / 2, len(T[h].split(" ")) > len(F[h].split(" ")), "TWS"]
                if tag == "TPS":
                    return [index, (len(F[h].split(",")) + len(T[h].split(","))) / 2, len(T[h].split(",")) > len(F[h].split(",")), "TPS"]
                if tag == "TSS":
                    return [index, (len(F[h].split(".")) + len(T[h].split("."))) / 2, len(T[h].split(".")) > len(F[h].split(".")), "TSS"]

        # --- lexical opposition: a discriminating token ---
        pros, cons = set(), set()
        for sep in (" ", "/", ":", "-"):
            for token in T[h].split(sep):
                pros.add(token.split("'")[0].split('"')[0])
            for token in F[h].split(sep):
                cons.add(token.split("'")[0].split('"')[0])
        clean_pros = [tok for tok in pros if 0 < len(tok) < max(2, len(T[h])) and tok not in F[h]]
        clean_cons = [tok for tok in cons if 0 < len(tok) < max(2, len(F[h])) and tok not in T[h]]
        choices = ([[index, tok, False, "T"] for tok in clean_pros] +
                   [[index, tok, True, "T"] for tok in clean_cons])
        if choices:
            return self._pick(choices)
        if T[h] not in F[h]:
            return [index, T[h], False, "T"]
        return [index, F[h], True, "T"]

    def apply_literal(self, X, literal):
        """Evaluate one literal on datapoint X. Robust to missing keys."""
        index, value, negated, tag = literal
        h = self.header[index]
        if h not in X:
            return False
        if tag == "N":
            return value <= X[h] if negated else value > X[h]
        if tag == "T":
            return value not in str(X[h]) if negated else value in str(X[h])
        # structural text tags reduce X[h] to an integer measure:
        s = str(X[h])
        if tag == "TN":
            n = len(s)
        elif tag == "TLN":
            n = len(set(s))
        elif tag == "TWS":
            n = len(s.split(" "))
        elif tag == "TPS":
            n = len(s.split(","))
        elif tag == "TSS":
            n = len(s.split("."))
        else:
            return False
        return value <= n if negated else value > n

    def apply_clause(self, X, clause):
        """A clause is an OR: True as soon as any literal fires."""
        for literal in clause:
            if self.apply_literal(X, literal) is True:
                return True
        return False

    def construct_clause(self, F, Ts):
        """A minimal clause: True on every T, False on F, no redundant literal.

        Grow: keep adding literals (each True on some still-uncovered T, False
        on F) until every T is covered. Prune: drop any literal whose removal
        still leaves all Ts covered. The prune is what bounds clause length to
        O(m) and keeps the whole engine at O(m·n^2).
        """
        clause = [self.oppose(self._pick(Ts), F)]
        remaining = [T for T in Ts if not self.apply_literal(T, clause[-1])]
        while remaining:
            clause.append(self.oppose(self._pick(remaining), F))
            remaining = [T for T in remaining if not self.apply_literal(T, clause[-1])]
        i = 0
        while i < len(clause):
            trial = clause[:i] + clause[i + 1:]
            if any(not self.apply_clause(T, trial) for T in Ts):
                i += 1              # literal i is load-bearing, keep it
            else:
                clause = trial      # redundant, drop it
        return clause

    def construct_sat(self, target_value):
        """Cover every row of class ``target_value`` with minimal clauses.

        F-rows are the class we want to characterise; T-rows are everyone else.
        Each clause separates one F from all Ts; we record which same-class rows
        it *fails* on (its "consequence") — those become lookalikes of F.
        """
        Fs = [self.population[i] for i in range(len(self.population)) if self.targets[i] == target_value]
        Ts = [self.population[i] for i in range(len(self.population)) if self.targets[i] != target_value]
        if not Ts:
            return []  # single-class slice: nothing to separate against
        instance = []
        while Fs:
            F = self._pick(Fs)
            clause = self.construct_clause(F, Ts)
            consequence = [i for i in range(len(self.population))
                           if self.targets[i] == target_value and not self.apply_clause(self.population[i], clause)]
            Fs = [f for f in Fs if self.apply_clause(f, clause)]
            instance.append((clause, consequence))
        return instance

    def _build_layer(self):
        """One randomized pass: build separating clauses for every class."""
        for target_value in sorted(set(self.targets)):
            row_conditions = {str(l): [] for l in range(len(self.population))
                              if self.targets[l] == target_value}
            for clause, consequence in self.construct_sat(target_value):
                self.clauses.append(clause)
                for l in consequence:
                    row_conditions[str(l)].append(len(self.clauses) - 1)
            for l in row_conditions:
                self.lookalikes[str(l)].append(row_conditions[str(l)])

    # ---- inference --------------------------------------------------------- #

    def lookalikes_of(self, X):
        """Training rows whose learned condition X fully satisfies.

        A condition is a list of clause indices that must *all* be False on X.
        Returns triples ``(row_index, discrete_target, condition)``.
        """
        clause_false = {i for i in range(len(self.clauses))
                        if not self.apply_clause(X, self.clauses[i])}
        found = []
        for l, conditions in self.lookalikes.items():
            for condition in conditions:
                if all(c in clause_false for c in condition):
                    found.append((int(l), self.targets[int(l)], condition))
        return found

    def predict_proba(self, X):
        """Class-probability vector from lookalike voting (classification)."""
        classes = sorted(set(self.targets))
        looks = self.lookalikes_of(X)
        if not looks:
            return {c: 1 / len(classes) for c in classes}
        return {c: sum(1 for _, t, _ in looks if t == c) / len(looks) for c in classes}

    def predict(self, X):
        """Predicted outcome.

        Classification: the argmax class (ties broken by the class ordering).
        Regression: the lookalike-weighted average of bin centres, falling back
        to the global mean when a query has no lookalikes.
        """
        if self.mode == "R":
            looks = self.lookalikes_of(X)
            if not looks:
                return mean(self.raw_targets)
            return mean(self.bin_centres.get(t, 0.0) for _, t, _ in looks)
        proba = self.predict_proba(X)
        best = max(proba.values())
        return next(c for c in proba if proba[c] == best)

    def confidence(self, X):
        """How strongly the lookalikes agree, in [0, 1] (classification).

        The share of a query's lookalikes that vote for the winning class — a
        cheap, honest uncertainty signal. 0.0 when there are no lookalikes.
        """
        if self.mode == "R":
            return None
        looks = self.lookalikes_of(X)
        if not looks:
            return 0.0
        proba = self.predict_proba(X)
        return max(proba.values())

    # ---- feature importance ------------------------------------------------ #

    def explain(self, X=None):
        """Rank features by how much the model leans on them.

        Global (X is None): count how often each feature appears as a literal
        across all learned clauses — the model's overall reliance on it.
        Local (X given): count only within the clauses that define X's
        lookalikes — why *this* prediction was made. Returns a list of
        ``(feature_name, weight_fraction)`` sorted high to low.
        """
        counts = {h: 0 for h in self.header[1:]}
        if X is None:
            clauses = self.clauses
        else:
            idxs = {c for _, _, condition in self.lookalikes_of(X) for c in condition}
            clauses = [self.clauses[i] for i in idxs]
        total = 0
        for clause in clauses:
            for literal in clause:
                counts[self.header[literal[0]]] += 1
                total += 1
        if total == 0:
            return [(h, 0.0) for h in self.header[1:]]
        ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        return [(h, c / total) for h, c in ranked]

    # ---- explanation ------------------------------------------------------- #

    def _phrase(self, literal):
        """Render one literal as an English fragment."""
        index, value, negated, tag = literal
        h = self.header[index]
        if tag == "N":
            return f"`{h}` is {'≤' if negated else '>'} {value:g}"
        if tag == "T":
            return f"`{h}` {'does not contain' if negated else 'contains'} “{value}”"
        noun = {"TN": "length", "TLN": "distinct characters",
                "TWS": "word count", "TPS": "comma-sections", "TSS": "sentences"}[tag]
        return f"`{h}` {noun} {'≤' if negated else '>'} {value:g}"

    def audit(self, X):
        """A full, human-readable explanation for a single query point."""
        looks = self.lookalikes_of(X)
        out = ["=== AUDIT ===",
               f"query: {X}",
               f"prediction: {self.predict(X)}",
               f"lookalikes: {len(looks)}"]
        if self.mode == "R":
            out.append(f"estimate: {self.predict(X):.4g}")
        else:
            for c, p in self.predict_proba(X).items():
                out.append(f"  P(class = {c}) = {100 * p:.1f}%")
        for row_index, target, condition in looks[:12]:
            literals = [lit for c in condition for lit in self.clauses[c]]
            reasons = "; ".join(self._phrase(lit) for lit in literals) or "(trivially)"
            out.append(f"↳ like row #{row_index} (class {target}) because {reasons}")
        if len(looks) > 12:
            out.append(f"  … and {len(looks) - 12} more lookalikes")
        return "\n".join(out)

    def augment(self, X):
        """X enriched with lookalikes, probabilities, prediction, and audit."""
        Y = dict(X)
        Y["Lookalikes"] = self.lookalikes_of(X)
        if self.mode != "R":
            Y["Probability"] = self.predict_proba(X)
            Y["Confidence"] = self.confidence(X)
        Y["Prediction"] = self.predict(X)
        Y["Audit"] = self.audit(X)
        return Y

    # ---- optional parallel batch inference (the only pandas touchpoint) ---- #

    def predict_frame(self, df, workers=None, augmented=False):
        """Score a pandas DataFrame in parallel; return a pandas Series.

        This is the single place pandas is imported, and it is entirely
        optional — the stdlib core above never needs it. Inference is
        read-only over shared immutable model state, so threads are safe.
        """
        import pandas as pd
        from concurrent.futures import ThreadPoolExecutor

        records = df.to_dict("records")
        fn = self.augment if augmented else self.predict
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(fn, records))
        return pd.Series(results, index=df.index, name=f"{self.target}_pred")

    # ---- validation / pruning --------------------------------------------- #

    def validate(self, rows, keep=0.5):
        """Prune each row's conditions against labelled held-out rows.

        A condition earns a point every time it fires on a same-class row and
        loses one on a wrong-class row; the lowest-scoring conditions are cut
        first, shrinking the model to ``keep`` of its layers. Simple, and it
        turns raw memorisation into something that generalises.
        """
        target_layers = max(1, int(self.n_layers * keep))
        weight = {l: [[0, 0] for _ in conds] for l, conds in self.lookalikes.items()}
        for X in rows:
            if self.target not in X:
                continue
            label = X[self.target]
            false_clauses = {i for i in range(len(self.clauses))
                             if not self.apply_clause(X, self.clauses[i])}
            for l, conditions in self.lookalikes.items():
                row_class = self.targets[int(l)]
                for k, condition in enumerate(conditions):
                    if all(c in false_clauses for c in condition):
                        weight[l][k][row_class == label] += 1
        for l, conditions in self.lookalikes.items():
            ranked = sorted(range(len(conditions)),
                            key=lambda k: weight[l][k][0] - weight[l][k][1])
            keep_idx = set(ranked[:target_layers])
            self.lookalikes[l] = [conditions[k] for k in sorted(keep_idx)]
        self.n_layers = target_layers
        self.say(f"# validated · kept {target_layers} layers")

    # ---- persistence (the single source of truth for the encoding) -------- #

    def to_json(self):
        """Return the full model as a JSON-serialisable dict."""
        return {
            "population": self.population,
            "header": self.header,
            "target": self.target,
            "targets": self.targets,
            "clauses": self.clauses,
            "lookalikes": self.lookalikes,
            "datatypes": self.datatypes,
            "n_layers": self.n_layers,
            "vocal": self.vocal,
            "log": self.log,
            "mode": self.mode,
            "regression_bins": self.regression_bins,
            "raw_targets": self.raw_targets,
            "bin_centres": {str(k): v for k, v in self.bin_centres.items()},
            "seed": self.seed,
            "literal_bank": self.literal_bank,
            "mi_top_k": self.mi_top_k,
            "mi_sample": self.mi_sample,
            "mi_greed": self.mi_greed,
            "mi_rate": self.mi_rate,
            "mi_rate_setting": self.mi_rate_setting,
        }

    def save(self, path="monceai.json"):
        """Persist the model to disk and return its dict form."""
        blob = self.to_json()
        with open(path, "w") as f:
            f.write(json.dumps(blob, indent=2))
        self.say(f"# saved → {path}")
        return blob

    def to_py(self, path="monceai_model.py"):
        """Freeze this trained model into a standalone, dependency-free .py file.

        The generated script hardcodes header, datatypes, clauses, lookalikes,
        targets and bin centres as plain literals, then embeds a tiny inference
        engine mirroring this class exactly. It imports only the stdlib and
        exposes ``predict(X)``. Ship one file, run anywhere.
        """
        def lit(v):
            return repr(v)

        parts = []
        parts.append('#!/usr/bin/env python3\n"""')
        parts.append("Standalone monceai inference — generated by Snake.to_py().")
        parts.append("Hardcoded model, zero dependencies. Exposes predict(X).")
        parts.append('"""\n')
        parts.append(f"MODE = {lit(self.mode)}")
        parts.append(f"HEADER = {lit(self.header)}")
        parts.append(f"DATATYPES = {lit(self.datatypes)}")
        parts.append(f"TARGETS = {lit(self.targets)}")
        parts.append(f"BIN_CENTRES = {lit({int(k): v for k, v in self.bin_centres.items()})}")
        parts.append(f"RAW_TARGET_MEAN = {lit(mean(self.raw_targets) if self.raw_targets else 0.0)}")
        parts.append(f"CLAUSES = {lit(self.clauses)}")
        parts.append(f"LOOKALIKES = {lit(self.lookalikes)}\n")
        # The inference engine, copied verbatim in behaviour from this class.
        parts.append('''
def apply_literal(X, literal):
    index, value, negated, tag = literal
    h = HEADER[index]
    if h not in X:
        return False
    if tag == "N":
        return value <= X[h] if negated else value > X[h]
    if tag == "T":
        return value not in str(X[h]) if negated else value in str(X[h])
    s = str(X[h])
    if tag == "TN":
        n = len(s)
    elif tag == "TLN":
        n = len(set(s))
    elif tag == "TWS":
        n = len(s.split(" "))
    elif tag == "TPS":
        n = len(s.split(","))
    elif tag == "TSS":
        n = len(s.split("."))
    else:
        return False
    return value <= n if negated else value > n


def apply_clause(X, clause):
    for literal in clause:
        if apply_literal(X, literal) is True:
            return True
    return False


def lookalikes(X):
    clause_false = {i for i in range(len(CLAUSES)) if not apply_clause(X, CLAUSES[i])}
    found = []
    for l, conditions in LOOKALIKES.items():
        for condition in conditions:
            if all(c in clause_false for c in condition):
                found.append((int(l), TARGETS[int(l)], condition))
    return found


def predict_proba(X):
    classes = sorted(set(TARGETS))
    looks = lookalikes(X)
    if not looks:
        return {c: 1 / len(classes) for c in classes}
    return {c: sum(1 for _, t, _ in looks if t == c) / len(looks) for c in classes}


def predict(X):
    if MODE == "R":
        looks = lookalikes(X)
        if not looks:
            return RAW_TARGET_MEAN
        vals = [BIN_CENTRES.get(t, 0.0) for _, t, _ in looks]
        return sum(vals) / len(vals)
    proba = predict_proba(X)
    best = max(proba.values())
    return next(c for c in proba if proba[c] == best)


if __name__ == "__main__":
    import json
    import sys
    row = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    print(predict(row))
''')
        code = "\n".join(parts)
        with open(path, "w") as f:
            f.write(code)
        self.say(f"# exported standalone inference → {path} ({len(self.clauses)} clauses)")
        return path

    def load(self, path="monceai.json"):
        """Load a model saved by :meth:`save`. Tolerant of classic v1 files."""
        with open(path, "r") as f:
            m = json.load(f)
        self.population = m["population"]
        self.header = m["header"]
        self.target = m["target"]
        self.targets = m["targets"]
        self.clauses = m["clauses"]
        self.lookalikes = m["lookalikes"]
        self.datatypes = m.get("datatypes", [])
        self.n_layers = m.get("n_layers", 100)
        self.vocal = m.get("vocal", True)
        self.log = m.get("log", self.BANNER)
        self.mode = m.get("mode", "C")
        self.regression_bins = m.get("regression_bins", 16)
        self.raw_targets = m.get("raw_targets", [])
        self.bin_centres = {int(k): v for k, v in m.get("bin_centres", {}).items()}
        self.seed = m.get("seed", None)
        self.literal_bank = [tuple(x) for x in m.get("literal_bank", [])]
        self.mi_top_k = m.get("mi_top_k", 200)
        self.mi_sample = m.get("mi_sample", 400)
        self.mi_greed = m.get("mi_greed", 2.0)
        self.mi_rate = m.get("mi_rate", 0.0)
        self.mi_rate_setting = m.get("mi_rate_setting", self.mi_rate)
        self._rng = Random(self.seed)
        self.say(f"# loaded ← {path}")


# ============================================================================ #
#  SECTION 2 — Self-test (proves the whole thing, zero dependencies)           #
# ============================================================================ #

def _write_temp_csv(path, header, rows):
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(c) for c in row) + "\n")


def selftest():
    """Prove classification, regression, importance, and round-trip on
    synthetic data — no external files, no dependencies. Returns True on pass.
    """
    import os
    import tempfile

    tmp = tempfile.mkdtemp(prefix="monceai_")
    ok = True

    # --- classification: y = 1 iff x > 0, with an uninformative `note` -------
    # `note` cycles through fixed labels unrelated to y, so `x` is the only
    # real signal — the importance check below should surface `x`, not noise.
    header = ["y", "x", "note"]
    noise = ["alpha", "beta", "gamma"]
    rows = []
    for i in range(-20, 21):
        rows.append([1 if i > 0 else 0, i, noise[(i + 20) % 3]])
    csv_path = os.path.join(tmp, "cls.csv")
    _write_temp_csv(csv_path, header, rows)

    def cls_row(i):
        return {"x": i, "note": noise[(i + 20) % 3]}

    model = Snake(csv_path, n_layers=8, vocal=False, target_index=0, seed=1)
    hits = sum(1 for i in range(-20, 21)
               if model.predict(cls_row(i)) == (1 if i > 0 else 0))
    acc = hits / 41
    print(f"  classification train accuracy: {acc:.3f}")
    ok = ok and acc >= 0.95

    conf = model.confidence(cls_row(15))
    print(f"  confidence on a clear point:   {conf:.3f}")
    ok = ok and 0.0 <= conf <= 1.0

    importance = model.explain()
    top = importance[0][0] if importance else None
    print(f"  top global feature:            {top}")
    ok = ok and top == "x"

    # --- round-trip ----------------------------------------------------------
    json_path = os.path.join(tmp, "cls.json")
    model.save(json_path)
    reloaded = Snake(json_path, vocal=False)
    same = all(reloaded.predict(cls_row(i)) == model.predict(cls_row(i))
               for i in range(-20, 21))
    print(f"  JSON round-trip identical:     {same}")
    ok = ok and same

    audit_ok = "AUDIT" in reloaded.audit(cls_row(15))
    print(f"  audit renders:                 {audit_ok}")
    ok = ok and audit_ok

    # --- regression: y = 2*x + 1 ---------------------------------------------
    rheader = ["y", "x"]
    rrows = [[2 * i + 1, i] for i in range(-20, 21)]
    rcsv = os.path.join(tmp, "reg.csv")
    _write_temp_csv(rcsv, rheader, rrows)
    reg = Snake(rcsv, n_layers=10, vocal=False, target_index=0, mode="R", regression_bins=12, seed=2)
    preds = [reg.predict({"x": i}) for i in range(-20, 21)]
    truth = [2 * i + 1 for i in range(-20, 21)]
    mae = mean(abs(p - t) for p, t in zip(preds, truth))
    span = max(truth) - min(truth)
    print(f"  regression MAE / span:         {mae:.2f} / {span} = {mae / span:.3f}")
    ok = ok and (mae / span) < 0.15

    print("  SELFTEST:", "PASS ✓" if ok else "FAIL ✗")
    return ok


# ============================================================================ #
#  SECTION 3 — Command-line interface                                          #
# ============================================================================ #

_HELP = """monceai — the single script that encodes everything

usage:
  python monceai.py train  <csv>  [--target I] [--layers N] [--mode C|R]
                                   [--bins N] [--exclude I,J] [--seed S] [-o OUT]
  python monceai.py predict <model.json> --csv <csv> [--target-name NAME]
  python monceai.py predict <model.json> --row '<json-object>'
  python monceai.py audit   <model.json> --row '<json-object>'
  python monceai.py explain <model.json> [--row '<json-object>']
  python monceai.py freeze  <model.json> <out.py>   # standalone inference script
  python monceai.py selftest
  python monceai.py help
"""


def _arg(argv, flag, default=None):
    """Fetch the value after ``flag`` in argv, or ``default`` if absent."""
    return argv[argv.index(flag) + 1] if flag in argv else default


def _typed_rows_from_csv(model, path):
    """Read a CSV into feature dicts typed according to the model's schema."""
    header, rows = model._read_csv(path)
    col = {h: header.index(h) for h in header}
    typed = []
    for row in rows:
        item = {}
        for i, h in enumerate(model.header):
            if h not in col:
                continue
            cell = row[col[h]] if col[h] < len(row) else ""
            dtt = model.datatypes[i]
            if dtt in "IB":
                item[h] = int(cell) if cell.strip() else 0
            elif dtt == "N":
                item[h] = to_float(cell)
            else:
                item[h] = str(cell)
        typed.append(item)
    return typed


def main(argv):
    if not argv or argv[0] in ("help", "-h", "--help"):
        print(_HELP)
        return 0

    cmd = argv[0]

    if cmd == "selftest":
        return 0 if selftest() else 1

    if cmd == "train":
        csv_path = argv[1]
        exclude = [int(x) for x in _arg(argv, "--exclude", "").split(",") if x != ""]
        seed = _arg(argv, "--seed")
        model = Snake(
            csv_path,
            n_layers=int(_arg(argv, "--layers", "100")),
            target_index=int(_arg(argv, "--target", "0")),
            excluded_features_index=exclude,
            mode=_arg(argv, "--mode", "C"),
            regression_bins=int(_arg(argv, "--bins", "16")),
            seed=int(seed) if seed is not None else None,
            vocal=True,
        )
        model.save(_arg(argv, "-o", "monceai.json"))
        if "--to-py" in argv:
            model.to_py(_arg(argv, "--to-py", "monceai_model.py"))
        return 0

    if cmd == "freeze":
        model = Snake(argv[1], vocal=False)
        model.to_py(argv[2] if len(argv) > 2 else "monceai_model.py")
        return 0

    if cmd in ("predict", "audit", "explain"):
        model = Snake(argv[1], vocal=False)
        row_json = _arg(argv, "--row")

        if cmd == "explain":
            X = json.loads(row_json) if row_json else None
            scope = "local" if X else "global"
            print(f"feature importance ({scope}):")
            for name, weight in model.explain(X):
                bar = "█" * int(round(weight * 40))
                print(f"  {name:<20} {weight * 100:5.1f}%  {bar}")
            return 0

        if row_json:
            X = json.loads(row_json)
            print(model.audit(X) if cmd == "audit" else model.predict(X))
            return 0

        csv_path = _arg(argv, "--csv")
        if not csv_path:
            print("error: provide --row '<json>' or --csv <path>", file=sys.stderr)
            return 2
        for X in _typed_rows_from_csv(model, csv_path):
            print(model.audit(X) + "\n" if cmd == "audit" else model.predict(X))
        return 0

    print(_HELP)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
