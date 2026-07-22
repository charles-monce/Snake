"""
################################################################
#                                                              #
#    Algorithme.ai : Snake         Author : Charles Dana       #
#                                                              #
#    A multiclass & regression .csv handler  -  O(mn^2)        #
#                                                              #
################################################################

Snake is an XAI (Explainable AI) polynomial-time oracle.

It classifies (binary / integer / float-labelled / text multiclass) *and*
regresses, while keeping a full audit trail for every prediction: each
outcome is justified by "lookalikes" drawn from the training set and the
exact logical AND-statements that tie them to the query point.

Design goals of this rewrite (v2), keeping the original spirit intact:

  * Single file, one class, pure Python standard library for the core.
    Nothing is imported that you do not already have.
  * O(mn^2) per layer  (n = rows, m = features) - see COMPLEXITY below.
  * Backward-compatible JSON encoding: models written by the original
    Snake load here, and models written here load there for the shared
    fields.  to_json()/from_json() are the single source of truth for the
    on-disk encoding, and every new capability is expressed through them.
  * New capabilities, all in the same clause/lookalike vocabulary:
      - regression targets (mode="R"), predicted by lookalike averaging
      - explicit multi-target labels, sorted deterministically
      - optional pandas-backed *parallel* batch inference (predict_frame),
        which is the only place pandas is ever touched and is entirely
        optional - the stdlib core never needs it.

COMPLEXITY (why O(mn^2), honestly)
----------------------------------
Let n = number of rows in the population, m = number of feature columns.

  * oppose()          picks a discriminating feature and a literal in O(m).
  * construct_clause() grows a clause literal-by-literal; each added literal
    filters the still-satisfied T set (<= n rows) and, once no T remains,
    a minimisation pass removes redundant literals.  A clause holds at most
    one *useful* literal per feature after minimisation, so its length L is
    O(m); the growth+minimisation cost is O(L * n) = O(m n).
  * construct_sat()    repeats clause construction until every F row (<= n)
    is covered, i.e. O(n) clauses, giving O(m n^2) for one target value.
  * construct_layer()  does the above once per target value.  Summed over
    all target values the F/T sets partition the population, so a layer is
    O(m n^2), NOT O(m n^2 * #targets).

Hence one layer is O(m n^2); a model of `n_layers` layers is
O(n_layers * m * n^2).  The bound assumes minimised clauses stay O(m) long,
which the minimisation pass in construct_clause() enforces.
"""

import json
from random import Random
from time import time


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #

def floatconversion(txt):
    """Parse a float from text, degrading to 0.0 on malformed input.

    Kept identical in spirit to the original so old CSVs behave the same.
    """
    try:
        return float(txt)
    except (ValueError, TypeError):
        return 0.0


def _mean(values):
    """Arithmetic mean of a non-empty iterable, 0.0 on empty. Stdlib only."""
    vals = list(values)
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


# --------------------------------------------------------------------------- #
#  Snake                                                                       #
# --------------------------------------------------------------------------- #

class Snake:
    """The XAI oracle.

    Construct from a ``.csv`` to train, or from a ``.json`` to load a model
    previously written by :meth:`to_json`.  The public surface mirrors the
    original Snake and extends it:

        get_prediction(X)      -> class label (C) or float estimate (R)
        get_probability(X)     -> {class: prob}      (classification only)
        get_lookalikes(X)      -> [[index, target, condition], ...]
        get_audit(X)           -> human-readable audit string
        get_augmented(X)       -> X plus all of the above
        predict_frame(df, ...) -> pandas Series of predictions (parallel)
        make_validation(Xs)    -> prune lookalikes against held-out rows
        to_json(fout)          -> persist (defines the on-disk encoding)

    ``mode`` is ``"C"`` for classification (binary / integer / float-label /
    text multiclass, exactly as before) or ``"R"`` for regression.  In
    regression the raw float targets are bucketed into ordered bins so the
    same clause machinery applies; predictions average the bin-centres of a
    query's lookalikes, yielding a continuous estimate.
    """

    # ---- construction ---------------------------------------------------- #

    def __init__(self, csv_path, n_layers=100, vocal=True, target_index=0,
                 excluded_features_index=None, mode="C", regression_bins=16,
                 seed=None):
        if excluded_features_index is None:
            excluded_features_index = []

        self.log = (
            "################################################################\n"
            "#                                                              #\n"
            "#    Algorithme.ai : Snake         Author : Charles Dana       #\n"
            "#                                                              #\n"
            "#    A multiclass & regression .csv handler  -  O(mn^2)        #\n"
            "#                                                              #\n"
            "################################################################\n"
        )

        # State (0-initialised like the original for JSON shape stability).
        self.population = 0
        self.header = 0
        self.target = 0
        self.targets = 0            # discrete labels used by the clause engine
        self.datatypes = 0
        self.clauses = []
        self.lookalikes = 0
        self.n_layers = n_layers
        self.vocal = vocal
        self.mode = mode            # "C" classification, "R" regression
        self.regression_bins = regression_bins
        self.seed = seed
        # Regression-only metadata (kept in JSON, ignored when classifying).
        self.raw_targets = []       # original numeric targets, pre-bucketing
        self.bin_centres = {}       # {discrete_label: float centre of the bin}

        # Deterministic-when-seeded RNG (original used the global `random`).
        self._rng = Random(seed)

        if ".csv" in csv_path:
            self._train_from_csv(csv_path, target_index, excluded_features_index)
        elif ".json" in csv_path:
            self.from_json(csv_path)
        else:
            self.qprint("# Algorithme.ai : Please provide a .csv or .json path")

    # ---- logging --------------------------------------------------------- #

    def qprint(self, txt):
        if self.vocal:
            print(txt)
        self.log += str(txt) + "\n"

    # ---- training -------------------------------------------------------- #

    def _train_from_csv(self, csv_path, target_index, excluded_features_index):
        self.qprint(
            f"# Initiated Snake ({'regression' if self.mode == 'R' else 'classification'}) "
            f"with {self.n_layers} layers, vocal={self.vocal}, csv={csv_path}"
        )

        with open(csv_path, "r") as f:
            lines = f.readlines()
        header = self.make_bloc_from_line(lines[0])
        rows = lines[1:]

        target_column = header[target_index]
        excl = excluded_features_index + [target_index]
        train_columns = [header[i] for i in range(len(header)) if i not in excl]
        header_index = [target_index] + [i for i in range(len(header)) if i not in excl]
        self.header = [target_column] + train_columns
        self.target = target_column
        self.qprint(f"# Analysis train columns {train_columns}")
        self.qprint(f"# Analysis header {self.header}")

        raw = [self.make_bloc_from_line(row)[target_index] for row in rows]
        self.datatypes = [self._infer_target_datatype(raw)]

        # Feature datatypes (numeric "N" vs text "T").
        for t in range(1, len(self.header)):
            hi = header_index[t]
            values = [self.make_bloc_from_line(row)[hi] for row in rows]
            universe = set("".join(values))
            dtt = "N" if all(c in "+-.0123456789e" for c in universe) else "T"
            kind = "numeric" if dtt == "N" else "text"
            self.qprint(f"#\t[{header[hi]}] {kind} field")
            self.datatypes += [dtt]
        self.qprint(f"# Analysis datatypes {self.datatypes}")

        occ = {t: sum(1 for x in self.targets if x == t) for t in set(self.targets)}
        self.qprint(f"# Algorithme.ai : Occurence Vector {occ}")

        # Build the population (feature dicts), dropping conflicting duplicates.
        self.population = self.make_population(csv_path, drop=True)
        self.target = self.header[0]
        self.lookalikes = {str(l): [] for l in range(len(self.population))}
        self.clauses = []

        t_0 = time()
        for i in range(self.n_layers):
            self.construct_layer()
            remainder = round((time() - t_0) * (self.n_layers - i) / (i + 1), 2)
            self.qprint(f"# Algorithme.ai : Layer {i}/{self.n_layers}, remainder {remainder}s.")

        self.to_json()

    def _infer_target_datatype(self, raw):
        """Set self.targets (+ regression metadata) and return the target datatype.

        Classification mirrors the original branch-by-branch.  Regression
        buckets the raw floats into ordered bins and records bin centres so
        predictions can be turned back into continuous estimates.
        """
        if self.mode == "R":
            self.raw_targets = [floatconversion(x) for x in raw]
            self._bucketize_regression()
            label = f"{min(self.raw_targets):.4g}..{max(self.raw_targets):.4g}"
            self.qprint(f"# Algorithme.ai : Snake regression on {self.target}, range {label}")
            # The target *cell* is a float (parsed into population as "N");
            # the discrete bin ids in self.targets drive the clause engine.
            return "N"

        uniq = sorted(set(raw))
        universe = set("".join(raw))
        if uniq == ["0", "1"]:
            self.targets = [int(x) for x in raw]
            self.qprint(f"# Algorithme.ai : Snake on {self.target} a binary problem 0/1")
            return "B"
        if uniq == ["False", "True"]:
            self.targets = [int("T" in x or "t" in x) for x in raw]
            self.qprint(f"# Algorithme.ai : Snake on {self.target} a binary problem True/False")
            return "B"
        if uniq == ["FALSE", "TRUE"]:
            self.targets = [int("T" in x or "t" in x) for x in raw]
            self.qprint(f"# Algorithme.ai : Snake on {self.target} a binary problem TRUE/FALSE")
            return "B"
        if all(c in "0123456789" for c in universe):
            self.targets = [int("0" + x) for x in raw]
            self.qprint(f"# Algorithme.ai : Snake on {self.target} multiclass integers {'/'.join(uniq)}")
            return "I"
        if all(c in "+-.0123456789e" for c in universe):
            self.targets = [floatconversion(x) for x in raw]
            self.qprint(f"# Algorithme.ai : Snake on {self.target} multiclass floats {'/'.join(uniq)}")
            return "N"
        self.targets = list(raw)
        self.qprint(f"# Algorithme.ai : Snake on {self.target} multiclass text {'/'.join(uniq)}")
        return "T"

    def _bucketize_regression(self):
        """Map raw floats -> ordered integer bins; record each bin's centre.

        Equal-width binning over [min, max].  ``self.targets`` becomes the
        bin id per row; ``self.bin_centres`` maps bin id -> representative
        value used to reconstruct a continuous prediction.
        """
        lo, hi = min(self.raw_targets), max(self.raw_targets)
        k = max(1, int(self.regression_bins))
        width = (hi - lo) / k if hi > lo else 1.0
        self.targets = []
        for v in self.raw_targets:
            b = int((v - lo) / width) if width else 0
            b = min(max(b, 0), k - 1)
            self.targets += [b]
        # Centre of each populated bin = mean of the raw values that fell in it.
        self.bin_centres = {}
        for b in set(self.targets):
            members = [self.raw_targets[i] for i in range(len(self.targets)) if self.targets[i] == b]
            self.bin_centres[b] = _mean(members)

    # ---- persistence (the single source of truth for the encoding) ------- #

    def to_json(self, fout="snakeclassifier.json"):
        """Serialise the model.  This method *defines* the on-disk encoding.

        The original fields are written verbatim for backward compatibility;
        v2 additions (mode, regression metadata, seed) are appended.  A reader
        that ignores the new keys still gets a valid classic model.
        """
        snake_classifier = {
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
            # ---- v2 additions ------------------------------------------- #
            "mode": self.mode,
            "regression_bins": self.regression_bins,
            "raw_targets": self.raw_targets,
            # JSON object keys must be strings; cast bin ids on the way out.
            "bin_centres": {str(k): v for k, v in self.bin_centres.items()},
            "seed": self.seed,
        }
        with open(fout, "w") as f:
            f.write(json.dumps(snake_classifier, indent=2))
        self.qprint(f"Safely saved to {fout}")
        return snake_classifier

    def from_json(self, filepath="snakeclassifier.json"):
        """Load a model written by :meth:`to_json`.

        Tolerant of classic (v1) files: any absent v2 key falls back to a
        classification default, so old artifacts keep working unchanged.
        """
        with open(filepath, "r") as f:
            m = json.load(f)
        self.population = m["population"]
        self.header = m["header"]
        self.target = m["target"]
        self.targets = m["targets"]
        self.clauses = m["clauses"]
        self.lookalikes = m["lookalikes"]
        self.n_layers = m["n_layers"]
        self.vocal = m["vocal"]
        self.datatypes = m["datatypes"]
        self.log = m["log"]
        # ---- v2 additions, with classic-file fallbacks ------------------ #
        self.mode = m.get("mode", "C")
        self.regression_bins = m.get("regression_bins", 16)
        self.raw_targets = m.get("raw_targets", [])
        self.bin_centres = {int(k): v for k, v in m.get("bin_centres", {}).items()}
        self.seed = m.get("seed", None)
        self._rng = Random(self.seed)
        self.qprint(f"# Algorithme.ai : Successful load from {filepath}")

    # ---- CSV parsing ----------------------------------------------------- #

    def make_bloc_from_line(self, line):
        """Parse one CSV line into a list of field strings, honouring quotes."""
        line = line.replace("\n", "")
        if '"' in line:
            quoted = False
            bloc, txt = [], ""
            for c in line:
                if c == '"':
                    quoted = not quoted
                elif c == "," and not quoted:
                    bloc += [txt]
                    txt = ""
                else:
                    txt += c
            bloc += [txt]
            return bloc
        return line.split(",")

    def read_csv(self, fname):
        """Read a pandas-style CSV into (header, rows-of-fields)."""
        if ".csv" not in fname:
            self.qprint("Algorithme.ai: Please input a .csv file")
            return 0, 0
        with open(fname, "r") as f:
            lines = f.readlines()
        header = self.make_bloc_from_line(lines[0])
        data = [self.make_bloc_from_line(lines[t]) for t in range(1, len(lines))]
        return header, data

    def make_population(self, fname, drop=False):
        """Build the list of feature dicts from a CSV, typed per datatypes."""
        POPULATION = []
        data_header, data = self.read_csv(fname)
        mapping_table = {h: -1 for h in self.header}
        for h in mapping_table:
            if h in data_header:
                mapping_table[h] = min(t for t in range(len(data_header)) if data_header[t] == h)
        hashes = set()
        for row in data:
            item_hash, item = "", {}
            for i in range(len(self.header)):
                h = self.header[i]
                dtt = self.datatypes[i]
                if mapping_table[h] == -1:
                    item[h] = "" if dtt == "T" else 0
                else:
                    cell = row[mapping_table[h]]
                    if dtt in "IB":
                        item[h] = int(cell)
                    elif dtt == "N":
                        item[h] = floatconversion(cell)
                    else:
                        item[h] = str(cell)
                if i > 0:
                    item_hash += str(item[h])
            if drop and item_hash in hashes:
                self.qprint(f"# Algorithme.ai : Dropped conflicting row {item}")
                continue
            hashes.add(item_hash)
            POPULATION += [item]
        return POPULATION

    # ---- literal / clause engine ---------------------------------------- #

    def _choice(self, seq):
        """Seedable replacement for random.choice (reproducible models)."""
        return self._rng.choice(seq)

    def oppose(self, T, F):
        """Return a literal true on T and false on F, over a differing feature.

        A literal is ``[feature_index, value, negation, datatype_tag]``.  Text
        features can be opposed structurally (length / alphabet / split counts)
        or lexically (a discriminating token); numeric features split at the
        midpoint.  Behaviour matches the original engine.
        """
        candidates = [i for i in range(1, len(self.header)) if T[self.header[i]] != F[self.header[i]]]
        index = self._choice(candidates)
        h = self.header[index]

        if self.datatypes[index] == "T":
            if self._choice(["Do it", "Don't"]) == "Do it":
                possible = []
                if len(F[h]) != len(T[h]):
                    possible += ["TN"]
                if len(set(F[h])) != len(set(T[h])):
                    possible += ["TLN"]
                if [c for c in set(F[h]) if c not in T[h]]:
                    possible += ["FA"]
                if [c for c in set(T[h]) if c not in F[h]]:
                    possible += ["TA"]
                if len(F[h].split(" ")) != len(T[h].split(" ")):
                    possible += ["TWS"]
                if len(F[h].split(",")) != len(T[h].split(",")):
                    possible += ["TPS"]
                if len(F[h].split(".")) != len(T[h].split(".")):
                    possible += ["TSS"]
                if possible:
                    todo = self._choice(possible)
                    if todo == "TN":
                        return [index, (len(F[h]) + len(T[h])) / 2, len(T[h]) > len(F[h]), "TN"]
                    if todo == "TLN":
                        return [index, (len(set(F[h])) + len(set(T[h]))) / 2, len(set(T[h])) > len(set(F[h])), "TLN"]
                    if todo == "FA":
                        return [index, self._choice([c for c in set(F[h]) if c not in T[h]]), True, "T"]
                    if todo == "TA":
                        return [index, self._choice([c for c in set(T[h]) if c not in F[h]]), False, "T"]
                    if todo == "TWS":
                        return [index, (len(F[h].split(" ")) + len(T[h].split(" "))) / 2, len(T[h].split(" ")) > len(F[h].split(" ")), "TWS"]
                    if todo == "TPS":
                        return [index, (len(F[h].split(",")) + len(T[h].split(","))) / 2, len(T[h].split(",")) > len(F[h].split(",")), "TWS"]
                    if todo == "TSS":
                        return [index, (len(F[h].split(".")) + len(T[h].split("."))) / 2, len(T[h].split(".")) > len(F[h].split(".")), "TWS"]
            pros, cons = set(), set()
            for sep in [" ", "/", ":", "-"]:
                for label in T[h].split(sep):
                    pros.add(label.split("'")[0].split('"')[0])
                for label in F[h].split(sep):
                    cons.add(label.split("'")[0].split('"')[0])
            clean_pros = [l for l in pros if len(l) and len(l) < max(2, len(T[h])) and l not in F[h]]
            clean_cons = [l for l in cons if len(l) and len(l) < max(2, len(F[h])) and l not in T[h]]
            possibilities = ([[index, l, False, "T"] for l in clean_pros] +
                             [[index, l, True, "T"] for l in clean_cons])
            if possibilities:
                return self._choice(possibilities)
            if T[h] != F[h] and T[h] not in F[h]:
                return [index, T[h], False, "T"]
            if T[h] != F[h] and F[h] not in T[h]:
                return [index, F[h], True, "T"]

        if self.datatypes[index] == "N":
            return [index, (F[h] + T[h]) / 2, T[h] > F[h], "N"]

    def apply_literal(self, X, literal):
        """True iff datapoint X satisfies the literal; robust to missing keys."""
        index, value, negat, datat = literal
        h = self.header[index]
        if h not in X:
            return False
        if datat == "TWS":
            n = len(str(X[h]).split(" "))
            return value <= n if negat else value > n
        if datat == "TPS":
            n = len(str(X[h]).split(","))
            return value <= n if negat else value > n
        if datat == "TSS":
            n = len(str(X[h]).split("."))
            return value <= n if negat else value > n
        if datat == "TLN":
            n = len(set(str(X[h])))
            return value <= n if negat else value > n
        if datat == "TN":
            n = len(str(X[h]))
            return value <= n if negat else value > n
        if datat == "T":
            return value not in str(X[h]) if negat else value in str(X[h])
        if datat == "N":
            return value <= X[h] if negat else value > X[h]

    def apply_clause(self, X, clause):
        """OR over literals: True iff any literal in the clause fires."""
        for literal in clause:
            if self.apply_literal(X, literal) is True:
                return True
        return False

    def construct_clause(self, F, Ts):
        """Minimal clause: True on all Ts, False on F, no redundant literal."""
        clause = [self.oppose(self._choice(Ts), F)]
        remainder = [T for T in Ts if self.apply_literal(T, clause[-1]) is False]
        while remainder:
            clause += [self.oppose(self._choice(remainder), F)]
            remainder = [T for T in remainder if self.apply_literal(T, clause[-1]) is False]
        # Drop any literal whose removal still keeps every T satisfied.
        i = 0
        while i < len(clause):
            sub = [clause[j] for j in range(len(clause)) if j != i]
            still_needed = any(self.apply_clause(T, sub) is False for T in Ts)
            if still_needed:
                i += 1
            else:
                clause = sub
        return clause

    def construct_sat(self, target_value):
        """Cover every F (rows == target) with minimal clauses true on the Ts."""
        Fs = [self.population[i] for i in range(len(self.population)) if self.targets[i] == target_value]
        Ts = [self.population[i] for i in range(len(self.population)) if self.targets[i] != target_value]
        sat = []
        while Fs:
            F = self._choice(Fs)
            clause = self.construct_clause(F, Ts)
            consequence = [i for i in range(len(self.population))
                           if self.targets[i] == target_value and self.apply_clause(self.population[i], clause) is False]
            Fs = [f for f in Fs if self.apply_clause(f, clause) is True]
            sat += [[clause, consequence]]
        return sat

    def construct_layer(self):
        """Add one lookalike layer across all target values.  O(m n^2)."""
        target_values = sorted(set(self.targets))
        for target_value in target_values:
            lookalikes = {str(l): [] for l in range(len(self.population)) if self.targets[l] == target_value}
            sat = self.construct_sat(target_value)
            for clause, consequence in sat:
                self.clauses += [clause]
                for l in consequence:
                    lookalikes[str(l)] += [len(self.clauses) - 1]
            for l in lookalikes:
                self.lookalikes[str(l)] += [lookalikes[str(l)]]

    # ---- inference ------------------------------------------------------- #

    def get_lookalikes(self, X):
        """Training rows whose learned condition is fully met by X.

        Returns triples ``[index, discrete_target, condition]`` where the
        condition is the list of clause indices that must all be *negated*
        (false) on X for the lookalike to apply.
        """
        clause_bool = [self.apply_clause(X, clause) for clause in self.clauses]
        negated = {i for i in range(len(clause_bool)) if clause_bool[i] is False}
        out = []
        for l in self.lookalikes:
            for condition in self.lookalikes[l]:
                if all(c in negated for c in condition):
                    out += [[int(l), self.targets[int(l)], condition]]
        return out

    def get_probability(self, X):
        """Class-probability vector from lookalike voting (classification)."""
        target_values = sorted(set(self.targets))
        looks = self.get_lookalikes(X)
        if not looks:
            return {tv: 1 / len(target_values) for tv in target_values}
        return {tv: sum(1 for t in looks if t[1] == tv) / len(looks) for tv in target_values}

    def get_prediction(self, X):
        """Predicted outcome.

        Classification -> the argmax class.  Regression -> lookalike-weighted
        average of bin centres (falls back to the global mean if a query has
        no lookalikes), giving a continuous estimate.
        """
        if self.mode == "R":
            looks = self.get_lookalikes(X)
            if not looks:
                return _mean(self.raw_targets)
            return _mean(self.bin_centres.get(t[1], 0.0) for t in looks)
        prob = self.get_probability(X)
        pr_max = max(prob.values())
        return [t for t in prob if prob[t] == pr_max][0]

    # ---- optional pandas-backed parallel batch inference ----------------- #

    def predict_frame(self, df, workers=None, augmented=False):
        """Predict over a pandas DataFrame in parallel; return a pandas Series.

        This is the ONLY method that touches pandas, and it is fully optional:
        the stdlib core above never needs it.  Each row becomes a feature dict
        and is scored on its own thread (Snake inference is pure-Python read-
        only over shared immutable model state, so threads are safe here).
        """
        import pandas as pd  # optional dependency, imported lazily on use
        from concurrent.futures import ThreadPoolExecutor

        records = df.to_dict("records")
        fn = self.get_augmented if augmented else self.get_prediction
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(fn, records))
        return pd.Series(results, index=df.index, name=f"{self.target}_pred")

    # ---- audit / explainability ----------------------------------------- #

    def get_plain_text_assertion(self, condition, l):
        """Render the AND-statement linking lookalike #l to the query point."""
        head = (
            f"\n        # Datapoint is a lookalike to #{l} of class "
            f"[{self.targets[int(l)]}]\n        - {self.population[int(l)]}\n\n"
            "        Because of the following AND statement that applies to both\n"
        )
        clause = []
        for c in condition:
            clause += self.clauses[c]
        lines = head
        for index, value, negat, datat in clause:
            h = self.header[index]
            if datat == "TWS":
                lines += (f"\n• The textfield {h} has words of length less than [{value}]" if negat
                          else f"\n• The textfield {h} has words of length more than [{value}]")
            elif datat == "TPS":
                lines += (f"\n• The textfield {h} has sections of length less than [{value}]" if negat
                          else f"\n• The textfield {h} has sections of length more than [{value}]")
            elif datat == "TSS":
                lines += (f"\n• The textfield {h} has sentences of length less than [{value}]" if negat
                          else f"\n• The textfield {h} has sentences of length more than [{value}]")
            elif datat == "TLN":
                lines += (f"\n• The textfield {h} has alphabet of length less than [{value}]" if negat
                          else f"\n• The textfield {h} has alphabet of length more than [{value}]")
            elif datat == "TN":
                lines += (f"\n• The textfield {h} has length less than [{value}]" if negat
                          else f"\n• The textfield {h} has more than [{value}]")
            elif datat == "T":
                lines += (f"\n• The text field {h} contains [{value}]" if negat
                          else f"\n• The text field {h} do not contains [{value}]")
            elif datat == "N":
                lines += (f"\n• The numeric field {h} is less than [{value}]" if negat
                          else f"\n• The numeric field {h} is more than [{value}]")
        return lines

    def get_audit(self, X):
        """Full human-readable audit for a query point."""
        looks = self.get_lookalikes(X)
        audit = (
            "### BEGIN AUDIT ###\n"
            f"        ### Datapoint {X}\n"
            f"        ## Number of lookalikes {len(looks)}\n"
            f"        ## Predicted outcome [{self.get_prediction(X)}]\n"
        )
        if self.mode == "R":
            audit += f"\n# Regression estimate : {self.get_prediction(X)}"
        else:
            for tv, p in self.get_probability(X).items():
                audit += f"\n# Probability of being equal to class {tv} : {100 * p}%"
        for triple in looks:
            audit += "\n" + self.get_plain_text_assertion(triple[2], triple[0])
        return audit

    def get_augmented(self, X):
        """Return X enriched with lookalikes, probabilities, prediction, audit."""
        Y = dict(X)
        Y["Lookalikes"] = self.get_lookalikes(X)
        if self.mode != "R":
            Y["Probability"] = self.get_probability(X)
        Y["Prediction"] = self.get_prediction(X)
        Y["Audit"] = self.get_audit(X)
        return Y

    # ---- validation / pruning ------------------------------------------- #

    def make_validation(self, Xs, pruning_coef=0.5):
        """Prune each row's lookalike conditions against labelled held-out Xs.

        Conditions that fire more on wrong-class rows than right-class rows are
        dropped first, shrinking the model from n_layers to n_layers*coef.
        """
        new_n_layers = int(self.n_layers * pruning_coef)
        self.qprint(f"# Algorithme.ai : Validation {self.n_layers} -> {new_n_layers} layers")
        weights = {l: [[0, 0] for _ in self.lookalikes[l]] for l in self.lookalikes}
        for X in Xs:
            if self.target not in X:
                continue
            target = X[self.target]
            unsat = [i for i in range(len(self.clauses)) if self.apply_clause(X, self.clauses[i]) is False]
            for l in weights:
                l_target = self.targets[int(l)]
                for c_index, condition in enumerate(self.lookalikes[l]):
                    if all(i in unsat for i in condition):
                        weights[l][c_index][l_target == target] += 1
        for l in self.lookalikes:
            ranked = sorted(range(len(weights[l])),
                            key=lambda c: weights[l][c][0] - weights[l][c][1])
            kept = ranked[:new_n_layers]
            self.lookalikes[l] = [self.lookalikes[l][c] for c in kept]
        self.n_layers = new_n_layers
