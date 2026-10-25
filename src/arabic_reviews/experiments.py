"""Experiment runner: a code-defined registry of experiments, evaluated on the fixed
3 x 5 grouped CV of the development partition and appended to ``reports/experiments.csv``.

Only ``split.load_dev()`` is used, so the held-out test partition is unreachable from here.
Each fold fits a fresh pipeline on its training rows only.
"""

from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from . import config, split
from .features import build_pipeline

TASKS = {"sentiment": "sentiment", "aspect": "aspect"}
LABELS = {"sentiment": list(config.SENTIMENTS), "aspect": list(config.ASPECTS)}
COLUMNS = [
    "exp_id", "timestamp", "git_sha", "task", "preprocessing", "features", "model", "params",
    "cv_macro_f1_mean", "cv_macro_f1_std", "fold_scores", "balanced_acc_mean",
    "accuracy_mean", "fit_seconds", "notes",
]
EXPERIMENTS_FILE = config.REPORTS_DIR / "experiments.csv"
ABLATIONS_FILE = config.REPORTS_DIR / "ablations.md"
MODELS_FILE = config.REPORTS_DIR / "models.md"
ERRORS_FILE = config.REPORTS_DIR / "error_analysis.md"
MIN_DELTA = 0.01  # acceptance rule (PROJECT_PLAN.md 1.1): mean paired gain and fold wins
MIN_WINS = 11


def _spec(name: str, model: str, representation: str = "word11", normalisation: str = "none",
          class_weight=None, params=None, notes: str = "", **options) -> dict:
    """Experiment spec; ``options`` may set symbols, digits, negation, stopwords, textstats."""
    return dict(name=name, model=model, representation=representation, normalisation=normalisation,
                class_weight=class_weight, params=params or {}, notes=notes, **options)


def describe_preprocessing(spec: dict) -> str:
    """Compact preprocessing label for the log ("none" when nothing is applied)."""
    parts = [spec["normalisation"]] if spec["normalisation"] != "none" else []
    for key in ("symbols", "digits"):
        if spec.get(key, "keep") != "keep":
            parts.append(f"{key}={spec[key]}")
    parts += [flag for flag in ("negation", "stopwords") if spec.get(flag)]
    return "|".join(parts) or "none"


# Specs shared by both tasks; the experiment id is "<task>/<name>".
REGISTRY: list[dict] = [
    _spec("dummy_most_frequent", "dummy_most_frequent", notes="baseline"),
    _spec("dummy_stratified", "dummy_stratified", notes="baseline"),
    _spec("logreg_word11", "logreg", "word11", notes="reference model; axis 1 start"),
    _spec("logreg_word12", "logreg", "word12", notes="axis 1: representation"),
    _spec("logreg_char25", "logreg", "char25", notes="axis 1: representation"),
    _spec("logreg_word11+char25", "logreg", "word11+char25", notes="axis 1: representation"),
]


def exp_id(task: str, name: str) -> str:
    return f"{task}/{name}"


def cv_scores(spec: dict, dev: pd.DataFrame, task: str) -> dict:
    """Score a spec on every (repeat, fold); returns per-fold lists in repeat-major order."""
    y = dev[TASKS[task]].to_numpy()
    X = dev["text"].to_numpy()
    pipeline = build_pipeline(spec)
    f1s, bal, acc, seconds = [], [], [], 0.0
    for col in split.CV_COLUMNS:
        folds = dev[col].to_numpy()
        for k in sorted(set(folds)):
            train, valid = folds != k, folds == k
            model = clone(pipeline)
            start = time.perf_counter()
            model.fit(X[train], y[train])
            seconds += time.perf_counter() - start
            pred = model.predict(X[valid])
            f1s.append(f1_score(y[valid], pred, average="macro", labels=LABELS[task], zero_division=0))
            bal.append(balanced_accuracy_score(y[valid], pred))
            acc.append(accuracy_score(y[valid], pred))
    return dict(f1=f1s, bal=bal, acc=acc, seconds=seconds)


def compare(base: list[float], cand: list[float]) -> dict:
    """Paired comparison on identical folds under the pre-declared acceptance rule."""
    diff = np.asarray(cand) - np.asarray(base)
    wins = int((diff > 0).sum())
    mean = float(diff.mean())
    return dict(mean_delta=mean, wins=wins, n=len(diff),
                accepted=bool(mean >= MIN_DELTA and wins >= MIN_WINS))


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, encoding="utf-8",
                          cwd=config.ROOT, check=True).stdout


def _git_sha() -> str:
    """Short HEAD hash, with "+dirty" when code differs from the commit: tracked changes outside
    reports/, untracked files under src/ or tests/, or a log that is not the committed log plus
    appended rows (the log is an input of the M4 decisions)."""
    try:
        sha = _git("rev-parse", "--short", "HEAD").strip()
        dirty = _git("status", "--porcelain", "--untracked-files=no", "--", ".", ":!reports").strip()
        dirty += _git("status", "--porcelain", "--untracked-files=normal", "--", "src", "tests").strip()
        inside = EXPERIMENTS_FILE.resolve().is_relative_to(config.ROOT.resolve())
        log = EXPERIMENTS_FILE.resolve().relative_to(config.ROOT.resolve()).as_posix() if inside else ""
        if log and _git("ls-files", log).strip():
            committed = _git("show", f"HEAD:{log}")
            current = EXPERIMENTS_FILE.read_text(encoding="utf-8") if EXPERIMENTS_FILE.exists() else ""
            dirty += "" if current.startswith(committed) else "log"
        return sha + ("+dirty" if dirty else "")
    except (OSError, ValueError, subprocess.CalledProcessError):
        return "unknown"


def read_log() -> pd.DataFrame:
    if EXPERIMENTS_FILE.exists():
        return pd.read_csv(EXPERIMENTS_FILE, keep_default_na=False)
    return pd.DataFrame(columns=COLUMNS)


def fold_scores(log: pd.DataFrame, eid: str) -> list[float]:
    rows = log.loc[log.exp_id == eid, "fold_scores"]
    if rows.empty:
        raise KeyError(f"experiment {eid} is not in the log")
    return json.loads(rows.iloc[0])


def row_identity(spec: dict) -> dict:
    """The log columns that identify what an experiment ran."""
    return dict(
        preprocessing=describe_preprocessing(spec),
        features=spec["representation"] + ("+textstats" if spec.get("textstats") else ""),
        params=json.dumps({**spec["params"], "class_weight": spec["class_weight"],
                           **({"tfidf": spec["tfidf"]} if spec.get("tfidf") else {})}, sort_keys=True),
    )


def _append(row: dict) -> None:
    """Append one finished experiment at once, so a crash never loses completed rows."""
    EXPERIMENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    fresh = not EXPERIMENTS_FILE.exists()
    pd.DataFrame([row], columns=COLUMNS).to_csv(
        EXPERIMENTS_FILE, mode="a", header=fresh, index=False, lineterminator="\n")


def run_experiment(task: str, spec: dict, dev: pd.DataFrame) -> dict:
    """Evaluate one spec on the 3 x 5 CV and append its row to the log."""
    res = cv_scores(spec, dev, task)
    row = dict(
        exp_id=exp_id(task, spec["name"]),
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        git_sha=_git_sha(), task=task, model=spec["model"], **row_identity(spec),
        cv_macro_f1_mean=round(float(np.mean(res["f1"])), 4),
        cv_macro_f1_std=round(float(np.std(res["f1"], ddof=1)), 4),
        fold_scores=json.dumps([round(float(v), 6) for v in res["f1"]]),
        balanced_acc_mean=round(float(np.mean(res["bal"])), 4),
        accuracy_mean=round(float(np.mean(res["acc"])), 4),
        fit_seconds=round(res["seconds"], 2), notes=spec["notes"],
    )
    _append(row)
    return row


def run_pending(dev: pd.DataFrame | None = None, registry: list[dict] | None = None) -> pd.DataFrame:
    """Run every registered experiment not yet in the log (append-only; finished
    experiments are never re-run or edited). Returns the rows appended."""
    dev = split.load_dev() if dev is None else dev
    registry = REGISTRY if registry is None else registry
    done = set(read_log().exp_id)
    rows = [run_experiment(task, spec, dev)
            for task in TASKS for spec in registry if exp_id(task, spec["name"]) not in done]
    return pd.DataFrame(rows, columns=COLUMNS)


AXIS1_CHAIN = ["logreg_word11", "logreg_word12", "logreg_char25", "logreg_word11+char25"]


def decide_chain(log: pd.DataFrame, task: str, chain: list[str]) -> list[dict]:
    """Walk an ordered axis: each candidate is compared with the current winner under the
    acceptance rule, and the first entry is the starting winner."""
    winner, steps = chain[0], []
    for cand in chain[1:]:
        res = compare(fold_scores(log, exp_id(task, winner)), fold_scores(log, exp_id(task, cand)))
        steps.append(dict(task=task, winner_before=winner, candidate=cand, **res))
        if res["accepted"]:
            winner = cand
    if steps:
        steps[-1]["final_winner"] = winner
    return steps


def axis_markdown(log: pd.DataFrame, title: str, chain: list[str]) -> str:
    lines = [f"## {title}", "",
             f"Rule: mean paired Macro-F1 gain >= {MIN_DELTA} and gain in >= {MIN_WINS} of 15 folds.", ""]
    for task in TASKS:
        lines += [f"### {task}", "", "| candidate | vs current winner | mean gain | folds won | decision |",
                  "|---|---|---|---|---|"]
        steps = decide_chain(log, task, chain)
        for s in steps:
            lines.append(f"| {s['candidate']} | {s['winner_before']} | {s['mean_delta']:+.4f} | "
                         f"{s['wins']}/{s['n']} | {'accepted' if s['accepted'] else 'rejected'} |")
        final = steps[-1]["final_winner"] if steps else chain[0]
        lines += ["", f"Winner for {task}: `{final}`", ""]
    return "\n".join(lines)


# Axes 2-6 in the pre-declared order (PROJECT_PLAN.md, M3). Every candidate is the current
# winner with one change applied, so it is compared with the winner on the same 15 folds.
AXES = [
    ("a2", "Axis 2 - normalisation",
     [("light", dict(normalisation="light")), ("full", dict(normalisation="full"))]),
    ("a3", "Axis 3 - symbols and digits",
     [("symbols_separate", dict(symbols="separate")), ("symbols_remove", dict(symbols="remove")),
      ("digits_token", dict(digits="token"))]),
    ("a4", "Axis 4 - negation and stopwords",
     [("negation", dict(negation=True)), ("stopwords", dict(stopwords=True))]),
    ("a5", "Axis 5 - text statistics", [("textstats", dict(textstats=True))]),
    ("a6", "Axis 6 - class weighting", [("balanced", dict(class_weight="balanced"))]),
]


def axis1_winner(log: pd.DataFrame, task: str) -> str:
    steps = decide_chain(log, task, AXIS1_CHAIN)
    return steps[-1]["final_winner"]


def resolve_axes(task: str, dev: pd.DataFrame | None = None) -> tuple[list[dict], dict]:
    """Walk axes 2-6 for one task. With ``dev`` given, missing experiments are run and logged;
    otherwise every experiment must already be in the log. Returns (decision rows, final spec)."""
    by_name = {s["name"]: s for s in REGISTRY}
    winner = dict(by_name[axis1_winner(read_log(), task)])
    rows = []
    for key, title, variants in AXES:
        for variant, change in variants:
            name = f"{key}_{variant}"
            spec = dict(winner, name=name, notes=f"{title}: {variant} (base: {winner['name']})", **change)
            if dev is not None and exp_id(task, name) not in set(read_log().exp_id):
                run_experiment(task, spec, dev)
            log = read_log()
            logged = log.loc[log.exp_id == exp_id(task, name)]
            if logged.empty:
                raise KeyError(f"experiment {exp_id(task, name)} is not in the log")
            stale = {k: v for k, v in row_identity(spec).items() if logged.iloc[0][k] != v}
            if stale:
                raise ValueError(f"logged {exp_id(task, name)} does not match the current winner chain: {stale}")
            res = compare(fold_scores(log, exp_id(task, winner["name"])), fold_scores(log, exp_id(task, name)))
            rows.append(dict(axis=title, candidate=name, winner_before=winner["name"], **res))
            if res["accepted"]:
                winner = spec
    return rows, winner


def _spec_summary(spec: dict) -> dict:
    keys = ("representation", "normalisation", "symbols", "digits", "negation", "stopwords",
            "textstats", "class_weight")
    return {k: spec.get(k) for k in keys if k in spec}


def axes_markdown(task_results: dict[str, tuple[list[dict], dict]]) -> str:
    lines = ["## Axes 2-6", "",
             "Each candidate is the current winner plus one change, compared on the same 15 folds.", ""]
    for task, (rows, final) in task_results.items():
        lines += [f"### {task}", "", "| axis | candidate | vs current winner | mean gain | folds won | decision |",
                  "|---|---|---|---|---|---|"]
        for r in rows:
            lines.append(f"| {r['axis']} | {r['candidate']} | {r['winner_before']} | {r['mean_delta']:+.4f} | "
                         f"{r['wins']}/{r['n']} | {'accepted' if r['accepted'] else 'rejected'} |")
        lines += ["", f"Final feature set for {task}: `{json.dumps(_spec_summary(final), sort_keys=True)}`", ""]
    return "\n".join(lines)


# ---- M4: model families and limited tuning on the frozen M3 feature sets -------------------

FAMILIES = [
    ("m4_linsvm", dict(model="linsvm"), "linear SVM: hinge loss is often stronger than log-loss on sparse TF-IDF"),
    ("m4_cnb", dict(model="cnb", class_weight=None),
     "Complement NB: cheap, designed for imbalanced text; no class weights"),
]
# Regularisation grids per family; the first entry of each list is the strongest regularisation
# (simplest model) and the plan's default value is in the grid.
GRIDS = {"logreg": ("C", [0.1, 0.3, 1, 3, 10]), "linsvm": ("C", [0.1, 0.3, 1, 3, 10]),
         "cnb": ("alpha", [3, 1, 0.3, 0.1, 0.03])}
GRID_DEFAULT = 1  # the library default of C and alpha, used by every M3 experiment
TFIDF_STEPS = [("min_df2", {"min_df": 2}), ("sublinear", {"sublinear_tf": True})]


def one_se_choice(means: list[float], stds: list[float], n: int = 15) -> int:
    """One-standard-error rule: the simplest candidate (lowest index) whose mean is within one
    standard error of the best mean, where the error is that of the best candidate's folds."""
    best = int(np.argmax(means))
    bar = means[best] - stds[best] / np.sqrt(n)
    return next(i for i, m in enumerate(means) if m >= bar)


def gate_complex_pick(grid: list, scores: list[list[float]], pick: int) -> tuple[int, dict | None]:
    """A grid value weaker-regularised than the default (later in the grid) adds complexity, so
    it must also pass the acceptance rule against the default; otherwise the default is kept
    (PROJECT_PLAN.md 1.1 and change log). Returns (index used, comparison or None)."""
    default = grid.index(GRID_DEFAULT)
    if pick <= default:
        return pick, None
    gate = compare(scores[default], scores[pick])
    return (pick if gate["accepted"] else default), gate


def _log_scores(log: pd.DataFrame, task: str, name: str) -> list[float]:
    return fold_scores(log, exp_id(task, name))


def _ensure(task: str, spec: dict, dev: pd.DataFrame | None) -> None:
    """Run and log a spec when ``dev`` is given; check a logged row still matches the spec."""
    eid = exp_id(task, spec["name"])
    if dev is not None and eid not in set(read_log().exp_id):
        run_experiment(task, spec, dev)
    logged = read_log().loc[lambda d: d.exp_id == eid]
    if logged.empty:
        raise KeyError(f"experiment {eid} is not in the log")
    stale = {k: v for k, v in row_identity(spec).items() if logged.iloc[0][k] != v}
    if stale or logged.iloc[0]["model"] != spec["model"]:
        raise ValueError(f"logged {eid} does not match the current M4 chain: {stale}")


def resolve_m4(task: str, dev: pd.DataFrame | None = None) -> dict:
    """Stage A compares model families with the M3 final spec under the acceptance rule;
    stage B tunes the regularisation strength of the surviving family (one-standard-error rule),
    then tries min_df=2 and sublinear_tf one at a time under the acceptance rule."""
    _, base = resolve_axes(task)
    base = dict(base)
    families, chosen, best_gain = [], base, 0.0
    for key, change, hypothesis in FAMILIES:
        spec = dict(base, name=key, notes=f"M4 family: {hypothesis}", params={}, **change)
        _ensure(task, spec, dev)
        res = compare(_log_scores(read_log(), task, base["name"]), _log_scores(read_log(), task, key))
        families.append(dict(candidate=key, vs=base["name"], **res))
        if res["accepted"] and res["mean_delta"] > best_gain:
            chosen, best_gain = spec, res["mean_delta"]
    param, grid = GRIDS[chosen["model"]]
    family = chosen["model"]
    grid_specs = []
    for value in grid:
        name = f"m4_{family}_{param}{value}"
        spec = dict(chosen, name=name, params={param: value}, notes=f"M4 tuning: {param}={value} on {family}")
        _ensure(task, spec, dev)
        grid_specs.append(spec)
    log = read_log()
    scores = [_log_scores(log, task, s["name"]) for s in grid_specs]
    means = [float(np.mean(v)) for v in scores]
    stds = [float(np.std(v, ddof=1)) for v in scores]
    one_se = one_se_choice(means, stds)
    pick, gate = gate_complex_pick(grid, scores, one_se)
    tuning = [dict(value=v, mean=m, std=sd, folds=sc) for v, m, sd, sc in zip(grid, means, stds, scores)]
    winner = grid_specs[pick]
    steps = []
    for label, option in TFIDF_STEPS:
        tfidf = {**winner.get("tfidf", {}), **option}
        spec = dict(winner, name=f"{winner['name']}+{label}", tfidf=tfidf,
                    notes=f"M4 tuning: {label} on {winner['name']}")
        _ensure(task, spec, dev)
        res = compare(_log_scores(read_log(), task, winner["name"]), _log_scores(read_log(), task, spec["name"]))
        steps.append(dict(candidate=spec["name"], vs=winner["name"], **res))
        if res["accepted"]:
            winner = spec
    final = compare(_log_scores(read_log(), task, base["name"]), _log_scores(read_log(), task, winner["name"]))
    return dict(base=base, families=families, family=family, param=param, tuning=tuning,
                one_se=grid[one_se], gate=gate, picked=grid[pick], tfidf_steps=steps, final=winner,
                final_vs_base=final)


def m4_markdown(results: dict[str, dict]) -> str:
    lines = ["# Model families and limited tuning (development CV only)", "",
             "Same 3 x 5 grouped CV, same rule: mean paired Macro-F1 gain >= "
             f"{MIN_DELTA} and gain in >= {MIN_WINS} of 15 folds. Features are the frozen M3 sets.", ""]
    for task, r in results.items():
        lines += [f"## {task}", "", f"Reference (M3 final): `{r['base']['name']}`", "",
                  "### Stage A - model families", "",
                  "| candidate | vs | mean gain | folds won | decision |", "|---|---|---|---|---|"]
        for s in r["families"]:
            lines.append(f"| {s['candidate']} | {s['vs']} | {s['mean_delta']:+.4f} | {s['wins']}/{s['n']} | "
                         f"{'accepted' if s['accepted'] else 'rejected'} |")
        lines += ["", f"Family carried to tuning: `{r['family']}`", "",
                  f"### Stage B - {r['param']} grid (one-standard-error rule, simplest first)", "",
                  f"| {r['param']} | mean Macro-F1 | std |", "|---|---|---|"]
        for t in r["tuning"]:
            mark = (" (one-SE pick)" if t["value"] == r["one_se"] else "") + \
                   (" **(used)**" if t["value"] == r["picked"] else "")
            lines.append(f"| {t['value']}{mark} | {t['mean']:.4f} | {t['std']:.4f} |")
        g, param = r["gate"], r["param"]
        if g is None:
            gate_text = "It is not weaker-regularised than the default, so no further check applies."
        else:
            outcome = "passes, so it is used" if g["accepted"] else f"fails, so the default {param} = {GRID_DEFAULT} is used"
            gate_text = (f"It is weaker-regularised than the default {param} = {GRID_DEFAULT}, i.e. it adds complexity, "
                         f"so it must also pass the acceptance rule: {g['mean_delta']:+.4f} mean, "
                         f"{g['wins']}/{g['n']} folds; it {outcome}.")
        lines += ["", f"The one-standard-error pick is {param} = {r['one_se']}. {gate_text} The standard error "
                  "is the sd of the 15 fold scores over sqrt(15); the three repeats reuse the same reviews, so this "
                  "understates the error, and with sqrt(5) the pick would move one step towards stronger "
                  "regularisation. The rule was committed before the grid was run and is not changed afterwards."]
        lines += ["", "| tf-idf option | vs | mean gain | folds won | decision |", "|---|---|---|---|---|"]
        for s in r["tfidf_steps"]:
            lines.append(f"| {s['candidate']} | {s['vs']} | {s['mean_delta']:+.4f} | {s['wins']}/{s['n']} | "
                         f"{'accepted' if s['accepted'] else 'rejected'} |")
        f = r["final_vs_base"]
        lines += ["", f"Tuned configuration `{r['final']['name']}` vs M3 reference: {f['mean_delta']:+.4f} mean, "
                      f"{f['wins']}/{f['n']} folds ({'passes' if f['accepted'] else 'does not pass'} the acceptance rule).", ""]
    lines += ["## Limits of this search", "",
              "- Families were compared at their default regularisation only; a tuned linear SVM might do slightly "
              "better than in Stage A, where it did not pass the rule for either task.",
              "- `min_df=2` shrinks the vocabulary and is arguably the simpler option; it was nevertheless treated as "
              "a candidate change that has to pass the rule, as declared before the run. Its gains were within noise "
              "for both tasks.", ""]
    return "\n".join(lines)


# ---- M4: out-of-fold predictions and error analysis ----------------------------------------

def oof_predictions(spec: dict, dev: pd.DataFrame, task: str, repeat: int = 0,
                    column: str | None = None) -> pd.DataFrame:
    """Predict every development row once, from a model fitted on the other four folds of one
    CV repeat (``column`` overrides the repeat with any fold column of ``dev``). Returns rid,
    fold, true label, predicted label (and the top class probability when the model has one)."""
    y = dev[TASKS[task]].to_numpy()
    X = dev["text"].to_numpy()
    folds = dev[column or split.CV_COLUMNS[repeat]].to_numpy()
    pred = np.empty(len(dev), dtype=object)
    score = np.full(len(dev), np.nan)
    for k in sorted(set(folds)):
        valid = folds == k
        model = clone(build_pipeline(spec)).fit(X[~valid], y[~valid])
        pred[valid] = model.predict(X[valid])
        if hasattr(model, "predict_proba"):
            score[valid] = model.predict_proba(X[valid]).max(axis=1)
    return pd.DataFrame(dict(rid=dev["rid"].to_numpy(), fold=folds, y=y, pred=pred, score=score))


ERROR_CATEGORIES_FILE = config.REPORTS_DIR / "error_categories.csv"
ERROR_SAMPLE_SIZE = {"sentiment": 45, "aspect": 30}
ERROR_SAMPLE_SEED = 7
CATEGORY_NAMES = {
    "lexical_variety": "lexical or semantic variety not captured by n-grams",
    "multi_aspect": "several aspects / mixed polarity in one text",
    "ambiguous_neutral": "borderline neutral",
    "dialect_spelling": "dialect or irregular spelling",
    "label_noise": "label noise (gold label looks wrong or debatable)",
    "negation": "negation",
    "sarcasm_or_implicit": "sarcasm or implicit opinion",
    "too_short_or_uninformative": "too short or uninformative",
    "other": "other",
}
GATE_CATEGORY, GATE_SHARE = "lexical_variety", 0.30  # deeper-representation gate (PROJECT_PLAN.md M4)


def oof_error_sample(oof: pd.DataFrame, n: int, seed: int = ERROR_SAMPLE_SEED) -> pd.DataFrame:
    """The seeded random sample of out-of-fold errors that is categorised by reading."""
    return oof[oof.y != oof.pred].sample(n=n, random_state=seed)


def gate_decision(cats: pd.DataFrame) -> dict[str, dict]:
    """Per task: share of the gate category, whether it is the largest, whether the gate opens."""
    out = {}
    for task, g in cats.groupby("task", sort=False):
        counts = g.category.value_counts()
        share = counts.get(GATE_CATEGORY, 0) / len(g)
        largest = counts.get(GATE_CATEGORY, 0) == counts.max() and (counts == counts.max()).sum() == 1
        out[task] = dict(n=len(g), share=float(share), largest=bool(largest), opens=bool(largest and share >= GATE_SHARE))
    return out


def categories_markdown(cats: pd.DataFrame, finals: dict[str, str] | None = None) -> str:
    tasks = list(dict.fromkeys(cats.task))
    sizes = {t: int((cats.task == t).sum()) for t in tasks}
    models = {t: cats.loc[cats.task == t, "model"].iloc[0] for t in tasks}
    lines = ["## Manual classification of real errors", "",
             f"A seeded random sample of out-of-fold errors (repeat 0, `oof_error_sample`, seed {ERROR_SAMPLE_SEED}; "
             + ", ".join(f"{sizes[t]} {t} errors of `{models[t]}`" for t in tasks)
             + ") was read, and each error was assigned one primary category from its text, gold label and "
             "prediction. The per-review categories (without texts) are in `reports/error_categories.csv`. The "
             "categories are judgements, not measurements, and the samples are small.", "",
             "| category | " + " | ".join(f"{t} (n={sizes[t]})" for t in tasks) + " |",
             "|---" * (len(tasks) + 1) + "|"]
    for key, label in CATEGORY_NAMES.items():
        cells = []
        for t in tasks:
            k = int(((cats.task == t) & (cats.category == key)).sum())
            cells.append(f"{k} ({k / sizes[t]:.0%})" if k else "0")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    gate = gate_decision(cats)
    lines += ["", f"Pre-declared gate for the optional pre-trained sentence-vector experiment: the category "
              f"\"{CATEGORY_NAMES[GATE_CATEGORY]}\" must be the largest and at least {GATE_SHARE:.0%} of the sample."]
    for t, d in gate.items():
        lines.append(f"- {t}: {d['share']:.0%}, {'the largest' if d['largest'] else 'not the largest'} category "
                     f"-> {'gate opens' if d['opens'] else 'gate stays closed'}.")
    for t in tasks:
        if finals and models[t] != finals[t]:
            lines += ["", f"The {t} sample was drawn from the errors of `{models[t]}`, the configuration chosen "
                      f"before the complexity check on the regularisation grid was added; the final configuration "
                      f"is `{finals[t]}` (`reports/models.md`). The gate concerns the representation, which both "
                      "share, not the regularisation strength."]
    return "\n".join(lines) + "\n"


def per_class_table(oof: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    from sklearn.metrics import precision_recall_fscore_support
    p, r, f, support = precision_recall_fscore_support(oof.y, oof.pred, labels=labels, zero_division=0)
    return pd.DataFrame(dict(label=labels, precision=p, recall=r, f1=f, support=support))


def confusion(oof: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    from sklearn.metrics import confusion_matrix
    return pd.DataFrame(confusion_matrix(oof.y, oof.pred, labels=labels), index=labels, columns=labels)


def error_markdown(task: str, spec: dict, oof: pd.DataFrame) -> str:
    labels = LABELS[task]
    table = per_class_table(oof, labels)
    macro = float(table.f1.mean())
    lines = [f"## {task}", "", f"Model: `{spec['name']}` ({spec['model']}), repeat 0 out-of-fold predictions "
             f"on {len(oof)} development reviews; Macro-F1 {macro:.4f}, accuracy {(oof.y == oof.pred).mean():.4f}, "
             f"errors {(oof.y != oof.pred).sum()}.", "",
             "| class | precision | recall | F1 | support |", "|---|---|---|---|---|"]
    for r in table.itertuples():
        lines.append(f"| {r.label} | {r.precision:.3f} | {r.recall:.3f} | {r.f1:.3f} | {r.support} |")
    lines += ["", "Confusion matrix (rows = true, columns = predicted):", "",
              "| true \\ pred | " + " | ".join(labels) + " |", "|---" * (len(labels) + 1) + "|"]
    cm = confusion(oof, labels)
    for label in labels:
        lines.append(f"| {label} | " + " | ".join(str(int(v)) for v in cm.loc[label]) + " |")
    return "\n".join(lines) + "\n"
