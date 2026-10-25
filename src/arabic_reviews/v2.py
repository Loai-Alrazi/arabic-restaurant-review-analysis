"""Version 2 of the sentiment model (``docs/V2_PLAN.md``), on the development partition only.

``run_experiments`` scores three pre-declared feature candidates against the frozen version-1
configuration on the selection folds, estimates the whole selection procedure with nested grouped
CV on fresh seeds, checks the stability of the chosen candidate on those folds, derives certainty
levels per predicted class and reports calibration. ``build`` freezes the version-2 protocol and
fits the version-2 pipelines on the whole development partition.

The old held-out test is never read: both commands run inside ``guard()``, which refuses to open
the test partition, saved test predictions and the test reports, and records which project data
files were opened. Version-1 files are not written.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from . import config, cues, experiments, final, protocol, split
from .features import build_pipeline

V2_DIR = config.REPORTS_DIR / "v2"
LOG_FILE = V2_DIR / "experiments.csv"
RESULTS_JSON = V2_DIR / "results.json"
RESULTS_MD = V2_DIR / "results.md"
PROTOCOL_FILE = V2_DIR / "protocol_v2.json"
BLIND_FILE = V2_DIR / "blind_reading.csv"
MODELS_DIR = config.ROOT / "models" / "v2"
CARD_FILE = config.ROOT / "models" / "model_card_v2.json"
CUES_FILE = Path(cues.__file__)
CUES_SHA256 = "ac5460bb887a6807941a15c95c55fead43c829fa5d71c9b119c92192eff422e0"  # frozen in docs/V2_PLAN.md

LABELS = list(config.SENTIMENTS)
ASPECT_LABELS = list(config.ASPECTS)
NEUTRAL = "محايد"
SELECTION = split.CV_COLUMNS  # seeds 42-44, the version-1 selection folds
OUTER_SEEDS = (300, 301, 302)  # never used before version 2
OUTER = tuple(f"cv_fold_s{s}" for s in OUTER_SEEDS)
INNER_SEED = 0
MIN_GAIN, MIN_NEUTRAL_GAIN, MIN_OTHER_CHANGE, TIE = 0.01, 0.02, -0.02, 0.001
MIN_WINS = {15: 11, 5: 4}
SINGLES = ("H1_phrases", "H2_cues")
BOOTSTRAP = dict(resamples=2000, seed=0, level=0.95)
GRID = tuple(float(t) for t in np.round(np.arange(0.30, 0.951, 0.05), 2))
TARGETS = (("high", 0.90), ("medium", 0.70))
MIN_GROUPS = 30
EDGE_WIDTH, EDGE_SLACK = 0.10, 0.05  # a level must also hold at its lower edge (plan section 9)
DROP_SLACK = 0.05
EPS = protocol.EDGE_EPS
CALIBRATION_BINS = 10
LOG_COLUMNS = ["exp_id", "timestamp", "git_sha", "task", "candidate", "scheme", "features", "params",
               "macro_f1_mean", "macro_f1_std", "fold_scores", "class_f1_folds", "accuracy_mean",
               "balanced_acc_mean", "fit_seconds", "notes"]
# Hand-written challenge sentences (not from the dataset, no ground truth): shown, never scored.
PROBES = ("الفرع في شارع التحلية", "يقدمون القهوة في أكواب ورقية", "زرت المقهى يوم الجمعة مع أخي",
          "القهوة لذيذة لكن الأسعار مرتفعة", "ممتاز بس مزدحم", "الخدمة لا بأس بها", "القهوة عادية",
          "ما عجبني الكيك", "الخدمة ممتازة", "الأسعار غالية جدا")


# ---- guard ----------------------------------------------------------------------------------

DENIED_NAMES = re.compile(r"^(test\.csv|final_test\..+|test_predictions.*|test_access_log\.json)$", re.IGNORECASE)
_GUARD = {"installed": False, "active": False, "denied_ids": set(), "opened": set()}


def _identity(path: str) -> tuple[int, int] | None:
    try:
        st = os.stat(path)
    except (OSError, ValueError):
        return None
    return (st.st_dev, st.st_ino) if st.st_ino else None


def _audit(event: str, args: tuple) -> None:
    if event != "open" or not _GUARD["active"] or isinstance(args[0], int):
        return
    try:
        path = os.path.realpath(os.fsdecode(args[0]))
    except (TypeError, ValueError):
        return
    if DENIED_NAMES.match(os.path.basename(path)) or (_GUARD["denied_ids"] and _identity(path) in _GUARD["denied_ids"]):
        raise PermissionError(f"{os.path.basename(path)} holds held-out test data; version 2 uses development data only")
    try:
        rel = Path(path).relative_to(config.ROOT.resolve())
    except ValueError:
        return
    if rel.parts and rel.parts[0] in ("data", "reports", "models"):
        _GUARD["opened"].add(rel.as_posix())


@contextlib.contextmanager
def guard():
    """Refuse, inside the block, every opening of the held-out test file (by name or file identity),
    of saved test predictions and of the test reports; yield the set of project data files opened.
    A guard against accidental reads in this process, not a sandbox."""
    if not _GUARD["installed"]:
        sys.addaudithook(_audit)
        _GUARD["installed"] = True
    identity = _identity(os.fspath(config.TEST_FILE))
    if identity:
        _GUARD["denied_ids"].add(identity)
    _GUARD["opened"].clear()
    _GUARD["active"] = True
    try:
        yield _GUARD["opened"]
    finally:
        _GUARD["active"] = False


# ---- candidates, folds and out-of-fold probabilities ---------------------------------------

def frozen_specs() -> dict[str, dict]:
    specs = protocol.final_specs()
    if (specs["sentiment"]["name"], specs["aspect"]["name"]) != ("m4_logreg_C0.3", "m4_logreg_C1"):
        raise RuntimeError("the logged decisions no longer give the frozen version-1 configurations")
    return specs


def candidates(v1: dict) -> dict[str, dict]:
    """``V1`` plus the three pre-declared feature changes (plan section 3), same model."""
    return {"V1": dict(v1, name="V1"),
            "H1_phrases": dict(v1, name="H1_phrases", representation="word12+char25"),
            "H2_cues": dict(v1, name="H2_cues", cues=True),
            "H3_both": dict(v1, name="H3_both", representation="word12+char25", cues=True)}


def with_outer_folds(dev: pd.DataFrame) -> pd.DataFrame:
    """Grouped, stratified 5-fold assignments for the outer seeds, built like the selection folds."""
    joint = (dev.aspect + "|" + dev.sentiment).reset_index(drop=True)
    groups = dev.group_id.reset_index(drop=True)
    out = dev.reset_index(drop=True).copy()
    for col, seed in zip(OUTER, OUTER_SEEDS):
        out[col] = split._grouped_folds(joint, groups, seed)
    return out


def fit_proba(spec: dict, X_train, y_train, X_valid, labels: list[str]) -> np.ndarray:
    model = clone(build_pipeline(spec)).fit(X_train, y_train)
    return model.predict_proba(X_valid)[:, [list(model.classes_).index(label) for label in labels]]


def oof_proba(spec: dict, X: np.ndarray, y: np.ndarray, folds: np.ndarray, labels: list[str]) -> tuple[np.ndarray, float]:
    """Probabilities in ``labels`` order for every row from a model fitted on the other folds."""
    P = np.zeros((len(y), len(labels)))
    start = time.perf_counter()
    for k in sorted(set(folds)):
        valid = folds == k
        P[valid] = fit_proba(spec, X[~valid], y[~valid], X[valid], labels)
    return P, time.perf_counter() - start


def predicted(P: np.ndarray, labels: list[str]) -> np.ndarray:
    return np.asarray(labels, dtype=object)[P.argmax(axis=1)]


def fold_metrics(y: np.ndarray, pred: np.ndarray, folds: np.ndarray, labels: list[str]) -> dict[str, list[float]]:
    """Macro-F1, F1 per class, accuracy and balanced accuracy of every fold (sorted fold order)."""
    out = {"macro": [], "accuracy": [], "balanced": [], **{label: [] for label in labels}}
    k = len(labels)
    yi, pi = final.encode(y, labels), final.encode(pred, labels)
    for fold in sorted(set(folds)):
        m = folds == fold
        f1 = final.per_class_f1(yi[m], pi[m], k)
        recall = [float((pi[m][yi[m] == j] == j).mean()) for j in range(k) if (yi[m] == j).any()]
        out["macro"].append(float(f1.mean()))
        out["accuracy"].append(float((yi[m] == pi[m]).mean()))
        out["balanced"].append(float(np.mean(recall)))
        for j, label in enumerate(labels):
            out[label].append(float(f1[j]))
    return out


def merge(parts: list[dict[str, list[float]]]) -> dict[str, list[float]]:
    """Concatenate fold metrics of several repeats, repeat-major."""
    return {key: [v for p in parts for v in p[key]] for key in parts[0]}


class Scheme:
    """Out-of-fold probabilities of specs on a set of fold columns, computed once per run."""

    def __init__(self, dev: pd.DataFrame, columns: tuple[str, ...], target: str, labels: list[str]):
        self.dev, self.columns, self.labels = dev, columns, labels
        self.X, self.y = dev["text"].to_numpy(), dev[target].to_numpy()
        self._cache: dict[str, tuple[list[np.ndarray], float]] = {}

    def proba(self, spec: dict) -> list[np.ndarray]:
        if spec["name"] not in self._cache:
            runs = [oof_proba(spec, self.X, self.y, self.dev[c].to_numpy(), self.labels) for c in self.columns]
            self._cache[spec["name"]] = ([P for P, _ in runs], sum(s for _, s in runs))
        return self._cache[spec["name"]][0]

    def seconds(self, spec: dict) -> float:
        self.proba(spec)
        return self._cache[spec["name"]][1]

    def preds(self, spec: dict) -> list[np.ndarray]:
        return [predicted(P, self.labels) for P in self.proba(spec)]

    def metrics(self, spec: dict) -> dict[str, list[float]]:
        return merge([fold_metrics(self.y, p, self.dev[c].to_numpy(), self.labels)
                      for p, c in zip(self.preds(spec), self.columns)])


# ---- selection rule (plan section 4) --------------------------------------------------------

def paired(base: dict, cand: dict) -> dict:
    """Paired fold differences candidate - base: Macro-F1 gain, wins and mean change of every class F1."""
    diff = np.asarray(cand["macro"]) - np.asarray(base["macro"])
    return dict(gain=float(diff.mean()), wins=int((diff > 0).sum()), n=len(diff),
                class_change={label: float(np.mean(np.asarray(cand[label]) - np.asarray(base[label]))) for label in LABELS})


def eligible(base: dict, cand: dict) -> dict:
    """Eligibility against V1: Macro-F1 gain, fold wins, neutral gain and the other classes' guardrail."""
    d = paired(base, cand)
    others = [v for label, v in d["class_change"].items() if label != NEUTRAL]
    d["eligible"] = bool(d["gain"] >= MIN_GAIN and d["wins"] >= MIN_WINS[d["n"]]
                         and d["class_change"][NEUTRAL] >= MIN_NEUTRAL_GAIN and min(others) >= MIN_OTHER_CHANGE)
    return d


def choose(scores: dict[str, dict]) -> dict:
    """The pre-declared decision order: H3 if eligible and better than the best single by the
    acceptance rule, else the best eligible single, else V1."""
    checks = {name: eligible(scores["V1"], scores[name]) for name in (*SINGLES, "H3_both")}
    g1, g2 = checks["H1_phrases"]["gain"], checks["H2_cues"]["gain"]
    best = "H1_phrases" if g1 - g2 >= TIE else "H2_cues"
    other = next(s for s in SINGLES if s != best)
    h3 = paired(scores[best], scores["H3_both"])
    h3["accepted"] = bool(h3["gain"] >= MIN_GAIN and h3["wins"] >= MIN_WINS[h3["n"]])
    if checks["H3_both"]["eligible"] and h3["accepted"]:
        chosen = "H3_both"
    elif checks[best]["eligible"]:
        chosen = best
    elif checks[other]["eligible"]:
        chosen = other
    else:
        chosen = "V1"
    return dict(checks=checks, best_single=best, h3_vs_best_single=h3, chosen=chosen)


# ---- nested CV (plan section 5) -------------------------------------------------------------

def nested(dev: pd.DataFrame, specs: dict[str, dict], outer: Scheme) -> dict:
    """In every outer fold, choose a candidate with the same rule on a grouped inner 5-fold CV of the
    outer-training rows, and predict the outer fold with it (refitted on all outer-training rows,
    which is the candidate's outer out-of-fold prediction)."""
    X, y = outer.X, outer.y
    chosen, procedure = [], []
    candidate_proba = {name: outer.proba(spec) for name, spec in specs.items()}
    for r, col in enumerate(OUTER):
        folds = dev[col].to_numpy()
        P = np.zeros((len(dev), len(LABELS)))
        for k in sorted(set(folds)):
            train = folds != k
            sub = dev.loc[train].reset_index(drop=True)
            inner = split._grouped_folds(sub.aspect + "|" + sub.sentiment, sub.group_id, INNER_SEED)
            scores = {}
            for name, spec in specs.items():
                Pi, _ = oof_proba(spec, X[train], y[train], inner, LABELS)
                scores[name] = fold_metrics(y[train], predicted(Pi, LABELS), inner, LABELS)
            pick = choose(scores)["chosen"]
            chosen.append(pick)
            P[~train] = candidate_proba[pick][r][~train]
        procedure.append(P)
    return dict(proba=procedure, chosen=chosen)


# ---- paired group bootstrap over repeats ----------------------------------------------------

def _weighted_f1(yi: np.ndarray, pi: np.ndarray, w: np.ndarray, k: int) -> np.ndarray:
    cm = np.bincount(yi * k + pi, weights=w, minlength=k * k).reshape(k, k)
    tp = np.diag(cm)
    denom = cm.sum(axis=1) + cm.sum(axis=0)
    return np.divide(2 * tp, denom, out=np.zeros(k), where=denom > 0)


def paired_bootstrap(y: np.ndarray, groups: np.ndarray, preds_a: list[np.ndarray], preds_b: list[np.ndarray],
                     labels: list[str], resamples: int = BOOTSTRAP["resamples"], seed: int = BOOTSTRAP["seed"],
                     level: float = BOOTSTRAP["level"]) -> dict:
    """Intervals of a - b for Macro-F1 and every class F1. Each resample draws the near-duplicate
    groups once and applies them to every repeat; the metric is computed within each repeat and the
    repeat-level differences are averaged."""
    k = len(labels)
    yi = final.encode(y, labels)
    a = [final.encode(p, labels) for p in preds_a]
    b = [final.encode(p, labels) for p in preds_b]
    _, g = np.unique(groups, return_inverse=True)
    n_groups = g.max() + 1
    rng = np.random.default_rng(seed)

    def diff(w: np.ndarray) -> np.ndarray:
        d = np.mean([_weighted_f1(yi, pa, w, k) - _weighted_f1(yi, pb, w, k) for pa, pb in zip(a, b)], axis=0)
        return np.append(d.mean(), d)  # Macro-F1 difference = mean of the class differences

    point = diff(np.ones(len(yi)))
    draws = np.array([diff(np.bincount(rng.integers(0, n_groups, n_groups), minlength=n_groups)[g].astype(float))
                      for _ in range(resamples)])
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(draws, [tail, 100 - tail], axis=0)
    names = ["macro", *labels]
    return {name: dict(point=float(point[i]), low=float(low[i]), high=float(high[i]),
                       share_le_0=float((draws[:, i] <= 0).mean())) for i, name in enumerate(names)}


# ---- certainty per predicted class (plan section 6, amended in section 9) -------------------

def _groups_in(groups: np.ndarray, mask: np.ndarray) -> int:
    return int(len(np.unique(groups[mask])))


def _level_holds(score, correct, groups, inside, t, target) -> bool:
    if _groups_in(groups, inside) < MIN_GROUPS or correct[inside].mean() < target:
        return False
    edge = inside & (score + EPS < t + EDGE_WIDTH)
    return _groups_in(groups, edge) < MIN_GROUPS or correct[edge].mean() >= target - EDGE_SLACK


def class_thresholds(score: np.ndarray, pred: np.ndarray, correct: np.ndarray, groups: np.ndarray,
                     labels: list[str], floor: dict[str, float] | None = None) -> dict[str, dict[str, float | None]]:
    """Per predicted class: "high" from the smallest grid value t whose tail (score >= t) is right at
    least 90% of the time over >= 30 distinct groups, and whose lower edge [t, t + 0.10) is right at
    least 85% of the time when it holds >= 30 groups; "medium" likewise for [t, high) with 70% / 65%.
    A level that no grid value supports is None (not shown for that class). With ``floor`` (the
    version-1 thresholds) no level may start below its version-1 threshold: levels can only be
    raised or removed per class, never widened (the released rule, plan section 9)."""
    out = {}
    for label in labels:
        own = pred == label
        found: dict[str, float | None] = {}
        upper = np.inf
        for level, target in TARGETS:
            found[level] = None
            for t in GRID:
                if floor is not None and t + EPS < floor[level]:
                    continue
                if t >= upper:
                    break
                inside = own & (score + EPS >= t) & (score + EPS < upper)
                if _level_holds(score, correct, groups, inside, t, target):
                    found[level] = t
                    break
            if found[level] is not None:
                upper = found[level]
        out[label] = found
    return out


def level_of(score: float, thresholds: dict[str, float | None]) -> str:
    for level in ("high", "medium"):
        t = thresholds.get(level)
        if t is not None and score + EPS >= t:
            return level
    return "low"


def level_table(score, pred, correct, groups, thresholds: dict, labels: list[str]) -> list[dict]:
    levels = np.array([level_of(s, thresholds[p]) for s, p in zip(score, pred)])
    rows = []
    for label in labels:
        for level in ("high", "medium", "low"):
            m = (pred == label) & (levels == level)
            rows.append(dict(label=label, level=level, rows=int(m.sum()), groups=_groups_in(groups, m),
                             share=float(m.mean()), accuracy=float(correct[m].mean()) if m.any() else None))
    return rows


def apply_drop_rule(thresholds: dict, stability: list[dict]) -> tuple[dict, list[str]]:
    """Merge a level into the next lower one for a class when its stability accuracy falls more than
    0.05 below its target (pre-declared); returns the thresholds used and what was dropped."""
    out = {label: dict(t) for label, t in thresholds.items()}
    dropped = []
    targets = dict(TARGETS)
    for row in stability:
        t = out[row["label"]]
        if row["level"] in targets and t.get(row["level"]) is not None and row["accuracy"] is not None \
                and row["accuracy"] < targets[row["level"]] - DROP_SLACK:
            if row["level"] == "high":
                t["medium"] = t["high"] if t.get("medium") is None else t["medium"]
            t[row["level"]] = None
            dropped.append(f"{row['label']}:{row['level']}")
    return out, dropped


# ---- calibration ----------------------------------------------------------------------------

def calibration(P: np.ndarray, y: np.ndarray, labels: list[str]) -> dict:
    """Top-label ECE (10 equal-width bins), signed gap (mean top probability - accuracy), multiclass
    Brier score (sum over classes of the squared error, mean over rows) and the reliability table."""
    yi = final.encode(y, labels)
    top, correct = P.max(axis=1), P.argmax(axis=1) == yi
    bins = np.minimum((top * CALIBRATION_BINS).astype(int), CALIBRATION_BINS - 1)
    table = [dict(bin=f"{b / CALIBRATION_BINS:.1f}-{(b + 1) / CALIBRATION_BINS:.1f}", rows=int((bins == b).sum()),
                  confidence=float(top[bins == b].mean()), accuracy=float(correct[bins == b].mean()))
             for b in range(CALIBRATION_BINS) if (bins == b).any()]
    ece = sum(r["rows"] / len(top) * abs(r["confidence"] - r["accuracy"]) for r in table)
    brier = float(((P - np.eye(len(labels))[yi]) ** 2).sum(axis=1).mean())
    return dict(ece=float(ece), gap=float(top.mean() - correct.mean()), brier=brier, mean_top=float(top.mean()),
                accuracy=float(correct.mean()), reliability=table)


# ---- diagnosis ------------------------------------------------------------------------------

def pair_agreement(dev: pd.DataFrame) -> list[dict]:
    """Sentiment agreement of every pair of rows inside the same near-duplicate group, by class."""
    pairs = []
    for _, g in dev.groupby("group_id"):
        s = g.sentiment.tolist()
        pairs += [(s[i], s[j]) for i in range(len(s)) for j in range(i + 1, len(s))]
    frame = pd.DataFrame(pairs, columns=["a", "b"])
    rows = [dict(label="all", pairs=len(frame), agreement=float((frame.a == frame.b).mean()) if len(frame) else None)]
    for label in LABELS:
        m = (frame.a == label) | (frame.b == label)
        rows.append(dict(label=label, pairs=int(m.sum()),
                         agreement=float((frame.a[m] == frame.b[m]).mean()) if m.any() else None))
    return rows


def cue_rates(dev: pd.DataFrame) -> dict[str, dict[str, float]]:
    vectors = pd.DataFrame([cues.cue_vector(t) for t in dev.text], columns=cues.FEATURES)
    vectors["any"] = [float(cues.has_opinion_cue(t)) for t in dev.text]
    binary = (vectors > 0).astype(float)
    binary["sentiment"] = dev.sentiment.to_numpy()
    return {label: {k: float(v) for k, v in g.drop(columns="sentiment").mean().items()}
            for label, g in binary.groupby("sentiment")}


def blind_reading(dev: pd.DataFrame) -> dict | None:
    """Cross-table of the committed blind reading (category per rid) against the gold sentiment."""
    if not BLIND_FILE.exists():
        return None
    reading = pd.read_csv(BLIND_FILE).merge(dev[["rid", "sentiment"]], on="rid", validate="one_to_one")
    table = pd.crosstab(reading.sentiment, reading.reading)
    return dict(n=len(reading), table={label: {k: int(v) for k, v in row.items()} for label, row in table.iterrows()})


# ---- log ------------------------------------------------------------------------------------

def read_log() -> pd.DataFrame:
    if LOG_FILE.exists():
        return pd.read_csv(LOG_FILE, keep_default_na=False)
    return pd.DataFrame(columns=LOG_COLUMNS)


def log_row(task: str, candidate: str, scheme: str, spec: dict | None, metrics: dict, seconds: float,
            labels: list[str], notes: str = "") -> dict:
    """Append a row, or check that a logged row with the same id has the same fold scores."""
    eid = f"{task}/{candidate}@{scheme}"
    folds = [round(float(v), 6) for v in metrics["macro"]]
    log = read_log()
    logged = log.loc[log.exp_id == eid]
    if not logged.empty:
        before = json.loads(logged.iloc[0]["fold_scores"])
        if len(before) != len(folds) or max(abs(a - b) for a, b in zip(before, folds)) > 1e-6:
            raise RuntimeError(f"{eid}: recomputed fold scores differ from the logged row")
        return logged.iloc[0].to_dict()
    identity = experiments.row_identity(spec) if spec else dict(features="", params="{}")
    row = dict(exp_id=eid, timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               git_sha=experiments._git_sha(), task=task, candidate=candidate, scheme=scheme,
               features=identity["features"] + ("+cues" if spec and spec.get("cues") else ""), params=identity["params"],
               macro_f1_mean=round(float(np.mean(metrics["macro"])), 4),
               macro_f1_std=round(float(np.std(metrics["macro"], ddof=1)), 4), fold_scores=json.dumps(folds),
               class_f1_folds=json.dumps({label: [round(v, 6) for v in metrics[label]] for label in labels},
                                         ensure_ascii=False),
               accuracy_mean=round(float(np.mean(metrics["accuracy"])), 4),
               balanced_acc_mean=round(float(np.mean(metrics["balanced"])), 4),
               fit_seconds=round(seconds, 2), notes=notes)
    V2_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row], columns=LOG_COLUMNS).to_csv(LOG_FILE, mode="a", header=not LOG_FILE.exists(), index=False,
                                                    lineterminator="\n")
    return row


# ---- run ------------------------------------------------------------------------------------

def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_frozen_cues() -> str:
    sha = hashlib.sha256(CUES_FILE.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    if sha != CUES_SHA256:
        raise RuntimeError(f"src/arabic_reviews/cues.py changed after it was frozen ({sha[:12]}...)")
    return sha


def _summary(metrics: dict, labels: list[str]) -> dict:
    return dict(macro=dict(mean=float(np.mean(metrics["macro"])), sd=float(np.std(metrics["macro"], ddof=1))),
                accuracy=float(np.mean(metrics["accuracy"])), balanced=float(np.mean(metrics["balanced"])),
                per_class={label: float(np.mean(metrics[label])) for label in labels})


def _pooled(scheme: Scheme, spec: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Score, prediction, correctness and group of every (repeat, row), repeats stacked."""
    P = np.vstack(scheme.proba(spec))
    pred = predicted(P, scheme.labels)
    y = np.tile(scheme.y, len(scheme.columns))
    return P.max(axis=1), pred, pred == y, np.tile(scheme.dev.group_id.to_numpy(), len(scheme.columns))


def _confusion(y: np.ndarray, preds: list[np.ndarray], labels: list[str]) -> list[list[int]]:
    yi = np.tile(final.encode(y, labels), len(preds))
    pi = np.concatenate([final.encode(p, labels) for p in preds])
    return np.bincount(yi * len(labels) + pi, minlength=len(labels) ** 2).reshape(len(labels), -1).tolist()


def _joint(sent: list[np.ndarray], asp: list[np.ndarray], dev: pd.DataFrame, columns) -> list[float]:
    out = []
    for s, a, col in zip(sent, asp, columns):
        ok = (s == dev.sentiment.to_numpy()) & (a == dev.aspect.to_numpy())
        folds = dev[col].to_numpy()
        out += [float(ok[folds == k].mean()) for k in sorted(set(folds))]
    return out


def run_experiments(dev: pd.DataFrame | None = None, check_v1: bool = True) -> dict:
    """Every step of plan sections 2-6 on the development partition; writes ``reports/v2/``.

    ``dev`` and ``check_v1=False`` are for tests on toy data: the real run loads the development
    partition and requires V1 and the aspect model to reproduce the logged version-1 fold scores."""
    with guard() as opened:
        cue_sha = check_frozen_cues()
        real = dev is None
        dev = with_outer_folds(split.load_dev() if real else dev)
        frozen = frozen_specs()
        specs = candidates(frozen["sentiment"])
        aspect_spec = dict(frozen["aspect"], name="V1_aspect")
        sel = Scheme(dev, SELECTION, "sentiment", LABELS)
        out = Scheme(dev, OUTER, "sentiment", LABELS)
        sel_aspect = Scheme(dev, SELECTION, "aspect", ASPECT_LABELS)
        out_aspect = Scheme(dev, OUTER, "aspect", ASPECT_LABELS)

        # selection (section 4); V1 must reproduce the version-1 log on the same folds
        v1_log = experiments.fold_scores(experiments.read_log(), "sentiment/m4_logreg_C0.3")
        scores = {name: sel.metrics(spec) for name, spec in specs.items()}
        if check_v1 and max(abs(a - b) for a, b in zip(scores["V1"]["macro"], v1_log)) > 1e-6:
            raise RuntimeError("V1 does not reproduce the logged version-1 fold scores")
        for name, spec in specs.items():
            log_row("sentiment", name, "selection", spec, scores[name], sel.seconds(spec), LABELS)
        selection = choose(scores)
        chosen = selection["chosen"]

        # nested estimate and stability (section 5)
        outer_scores = {name: out.metrics(spec) for name, spec in specs.items()}
        for name, spec in specs.items():
            log_row("sentiment", name, "outer", spec, outer_scores[name], out.seconds(spec), LABELS)
        nest = nested(dev, specs, out)
        proc_preds = [predicted(P, LABELS) for P in nest["proba"]]
        proc_metrics = merge([fold_metrics(out.y, p, dev[c].to_numpy(), LABELS) for p, c in zip(proc_preds, OUTER)])
        log_row("sentiment", "procedure", "nested", None, proc_metrics, 0.0, LABELS,
                notes="chosen per outer fold: " + json.dumps(nest["chosen"]))
        groups = dev.group_id.to_numpy()
        nested_vs_v1 = dict(paired=paired(outer_scores["V1"], proc_metrics),
                            bootstrap=paired_bootstrap(out.y, groups, proc_preds, out.preds(specs["V1"]), LABELS),
                            chosen_counts={n: nest["chosen"].count(n) for n in specs})
        stability = None
        if chosen != "V1":
            stability = dict(paired=paired(outer_scores["V1"], outer_scores[chosen]),
                             bootstrap=paired_bootstrap(out.y, groups, out.preds(specs[chosen]),
                                                        out.preds(specs["V1"]), LABELS))
        n = nested_vs_v1
        adopted = bool(chosen != "V1" and n["paired"]["gain"] >= MIN_GAIN and n["bootstrap"]["macro"]["low"] > 0
                       and n["paired"]["class_change"][NEUTRAL] > 0
                       and min(v for k, v in n["paired"]["class_change"].items() if k != NEUTRAL) >= MIN_OTHER_CHANGE)
        final_name = chosen if adopted else "V1"
        final_spec = specs[final_name]

        # aspect: the version-1 model, for joint accuracy and certainty
        aspect_sel, aspect_out = sel_aspect.metrics(aspect_spec), out_aspect.metrics(aspect_spec)
        log_row("aspect", "V1_aspect", "selection", aspect_spec, aspect_sel, sel_aspect.seconds(aspect_spec), ASPECT_LABELS)
        log_row("aspect", "V1_aspect", "outer", aspect_spec, aspect_out, out_aspect.seconds(aspect_spec), ASPECT_LABELS)
        aspect_v1_log = experiments.fold_scores(experiments.read_log(), "aspect/m4_logreg_C1")
        if check_v1 and max(abs(a - b) for a, b in zip(aspect_sel["macro"], aspect_v1_log)) > 1e-6:
            raise RuntimeError("the aspect model does not reproduce the logged version-1 fold scores")
        joint = {name: dict(selection=float(np.mean(_joint(sel.preds(specs[name]), sel_aspect.preds(aspect_spec), dev, SELECTION))),
                            outer=float(np.mean(_joint(out.preds(specs[name]), out_aspect.preds(aspect_spec), dev, OUTER))))
                 for name in dict.fromkeys(["V1", final_name])}

        # certainty per predicted class (section 6), for the released sentiment model and the aspect model
        certainty = {}
        for task, s_sel, s_out, spec, labels in (("sentiment", sel, out, final_spec, LABELS),
                                                 ("aspect", sel_aspect, out_aspect, aspect_spec, ASPECT_LABELS)):
            score, pred, correct, grp = _pooled(s_sel, spec)
            o_score, o_pred, o_correct, o_grp = _pooled(s_out, spec)
            v1_thresholds = protocol_thresholds(task)
            v1_levels = {label: dict(v1_thresholds) for label in labels}
            rules = {}
            for rule, floor in (("registered", None), ("released", v1_thresholds)):
                proposed = class_thresholds(score, pred, correct, grp, labels, floor)
                used, dropped = apply_drop_rule(proposed, level_table(o_score, o_pred, o_correct, o_grp, proposed, labels))
                rules[rule] = dict(proposed=proposed, thresholds=used, dropped=dropped,
                                   selection_levels=level_table(score, pred, correct, grp, used, labels),
                                   outer_levels=level_table(o_score, o_pred, o_correct, o_grp, used, labels))
            certainty[task] = dict(
                rules, thresholds=rules["released"]["thresholds"], outer_levels=rules["released"]["outer_levels"],
                v1_thresholds=v1_thresholds,
                v1_rule_selection_levels=level_table(score, pred, correct, grp, v1_levels, labels),
                v1_rule_outer_levels=level_table(o_score, o_pred, o_correct, o_grp, v1_levels, labels))

        # calibration (section 6) and diagnosis (section 2)
        y_rep = np.tile(sel.y, len(SELECTION))
        calib = {name: calibration(np.vstack(sel.proba(specs[name])), y_rep, LABELS) for name in dict.fromkeys(["V1", final_name])}
        v1_rule = {label: protocol_thresholds("sentiment") for label in LABELS}
        probes = probe_predictions(dev, {
            "V1, version-1 rule": (specs["V1"], v1_rule),
            f"{final_name}, registered per-class rule": (final_spec, certainty["sentiment"]["registered"]["thresholds"]),
            f"{final_name}, released rule": (final_spec, certainty["sentiment"]["thresholds"])})
        result = dict(
            plan="docs/V2_PLAN.md", cues_sha256=cue_sha, git_sha=experiments._git_sha(),
            dev=dict(rows=len(dev), sha256=file_sha256(config.DEV_FILE) if real else "toy",
                     split_sha256=file_sha256(config.SPLIT_FILE) if real else "toy",
                     class_counts={k: int(v) for k, v in dev.sentiment.value_counts().items()}),
            versions=final.runtime_versions(),
            selection=dict(scores={name: _summary(m, LABELS) for name, m in scores.items()}, **selection),
            nested=nested_vs_v1, stability=stability, adopted=adopted, released=final_name,
            outer_scores={name: _summary(m, LABELS) for name, m in outer_scores.items()},
            procedure_outer=_summary(proc_metrics, LABELS),
            aspect=dict(selection=_summary(aspect_sel, ASPECT_LABELS), outer=_summary(aspect_out, ASPECT_LABELS)),
            joint_accuracy=joint, certainty=certainty, calibration=calib, probes=probes,
            diagnosis=dict(confusion_v1=_confusion(sel.y, sel.preds(specs["V1"]), LABELS),
                           confusion_released=_confusion(sel.y, sel.preds(final_spec), LABELS),
                           pair_agreement=pair_agreement(dev), cue_rates=cue_rates(dev),
                           blind_reading=blind_reading(dev) if real else None),
            inputs=sorted(opened),
        )
    V2_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(json.dumps(_round(result), ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    RESULTS_MD.write_text(results_markdown(json.loads(RESULTS_JSON.read_text(encoding="utf-8"))), encoding="utf-8",
                          newline="\n")
    return result


def protocol_thresholds(task: str) -> dict[str, float]:
    p = json.loads(protocol.PROTOCOL_JSON.read_text(encoding="utf-8"))
    return p["tasks"][task]["certainty"]["thresholds"]


def probe_predictions(dev: pd.DataFrame, models: dict[str, tuple[dict, dict]]) -> list[dict]:
    """Hand-written challenge sentences scored by models fitted on the whole development partition;
    ``models`` maps a column title to (spec, thresholds per predicted class)."""
    X, y = dev["text"].to_numpy(), dev["sentiment"].to_numpy()
    fitted = {title: (clone(build_pipeline(spec)).fit(X, y), t) for title, (spec, t) in models.items()}
    rows = []
    for text in PROBES:
        row = dict(text=text, cues=cues.has_opinion_cue(text), models={})
        for title, (model, thresholds) in fitted.items():
            P = model.predict_proba([text])[0][[list(model.classes_).index(label) for label in LABELS]]
            label = LABELS[int(P.argmax())]
            row["models"][title] = dict(label=label, level=level_of(float(P.max()), thresholds[label]))
        rows.append(row)
    return rows


def _round(obj, digits: int = 6):
    if isinstance(obj, float):
        return round(obj, digits)
    if isinstance(obj, dict):
        return {k: _round(v, digits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round(v, digits) for v in obj]
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return _round(obj.item(), digits)
    return obj


# ---- report ---------------------------------------------------------------------------------

def _f(v, digits: int = 3) -> str:
    return "–" if v is None else f"{v:.{digits}f}"


def _signed(v) -> str:
    return f"{v:+.4f}"


def _ci(b: dict) -> str:
    return f"{_signed(b['point'])} [{b['low']:+.3f}, {b['high']:+.3f}]"


def results_markdown(r: dict) -> str:
    L = LABELS
    sel, nest = r["selection"], r["nested"]
    lines = ["# Version 2 - sentiment on development data only", "",
             "Plan and decision rules: `docs/V2_PLAN.md`. Every number below comes from the development partition "
             f"({r['dev']['rows']} reviews, `dev.csv` SHA-256 `{r['dev']['sha256'][:12]}...`); the old held-out test was "
             "not opened by this run (files opened under data/, reports/, models/: "
             + ", ".join(f"`{p}`" for p in r["inputs"]) + "). "
             f"Polarity-cue file SHA-256 `{r['cues_sha256'][:12]}...` (frozen before any candidate was scored). "
             f"Commit `{r['git_sha']}`.", "",
             "**What these numbers can and cannot show.** The design of version 2 (its target, candidates and cue list) "
             "came after reading development reviews, and the same reviews are used here. The nested CV estimates the "
             "whole selection procedure without fold-selection optimism, and the outer folds use new seeds, but no new "
             "reviews: the estimates are development-only and optimistic to an unknown degree. There is no independent "
             "test for version 2, and no result is shown on the old test.", "",
             "## 1. Diagnosis of the neutral class", ""]
    d = r["diagnosis"]
    counts = r["dev"]["class_counts"]
    lines += ["Class counts: " + ", ".join(f"{k} {counts[k]}" for k in L) + ".", "",
              "Out-of-fold confusion of V1 on the selection folds (3 repeats pooled, every review three times; rows = "
              "true, columns = predicted):", "", "| true \\ pred | " + " | ".join(L) + " |", "|---" * (len(L) + 1) + "|"]
    for label, row in zip(L, d["confusion_v1"]):
        lines.append(f"| {label} | " + " | ".join(str(v) for v in row) + " |")
    lines += ["", "Share of reviews with each cue type (fixed list, `src/arabic_reviews/cues.py`):", "",
              "| cue | " + " | ".join(L) + " |", "|---" * (len(L) + 1) + "|"]
    for key in [*cues.FEATURES, "any"]:
        lines.append(f"| {key} | " + " | ".join(f"{d['cue_rates'][label][key]:.1%}" for label in L) + " |")
    lines += ["", "Sentiment agreement inside near-duplicate groups (pairs of rows of one group; the texts differ, "
              "for example a full review and a fragment of it):", "", "| pairs involving | pairs | agreement |",
              "|---|---|---|"]
    for row in d["pair_agreement"]:
        lines.append(f"| {row['label']} | {row['pairs']} | {_f(row['agreement'])} |")
    if d.get("blind_reading"):
        b = d["blind_reading"]
        cats = ["positive", "negative", "mixed", "lukewarm", "descriptive", "unclear"]
        lines += ["", f"Blind reading of a shuffled sample of {b['n']} development reviews (60 neutral, 30 positive, 30 "
                  "negative), done without the labels in a context separate from this analysis; one reading, not a second "
                  "human annotation and not a measure of annotator agreement. Categories per rid: "
                  "`reports/v2/blind_reading.csv`.", "", "| gold \\ reading | " + " | ".join(cats) + " |",
                  "|---" * (len(cats) + 1) + "|"]
        for label in L:
            row = b["table"].get(label, {})
            lines.append(f"| {label} | " + " | ".join(str(row.get(c, 0)) for c in cats) + " |")
    lines += ["", "## 2. Selection (seeds 42-44, 15 paired folds)", "",
              "| candidate | Macro-F1 (mean ± sd) | " + " | ".join(f"F1 {label}" for label in L)
              + " | gain vs V1 | folds won | neutral F1 change | eligible |", "|---" * (len(L) + 6) + "|"]
    for name, s in sel["scores"].items():
        c = sel["checks"].get(name)
        tail = (f" | {_signed(c['gain'])} | {c['wins']}/{c['n']} | {_signed(c['class_change'][NEUTRAL])} | "
                f"{'yes' if c['eligible'] else 'no'} |") if c else " | – | – | – | reference |"
        lines.append(f"| {name} | {s['macro']['mean']:.4f} ± {s['macro']['sd']:.4f} | "
                     + " | ".join(f"{s['per_class'][label]:.3f}" for label in L) + tail)
    h3 = sel["h3_vs_best_single"]
    lines += ["", f"Best single candidate by Macro-F1: `{sel['best_single']}`; H3 against it: {_signed(h3['gain'])}, "
              f"{h3['wins']}/{h3['n']} folds ({'passes' if h3['accepted'] else 'does not pass'}). "
              f"**Chosen by the rule: `{sel['chosen']}`.** The rule is a deterministic selection heuristic, not a "
              "significance test: the 15 folds are correlated (each review appears in three repeats).", "",
              "Ablation (2 x 2, same folds): V1 = neither change, H1 = word bigrams, H2 = cue block, H3 = both; the rows "
              "above are the four cells.", "",
              "## 3. Nested CV estimate of the selection procedure (outer seeds 300-302)", "",
              "Inside each of the 15 outer folds the same rule chose a candidate on a grouped inner 5-fold CV of the "
              "outer-training rows (fold wins >= 4 of 5), and the chosen candidate predicted the outer fold. Choices: "
              + ", ".join(f"{k} {v}" for k, v in nest["chosen_counts"].items()) + ".", "",
              "| comparison on the outer folds | Macro-F1 gain [95% interval] | " + " | ".join(f"{label} F1 change" for label in L)
              + " | folds won |", "|---" * (len(L) + 3) + "|"]
    rows = [("procedure vs V1", nest)] + ([(f"{sel['chosen']} vs V1 (stability)", r["stability"])] if r["stability"] else [])
    for title, block in rows:
        b = block["bootstrap"]
        lines.append(f"| {title} | {_ci(b['macro'])} | " + " | ".join(_ci(b[label]) for label in L)
                     + f" | {block['paired']['wins']}/{block['paired']['n']} |")
    lines += ["", "Intervals: paired group bootstrap, 2000 resamples (seed 0); each resample draws the near-duplicate "
              "groups once, applies them to the three repeats, computes the metric within each repeat and averages the "
              "three differences.", "",
              f"**Decision (plan section 5): {'version 2 sentiment model adopted: `' + r['released'] + '`' if r['adopted'] else 'no new sentiment model adopted; version 2 keeps the version-1 sentiment configuration'}.**",
              ""]
    o = r["outer_scores"]
    lines += ["Outer-fold scores of every candidate (no choice made on them):", "",
              "| candidate | Macro-F1 (mean ± sd) | " + " | ".join(f"F1 {label}" for label in L) + " |",
              "|---" * (len(L) + 2) + "|"]
    for name, s in o.items():
        lines.append(f"| {name} | {s['macro']['mean']:.4f} ± {s['macro']['sd']:.4f} | "
                     + " | ".join(f"{s['per_class'][label]:.3f}" for label in L) + " |")
    p = r["procedure_outer"]
    lines.append(f"| procedure (nested) | {p['macro']['mean']:.4f} ± {p['macro']['sd']:.4f} | "
                 + " | ".join(f"{p['per_class'][label]:.3f}" for label in L) + " |")
    lines += ["", "Joint accuracy (sentiment and aspect both right; aspect = the version-1 model): "
              + "; ".join(f"{name}: selection {v['selection']:.4f}, outer {v['outer']:.4f}" for name, v in r["joint_accuracy"].items())
              + ".", "", "## 4. Certainty levels per predicted class", "",
              "Score = the top class probability of the released model. Per predicted class, from the pooled "
              "out-of-fold predictions of the selection folds (support counted in distinct near-duplicate groups): high "
              "starts at the smallest value t on the grid 0.30, 0.35, ..., 0.95 whose tail (score >= t) is right >= 90% "
              "of the time over >= 30 groups and whose lower edge [t, t + 0.10) is right >= 85% of the time when it holds "
              ">= 30 groups; medium likewise for [t, high) with 70% / 65%. A dash means the level does not exist for that "
              "class. A level whose accuracy on the outer folds falls more than 0.05 below its target is merged into the "
              "next lower one.", "",
              "**Released rule (plan section 9, decided after the registered rule's results were seen):** the same search, "
              "but no level may start below its version-1 threshold, so a level can only be raised or removed for a class, "
              "never widened. The registered rule widened high for the largest classes (for example positive from 0.60 to "
              "0.45), which in turn showed high certainty on hand-written descriptive sentences (section 6); the aim was "
              "to stop over-claiming, so the release keeps the stricter variant. Both are reported.", ""]
    for task in ("sentiment", "aspect"):
        c = r["certainty"][task]
        labels = L if task == "sentiment" else ASPECT_LABELS
        reg = c["registered"]
        lines += [f"### {task}", "", f"Version-1 thresholds (one pair for every class): high {c['v1_thresholds']['high']:.2f}, "
                  f"medium {c['v1_thresholds']['medium']:.2f}.", "",
                  "| predicted | released: high from | released: medium from | registered: high from | registered: medium from |",
                  "|---|---|---|---|---|"]
        for label in labels:
            t, g = c["thresholds"][label], reg["thresholds"][label]
            lines.append(f"| {label} | {_f(t['high'], 2)} | {_f(t['medium'], 2)} | {_f(g['high'], 2)} | {_f(g['medium'], 2)} |")
        lines += ["", f"Dropped by the outer-fold check: released {', '.join(c['released']['dropped']) or 'none'}; "
                  f"registered {', '.join(reg['dropped']) or 'none'}.", "",
                  "| predicted | level | released: selection rows (groups) | accuracy | released: outer rows (groups) | accuracy "
                  "| v1 rule: outer rows | accuracy | registered: outer rows | accuracy |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for a, b, v, g in zip(c["released"]["selection_levels"], c["released"]["outer_levels"], c["v1_rule_outer_levels"],
                              reg["outer_levels"]):
            lines.append(f"| {a['label']} | {a['level']} | {a['rows']} ({a['groups']}) | {_f(a['accuracy'])} | "
                         f"{b['rows']} ({b['groups']}) | {_f(b['accuracy'])} | {v['rows']} | {_f(v['accuracy'])} | "
                         f"{g['rows']} | {_f(g['accuracy'])} |")
        lines.append("")
    lines += ["## 5. Calibration (selection folds, pooled)", "",
              "The interface never shows probabilities; this section explains why the levels are read from measured "
              "accuracy and not from the probability itself.", "",
              "| model | top-label ECE | signed gap (mean top probability - accuracy) | multiclass Brier | mean top probability | accuracy |",
              "|---|---|---|---|---|---|"]
    for name, c in r["calibration"].items():
        lines.append(f"| {name} | {c['ece']:.3f} | {c['gap']:+.3f} | {c['brier']:.3f} | {c['mean_top']:.3f} | {c['accuracy']:.3f} |")
    lines += ["", "## 6. Hand-written challenge sentences (qualitative only)", "",
              "Written for this check, not taken from the data and without ground truth; models fitted on the whole "
              "development partition; the level by each rule.", "",
              "| sentence | listed cue | " + " | ".join(r["probes"][0]["models"]) + " |",
              "|---|---|" + "---|" * len(r["probes"][0]["models"])]
    for row in r["probes"]:
        lines.append(f"| {row['text']} | {'yes' if row['cues'] else 'no'} | "
                     + " | ".join(f"{m['label']} ({m['level']})" for m in row["models"].values()) + " |")
    lines.append("")
    return "\n".join(lines)


# ---- build (freeze version 2) ---------------------------------------------------------------

def protocol_v2(r: dict, specs: dict[str, dict]) -> dict:
    pipelines = {task: protocol.pipeline_params(spec) for task, spec in specs.items()}
    body = dict(
        version="v2", plan="docs/V2_PLAN.md", released=r["released"], adopted=r["adopted"],
        configurations={task: experiments._spec_summary(spec) | {"params": spec.get("params", {}),
                                                                 "cues": bool(spec.get("cues"))}
                        for task, spec in specs.items()},
        pipelines=pipelines,
        certainty={task: dict(score="top class probability", thresholds_by_class=r["certainty"][task]["thresholds"])
                   for task in specs},
        cues_sha256=r["cues_sha256"], dev=r["dev"],
        rules=dict(min_gain=MIN_GAIN, min_neutral_gain=MIN_NEUTRAL_GAIN, min_other_change=MIN_OTHER_CHANGE,
                   min_wins=MIN_WINS, outer_seeds=OUTER_SEEDS, grid=GRID, targets=dict(TARGETS),
                   min_groups=MIN_GROUPS, edge=[EDGE_WIDTH, EDGE_SLACK], drop_slack=DROP_SLACK),
    )
    body = _round(json.loads(json.dumps(body, ensure_ascii=False, default=list)))
    fingerprint = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return dict(body, fingerprint=fingerprint)


def build() -> dict:
    """Freeze the version-2 protocol from ``results.json`` and fit the version-2 pipelines on the whole
    development partition (local files), then write the version-2 model card."""
    with guard() as opened:
        cue_sha = check_frozen_cues()
        r = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
        dev = split.load_dev()
        if r["cues_sha256"] != cue_sha or r["dev"]["sha256"] != file_sha256(config.DEV_FILE) \
                or r["dev"]["split_sha256"] != file_sha256(config.SPLIT_FILE):
            raise RuntimeError("results.json was computed on other data or another cue list; re-run v2-experiments")
        frozen = frozen_specs()
        specs = {"sentiment": candidates(frozen["sentiment"])[r["released"]], "aspect": frozen["aspect"]}
        p = protocol_v2(r, specs)
        PROTOCOL_FILE.write_text(json.dumps(p, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
        X = dev["text"].to_numpy()
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        digests = {}
        for task, spec in specs.items():
            model = build_pipeline(spec).fit(X, dev[experiments.TASKS[task]].to_numpy())
            joblib.dump(model, MODELS_DIR / f"{task}.joblib")
            digests[task] = final.dev_digest(model, X)
        recorded = json.loads(final.DIGESTS_FILE.read_text(encoding="utf-8"))
        same_versions = recorded["versions"] == final.runtime_versions()
        for task in specs:
            same_model = task == "aspect" or r["released"] == "V1"
            if same_model and same_versions and digests[task] != recorded["digests"][task]:
                raise RuntimeError(f"the version-2 {task} model should equal the version-1 one but its digest differs")
        card = model_card_v2(r, p, specs, digests, dev)
        CARD_FILE.write_text(json.dumps(card, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
        return dict(fingerprint=p["fingerprint"], released=r["released"], digests=digests, inputs=sorted(opened))


def model_card_v2(r: dict, p: dict, specs: dict, digests: dict, dev: pd.DataFrame) -> dict:
    lengths = dev.text.str.len()
    sel = r["selection"]["scores"]
    return _round(dict(
        name="Arabic cafe review sentiment and main-aspect classifiers, version 2", version="v2",
        fingerprint=p["fingerprint"], released_sentiment=r["released"], adopted_new_sentiment_model=r["adopted"],
        evaluation="development data only (grouped CV); no independent test exists for version 2, and the old "
                   "held-out test was not used",
        trained_on=dict(partition="development", rows=len(dev), sha256=r["dev"]["sha256"],
                        split_sha256=r["dev"]["split_sha256"]),
        source=dict(record=config.SOURCE["record"], doi=config.SOURCE["doi"], license=config.SOURCE["license"]),
        versions=final.runtime_versions(),
        tasks={task: dict(
            file=f"models/v2/{task}.joblib", labels=LABELS if task == "sentiment" else ASPECT_LABELS,
            configuration=p["configurations"][task], same_as_v1=(task == "aspect" or r["released"] == "V1"),
            dev_digest=digests[task],
            certainty=dict(thresholds_by_class=r["certainty"][task]["thresholds"],
                           outer_levels=r["certainty"][task]["outer_levels"]),
            cv=(dict(selection=sel[r["released"]], outer=r["outer_scores"][r["released"]])
                if task == "sentiment" else r["aspect"]),
        ) for task in specs},
        nested_vs_v1=dict(gain=r["nested"]["bootstrap"]["macro"], neutral=r["nested"]["bootstrap"][NEUTRAL],
                          chosen_counts=r["nested"]["chosen_counts"]),
        joint_accuracy=r["joint_accuracy"],
        input=dict(development_length_chars=dict(min=int(lengths.min()), median=float(lengths.median()),
                                                 p99=float(lengths.quantile(0.99)), max=int(lengths.max()))),
    ))
