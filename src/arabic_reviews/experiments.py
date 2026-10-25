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
COLUMNS = [
    "exp_id", "timestamp", "git_sha", "task", "preprocessing", "features", "model", "params",
    "cv_macro_f1_mean", "cv_macro_f1_std", "fold_scores", "balanced_acc_mean",
    "accuracy_mean", "fit_seconds", "notes",
]
EXPERIMENTS_FILE = config.REPORTS_DIR / "experiments.csv"
ABLATIONS_FILE = config.REPORTS_DIR / "ablations.md"
MIN_DELTA = 0.01  # acceptance rule (PROJECT_PLAN.md 1.1): mean paired gain and fold wins
MIN_WINS = 11


def _spec(name: str, model: str, representation: str = "word11", normalisation: str = "none",
          class_weight=None, params=None, notes: str = "") -> dict:
    return dict(name=name, model=model, representation=representation, normalisation=normalisation,
                class_weight=class_weight, params=params or {}, notes=notes)


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
            f1s.append(f1_score(y[valid], pred, average="macro", zero_division=0))
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


def _git_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, cwd=config.ROOT, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def read_log() -> pd.DataFrame:
    if EXPERIMENTS_FILE.exists():
        return pd.read_csv(EXPERIMENTS_FILE, keep_default_na=False)
    return pd.DataFrame(columns=COLUMNS)


def fold_scores(log: pd.DataFrame, eid: str) -> list[float]:
    return json.loads(log.loc[log.exp_id == eid, "fold_scores"].iloc[0])


def run_pending(dev: pd.DataFrame | None = None, registry: list[dict] | None = None) -> pd.DataFrame:
    """Run every registered experiment not yet in the log (append-only; finished
    experiments are never re-run or edited). Returns the rows appended."""
    dev = split.load_dev() if dev is None else dev
    registry = REGISTRY if registry is None else registry
    log = read_log()
    done = set(log.exp_id)
    rows = []
    for task in TASKS:
        for spec in registry:
            eid = exp_id(task, spec["name"])
            if eid in done:
                continue
            res = cv_scores(spec, dev, task)
            rows.append(dict(
                exp_id=eid,
                timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                git_sha=_git_sha(), task=task, preprocessing=spec["normalisation"],
                features=spec["representation"], model=spec["model"],
                params=json.dumps({**spec["params"], "class_weight": spec["class_weight"]}, sort_keys=True),
                cv_macro_f1_mean=round(float(np.mean(res["f1"])), 4),
                cv_macro_f1_std=round(float(np.std(res["f1"], ddof=1)), 4),
                fold_scores=json.dumps([round(float(v), 4) for v in res["f1"]]),
                balanced_acc_mean=round(float(np.mean(res["bal"])), 4),
                accuracy_mean=round(float(np.mean(res["acc"])), 4),
                fit_seconds=round(res["seconds"], 2), notes=spec["notes"],
            ))
    new = pd.DataFrame(rows, columns=COLUMNS)
    if len(new):
        out = pd.concat([log, new], ignore_index=True) if len(log) else new
        EXPERIMENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(EXPERIMENTS_FILE, index=False, lineterminator="\n")
    return new


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
