"""Frozen evaluation protocol (M4): the final configuration per task, certainty levels from
out-of-fold probabilities, the expected test range from CV repeats with fresh seeds, and a
clean-tree rerun of the log rows the protocol cites.

Everything here uses ``split.load_dev()`` only; the held-out test partition is not read.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from . import config, experiments, split

FRESH_SEEDS = (100, 101, 102)  # never used for selection (PROJECT_PLAN.md 1.1)
FRESH_COLUMNS = tuple(f"cv_fold_s{s}" for s in FRESH_SEEDS)
REFERENCE = "logreg_word11"  # the simple reference model of PROJECT_PLAN.md 1.1
BASELINES = ("dummy_most_frequent", "dummy_stratified", REFERENCE)
# Certainty levels: a score band is "high" when every populated band above it is right at least
# 90% of the time out of fold, "medium" at least 70%; anything lower is "low".
LEVEL_TARGETS = (("high", 0.90), ("medium", 0.70))
BIN_WIDTH = 0.05
MIN_BIN = 30  # bands with fewer out-of-fold rows are too noisy to veto a threshold
BOOTSTRAP = dict(resamples=2000, seed=0, unit="group_id", level=0.95)
PROTOCOL_JSON = config.REPORTS_DIR / "protocol.json"
PROTOCOL_MD = config.REPORTS_DIR / "final_protocol.md"


def final_specs() -> dict[str, dict]:
    """The configuration chosen for each task by the logged M3 and M4 decisions."""
    return {task: experiments.resolve_m4(task)["final"] for task in experiments.TASKS}


def reference_spec() -> dict:
    return next(s for s in experiments.REGISTRY if s["name"] == REFERENCE)


def with_fresh_folds(dev: pd.DataFrame) -> pd.DataFrame:
    """Add grouped, stratified 5-fold assignments for the fresh seeds, built exactly like the
    selection folds (joint label strata, near-duplicate groups kept whole)."""
    joint = (dev.aspect + "|" + dev.sentiment).reset_index(drop=True)
    groups = dev.group_id.reset_index(drop=True)
    out = dev.reset_index(drop=True).copy()
    for col, seed in zip(FRESH_COLUMNS, FRESH_SEEDS):
        out[col] = split._grouped_folds(joint, groups, seed)
    return out


# ---- certainty levels -----------------------------------------------------------------------

def band_table(score: np.ndarray, correct: np.ndarray) -> pd.DataFrame:
    """Out-of-fold accuracy per score band [lo, lo + BIN_WIDTH); the last band includes 1."""
    edges = np.round(np.arange(0, 1, BIN_WIDTH), 2)
    idx = np.minimum((np.asarray(score) / BIN_WIDTH + 1e-9).astype(int), len(edges) - 1)
    correct = np.asarray(correct, dtype=float)
    rows = [dict(lo=lo, n=int((idx == i).sum()),
                 accuracy=float(correct[idx == i].mean()) if (idx == i).any() else np.nan)
            for i, lo in enumerate(edges)]
    return pd.DataFrame(rows)


def threshold(table: pd.DataFrame, target: float) -> float:
    """Smallest band edge from which every band with at least MIN_BIN rows reaches ``target``."""
    for lo in table.lo:
        above = table[(table.lo >= lo) & (table.n >= MIN_BIN)]
        if not above.empty and (above.accuracy >= target).all():
            return float(lo)
    return 1.0


def certainty_thresholds(score: np.ndarray, correct: np.ndarray) -> dict[str, float]:
    table = band_table(score, correct)
    return {level: threshold(table, target) for level, target in LEVEL_TARGETS}


def certainty_level(score: float, thresholds: dict[str, float]) -> str:
    if score >= thresholds["high"]:
        return "high"
    return "medium" if score >= thresholds["medium"] else "low"


def level_summary(score: np.ndarray, correct: np.ndarray, thresholds: dict[str, float]) -> list[dict]:
    levels = np.array([certainty_level(s, thresholds) for s in score])
    correct = np.asarray(correct, dtype=bool)
    return [dict(level=lv, n=int((levels == lv).sum()), share=float((levels == lv).mean()),
                 accuracy=float(correct[levels == lv].mean()) if (levels == lv).any() else float("nan"))
            for lv in ("high", "medium", "low")]


# ---- expected range -------------------------------------------------------------------------

def fold_metrics(oofs: list[pd.DataFrame], labels: list[str]) -> list[float]:
    """Macro-F1 of every (repeat, fold) from out-of-fold predictions, repeat-major."""
    return [f1_score(g.y, g.pred, average="macro", labels=labels, zero_division=0)
            for oof in oofs for _, g in oof.groupby("fold", sort=True)]


def joint_fold_accuracy(sent: list[pd.DataFrame], asp: list[pd.DataFrame]) -> list[float]:
    """Share of reviews with both labels right, per (repeat, fold)."""
    out = []
    for s, a in zip(sent, asp):
        both = s.merge(a, on=["rid", "fold"], suffixes=("_s", "_a"))
        both["ok"] = (both.y_s == both.pred_s) & (both.y_a == both.pred_a)
        out += [float(g.ok.mean()) for _, g in both.groupby("fold", sort=True)]
    return out


def expected_range(values: list[float]) -> dict:
    mean, sd = float(np.mean(values)), float(np.std(values, ddof=1))
    return dict(mean=mean, sd=sd, low=max(0.0, mean - 2 * sd), high=min(1.0, mean + 2 * sd), n=len(values))


# ---- clean-tree rerun -----------------------------------------------------------------------

def clean_rerun(dev: pd.DataFrame, finals: dict[str, dict]) -> list[dict]:
    """Re-run the cited configurations on the selection folds from a committed tree, log them as
    ``rerun_<name>`` and compare their fold scores with the originally logged rows."""
    sha = experiments._git_sha()
    if sha == "unknown" or sha.endswith("+dirty"):
        raise RuntimeError(f"clean rerun needs a committed tree (git state: {sha})")
    by_name = {s["name"]: s for s in experiments.REGISTRY}
    out = []
    for task in experiments.TASKS:
        specs = [by_name[n] for n in BASELINES] + [experiments.resolve_axes(task)[1], finals[task]]
        for spec in specs:
            name = f"rerun_{spec['name']}"
            if experiments.exp_id(task, name) not in set(experiments.read_log().exp_id):
                experiments.run_experiment(task, dict(spec, name=name, notes=f"clean-tree rerun of {spec['name']}"), dev)
            log = experiments.read_log()
            row = log.loc[log.exp_id == experiments.exp_id(task, name)].iloc[0]
            orig = log.loc[log.exp_id == experiments.exp_id(task, spec["name"])].iloc[0]
            diff = np.abs(np.array(json.loads(row.fold_scores)) - np.array(json.loads(orig.fold_scores))).max()
            out.append(dict(task=task, name=spec["name"], original_sha=orig.git_sha, rerun_sha=row.git_sha,
                            original_mean=float(orig.cv_macro_f1_mean), rerun_mean=float(row.cv_macro_f1_mean),
                            max_fold_diff=float(diff)))
    return out


# ---- protocol -------------------------------------------------------------------------------

def _config(spec: dict) -> dict:
    return {k: v for k, v in spec.items() if k != "notes"}


def fingerprint(tasks: dict) -> str:
    """SHA-256 of the frozen configurations and thresholds (what M5 must match)."""
    frozen = {t: dict(final=v["final"], reference=v["reference"], certainty=v["certainty"]["thresholds"])
              for t, v in tasks.items()}
    return hashlib.sha256(json.dumps(frozen, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def build_protocol(dev: pd.DataFrame | None = None, rerun: bool = True) -> dict:
    dev = split.load_dev() if dev is None else dev
    finals = final_specs()
    reruns = clean_rerun(dev, finals) if rerun else []
    fresh = with_fresh_folds(dev)
    tasks, fresh_oof = {}, {}
    for task, spec in finals.items():
        labels = experiments.LABELS[task]
        selection = pd.concat([experiments.oof_predictions(spec, dev, task, r) for r in range(len(split.CV_COLUMNS))])
        ok = (selection.y == selection.pred).to_numpy()
        thresholds = certainty_thresholds(selection.score.to_numpy(), ok)
        fresh_oof[task] = [experiments.oof_predictions(spec, fresh, task, column=c) for c in FRESH_COLUMNS]
        ref_oof = [experiments.oof_predictions(reference_spec(), fresh, task, column=c) for c in FRESH_COLUMNS]
        pooled = pd.concat(fresh_oof[task])
        tasks[task] = dict(
            final=_config(spec), reference=_config(reference_spec()),
            certainty=dict(score="highest class probability of the logistic-regression pipeline",
                           thresholds=thresholds,
                           selection_levels=level_summary(selection.score.to_numpy(), ok, thresholds),
                           fresh_levels=level_summary(pooled.score.to_numpy(),
                                                      (pooled.y == pooled.pred).to_numpy(), thresholds)),
            expected_range=dict(final=expected_range(fold_metrics(fresh_oof[task], labels)),
                                reference=expected_range(fold_metrics(ref_oof, labels))),
        )
    return dict(
        version=1,
        dev_sha256=hashlib.sha256(config.DEV_FILE.read_bytes()).hexdigest(),
        split_sha256=hashlib.sha256(config.SPLIT_FILE.read_bytes()).hexdigest(),
        dev_rows=len(dev), fresh_seeds=list(FRESH_SEEDS),
        tasks=tasks,
        joint_expected_range=expected_range(joint_fold_accuracy(fresh_oof["sentiment"], fresh_oof["aspect"])),
        bootstrap=BOOTSTRAP, clean_rerun=reruns, fingerprint=fingerprint(tasks),
    )


def _rounded(obj, digits: int = 6):
    if isinstance(obj, float):
        return round(obj, digits)
    if isinstance(obj, dict):
        return {k: _rounded(v, digits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_rounded(v, digits) for v in obj]
    return obj


def _range_cell(r: dict) -> str:
    return f"{r['mean']:.4f} | {r['sd']:.4f} | [{r['low']:.3f}, {r['high']:.3f}]"


def protocol_markdown(p: dict) -> str:
    log = experiments.read_log()
    lines = [
        "# Final evaluation protocol (frozen before the held-out test is read)", "",
        "This file and `reports/protocol.json` fix every choice that the one-shot test in M5 depends on. "
        "All numbers come from the development partition only; the held-out test partition has not been "
        "read by any step up to here (experiment code can only call `split.load_dev()`).", "",
        f"- Protocol fingerprint (configurations + certainty thresholds): `{p['fingerprint']}`",
        f"- Development partition: {p['dev_rows']} reviews, `data/processed/dev.csv` SHA-256 `{p['dev_sha256']}`",
        f"- Split file `data/splits/split_v1.csv` SHA-256 `{p['split_sha256']}`", "",
        "## 1. Frozen configurations", "",
        "| task | role | configuration | model | features | normalisation | class weight | parameters "
        "| selection CV Macro-F1 (3 x 5) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for task, t in p["tasks"].items():
        for role in ("final", "reference"):
            s = t[role]
            row = log.loc[log.exp_id == experiments.exp_id(task, s["name"])].iloc[0]
            lines.append(f"| {task} | {role} | `{s['name']}` | {s['model']} | {s['representation']} TF-IDF | "
                         f"{s['normalisation']} | {s['class_weight']} | `{json.dumps(s['params'])}` | "
                         f"{row.cv_macro_f1_mean:.4f} ± {row.cv_macro_f1_std:.4f} |")
    lines += [
        "", "TF-IDF uses scikit-learn defaults (no `min_df`, no `sublinear_tf`); `word11` is word unigrams, "
        "`char25` is character 2-5 grams inside word boundaries, `+` joins both in one feature union. Logistic "
        "regression uses `max_iter=2000` and `random_state=0`. The dummy baselines (`most_frequent`, "
        "`stratified`) are reported next to the models in M5.", "",
        "How the configurations were chosen (all on the same 3 x 5 grouped CV; rule: mean paired Macro-F1 gain "
        ">= 0.01 and a gain in >= 11 of 15 folds):", "",
        "- Representation and preprocessing: `reports/ablations.md`. Sentiment uses word unigrams + character "
        "2-5 grams, aspect character 2-5 grams. No normalisation, symbol, digit, negation, stopword or "
        "text-statistic option passed the rule; balanced class weights passed for both tasks.",
        "- Model family and regularisation: `reports/models.md`. Linear SVM and Complement NB did not pass the "
        "rule, so logistic regression stays. The regularisation strength comes from the pre-declared C grid "
        "with the one-standard-error rule (sentiment C = 0.3, aspect C = 3). Both picks differ from the default "
        "C = 1 by less than the fold noise (sentiment -0.003, aspect +0.003 mean Macro-F1). They are kept "
        "because the selection procedure was fixed before the grid was run, not because they are better.",
        "- The optional pre-trained sentence-vector experiment was not run: its pre-declared condition was not "
        "met (`reports/error_analysis.md`).", "",
        "## 2. Certainty levels", "",
        "Score: the highest class probability of the task's logistic-regression pipeline. Thresholds come from "
        "out-of-fold predictions of the frozen configuration on the three selection repeats (seeds 42-44). "
        f"Scores are cut into bands of width {BIN_WIDTH}; a level starts at the smallest band edge from which "
        f"every band with at least {MIN_BIN} out-of-fold rows is right at least as often as its target "
        "(high: 90%, medium: 70%); lower scores are low. The interface shows the level in words, never the raw "
        "probability.", "",
        "| task | high from | medium from | level | selection OOF rows | share | accuracy "
        "| fresh-seed OOF share | accuracy |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for task, t in p["tasks"].items():
        th = t["certainty"]["thresholds"]
        for sel, fr in zip(t["certainty"]["selection_levels"], t["certainty"]["fresh_levels"]):
            lines.append(f"| {task} | {th['high']:.2f} | {th['medium']:.2f} | {sel['level']} | {sel['n']} | "
                         f"{sel['share']:.3f} | {sel['accuracy']:.3f} | {fr['share']:.3f} | {fr['accuracy']:.3f} |")
    lines += [
        "", "Caveats: the final models are trained on the whole development partition (about 25% more rows than "
        "a CV training fold), so their probabilities may be somewhat higher and the share of high-certainty "
        "answers larger. The real accuracy per level is measured on the test partition in M5 and reported as "
        "found. The fresh-seed columns use folds that played no role in setting the thresholds.", "",
        "## 3. Expected test range", "",
        "Computed after the configurations were frozen, from CV repeats with fresh seeds "
        f"{', '.join(map(str, p['fresh_seeds']))} (grouped and stratified like the selection folds, never used "
        "for any choice): mean ± 2 standard deviations of the 15 fold scores.", "",
        "| task | configuration | mean Macro-F1 | sd | expected range |", "|---|---|---|---|---|",
    ]
    for task, t in p["tasks"].items():
        for role in ("final", "reference"):
            lines.append(f"| {task} | {role} `{t[role]['name']}` | {_range_cell(t['expected_range'][role])} |")
    j = p["joint_expected_range"]
    lines += [
        "", f"Joint accuracy (both labels right, final configurations): mean {j['mean']:.4f}, sd {j['sd']:.4f}, "
        f"expected range [{j['low']:.3f}, {j['high']:.3f}].", "",
        "A CV fold holds about 443 reviews and is predicted by a model trained on about 1,770; the test partition "
        "holds 554 and is predicted by a model trained on all 2,215, so the test score may sit slightly above the "
        "CV mean. The range is recorded for comparison only: it is not a pass mark, and no result outside it "
        "leads to another test run.", "",
        "## 4. Clean-tree rerun of the cited rows", "",
        "The configurations cited here were run again on the selection folds from a committed tree and compared "
        "fold by fold with the logged rows they were selected from.", "",
        "| task | configuration | original commit | rerun commit | original mean | rerun mean "
        "| largest fold difference |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in p["clean_rerun"]:
        lines.append(f"| {r['task']} | `{r['name']}` | {r['original_sha']} | {r['rerun_sha']} | "
                     f"{r['original_mean']:.4f} | {r['rerun_mean']:.4f} | {r['max_fold_diff']:.1e} |")
    b = p["bootstrap"]
    lines += [
        "", "## 5. One-shot test procedure (M5)", "",
        "1. `python -m arabic_reviews final` recomputes the fingerprint from the code and this protocol and stops "
        "if they differ.",
        "2. For each task it fits the frozen configuration, the reference model and both dummy baselines on the "
        "whole development partition.",
        "3. It reads the held-out test partition once, predicts it once per model, and records the access "
        "(time, commit, fingerprint, file hash) in `reports/test_access_log.json`. A second run is refused "
        "unless an explicit flag and a written methodological reason are given; the first result stays the "
        "reported one.",
        "4. Reported per task: Macro-F1 (primary), per-class precision / recall / F1, balanced accuracy, "
        "accuracy and the confusion matrix; joint accuracy over both tasks; accuracy and share per certainty "
        "level; the position of the test Macro-F1 relative to the expected range above.",
        f"5. Uncertainty: {b['level']:.0%} percentile bootstrap intervals for Macro-F1 and every per-class F1 "
        f"with {b['resamples']} resamples (seed {b['seed']}), resampling whole near-duplicate groups "
        f"(`{b['unit']}`) because rows of a group are not independent, and a paired bootstrap of the Macro-F1 "
        "difference between the final configuration and the reference model on the same resamples. No claim is "
        "made from very small cells (for example prices x neutral).",
        "6. Errors on the test partition are described, but no model, threshold or feature is changed after the "
        "test has been read.", "",
    ]
    return "\n".join(lines)


def write_protocol(p: dict) -> None:
    PROTOCOL_JSON.write_text(json.dumps(_rounded(p), indent=1, ensure_ascii=False, sort_keys=True) + "\n",
                             encoding="utf-8", newline="\n")
    PROTOCOL_MD.write_text(protocol_markdown(p), encoding="utf-8", newline="\n")
