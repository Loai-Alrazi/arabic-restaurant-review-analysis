"""Helpers behind ``notebooks/project_walkthrough.ipynb``.

The notebook walks through the project in order and keeps its cells short: the logic lives in the
project modules, and this module only prepares the local data in a fresh clone, locks the held-out
test file for the session, records the checks the notebook makes, and turns logs and reports into
tables and figures. Development-partition results are recomputed and compared with the committed
reports; held-out test results are read from ``reports/final_test.json`` only and never recomputed.
Figures use English label names because matplotlib does not shape Arabic script.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import platform
import sys
import unicodedata
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, eda, experiments, final, ingest, predict, protocol, split, text, validate
from .features import REPRESENTATIONS, build_pipeline

TASKS = tuple(experiments.TASKS)
LABEL_EN = eda.LABEL_EN

# ---- checks -----------------------------------------------------------------------------------

CHECKS: dict[str, dict] = {}


def check(name: str, ok: bool, detail: str = "") -> None:
    """Record one verification of the notebook run; a failed check stops the run."""
    CHECKS[name] = dict(check=name, result="pass" if ok else "FAIL", detail=detail)
    if not ok:
        raise AssertionError(f"{name}: {detail}")
    print(f"✓ {name}" + (f" — {detail}" if detail else ""))


def skip(name: str, reason: str) -> None:
    """Record a check that does not apply in this environment, with the reason."""
    CHECKS[name] = dict(check=name, result="not applicable", detail=reason)
    print(f"– {name}: {reason}")


def checks_table() -> pd.DataFrame:
    return pd.DataFrame(list(CHECKS.values()), columns=["check", "result", "detail"])


def _rel(path: Path) -> str:
    """Path relative to the repository (outputs never show local absolute paths)."""
    path = Path(path).resolve()
    return path.relative_to(config.ROOT.resolve()).as_posix() if path.is_relative_to(config.ROOT.resolve()) else path.name


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frame_sha256(frame: pd.DataFrame) -> str:
    """SHA-256 of a frame written the way the project writes its CSV files."""
    return hashlib.sha256(frame.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()


# ---- display ----------------------------------------------------------------------------------

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def setup_display() -> None:
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 100, "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold",
        "axes.titlecolor": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.8, "xtick.color": AXIS, "ytick.color": AXIS,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "legend.frameon": False,
        "lines.linewidth": 2, "lines.solid_capstyle": "round",
        # SVG figures in the notebook: text kept as text, stable element ids so a re-run gives the same output
        "svg.fonttype": "none", "svg.hashsalt": "arabic-reviews",
    })
    pd.set_option("display.max_colwidth", 100)
    pd.set_option("display.precision", 4)
    pd.set_option("display.width", 160)


def _cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("ramp", RAMP)


def environment_table() -> pd.DataFrame:
    """Installed library versions next to the ones recorded with the saved models."""
    recorded = json.loads(predict.CARD_FILE.read_text(encoding="utf-8"))["versions"]
    rows = []
    for name in ("python", "scikit-learn", "numpy", "pandas", "joblib"):
        installed = platform.python_version() if name == "python" else metadata.version(name)
        rows.append(dict(library=name, installed=installed, recorded_with_models=recorded[name],
                         same=installed == recorded[name]))
    return pd.DataFrame(rows)


# ---- local data and the test lock -------------------------------------------------------------

PUBLISHED_BY_STEPS = (config.MANIFEST_FILE, config.QUALITY_REPORT, config.LINEAGE_FILE, config.SPLIT_FILE,
                      config.NEAR_DUPLICATES_FILE, config.SPLIT_REPORT)


def _without_date(manifest: bytes) -> dict:
    return {k: v for k, v in json.loads(manifest.decode("utf-8")).items() if k != "retrieved"}


def prepare_local_data() -> pd.DataFrame:
    """Make the local data files the notebook needs with the project's own steps.

    The source file is downloaded and verified when it is missing (``ingest``); the ``validate`` and
    ``split`` steps run, in that order, only when the development file is missing. Those steps rewrite
    published files: the manifest keeps its published retrieval date when nothing else differs, and any
    other published file that comes out different is restored and stops the run."""
    from . import __main__ as cli

    published = {p: p.read_bytes() for p in PUBLISHED_BY_STEPS}
    rows = []
    if config.RAW_FILE.exists():
        ingest.verify_file(config.RAW_FILE, config.SOURCE)
        rows.append((_rel(config.RAW_FILE), "present; size, MD5 and SHA-256 verified"))
    else:
        ingest.ingest()
        if _without_date(config.MANIFEST_FILE.read_bytes()) == _without_date(published[config.MANIFEST_FILE]):
            config.MANIFEST_FILE.write_bytes(published[config.MANIFEST_FILE])  # same source, published date
        rows.append((_rel(config.RAW_FILE), "downloaded from Zenodo and verified (ingest step)"))
    ran = not config.DEV_FILE.exists()
    if ran:
        with contextlib.redirect_stdout(io.StringIO()):
            cli.cmd_validate()
            cli.cmd_split()
    rows.append((_rel(config.DEV_FILE), "written by the validate and split steps" if ran else "present"))
    changed = [p for p in PUBLISHED_BY_STEPS if p.read_bytes() != published[p]]
    for p in changed:
        p.write_bytes(published[p])
    check("published data manifest, validation and split reports unchanged", not changed,
          "byte-identical" if not changed else f"differed and were restored: {[_rel(p) for p in changed]}")
    return pd.DataFrame(rows, columns=["local file", "state"])


_LOCKED_PATHS: set[str] = set()
_LOCKED_IDS: set[tuple[int, int]] = set()


def _norm(path: str) -> str:
    return os.path.normcase(os.path.realpath(path))


def _identity(path: str) -> tuple[int, int] | None:
    try:
        st = os.stat(path)
    except (OSError, ValueError):
        return None
    return (st.st_dev, st.st_ino) if st.st_ino else None


def _refuse_locked(event: str, args: tuple) -> None:
    if event != "open" or not _LOCKED_PATHS or isinstance(args[0], int):
        return  # an integer is an already open descriptor, whose opening was itself checked
    try:
        path = os.fsdecode(args[0])
    except TypeError:
        return
    if _norm(path) in _LOCKED_PATHS or (_LOCKED_IDS and _identity(path) in _LOCKED_IDS):
        raise PermissionError(f"{os.path.basename(path)} is the locked held-out test file; its results are "
                              "shown from the saved reports only")


def lock_test_partition() -> str:
    """Refuse every later attempt in this Python process to open the held-out test file.

    A ``sys.addaudithook`` hook sees every file opening (``open``, ``pathlib``, pandas, ``os.open``) and
    matches the resolved path and the file identity, so links and relative or byte paths are refused
    too. It cannot be removed, so it lasts until the kernel restarts. It is a guard against accidental
    reads, not a sandbox: code could still copy the file through other processes."""
    if not _LOCKED_PATHS:
        sys.addaudithook(_refuse_locked)
    _LOCKED_PATHS.add(_norm(os.fspath(config.TEST_FILE)))
    identity = _identity(os.fspath(config.TEST_FILE))
    if identity:
        _LOCKED_IDS.add(identity)
    name = _rel(config.TEST_FILE)
    print(f"locked: {name} cannot be opened again in this session")
    return name


def access_log_table() -> pd.DataFrame:
    entries = json.loads(final.ACCESS_LOG.read_text(encoding="utf-8"))
    check("the held-out test was read exactly once", len(entries) == 1, f"{len(entries)} entry in {_rel(final.ACCESS_LOG)}")
    return pd.DataFrame(entries)[["run", "time", "commit", "test_rows", "reason", "fingerprint"]]


# ---- 1. problem definition -------------------------------------------------------------------

def predeclared_settings() -> pd.DataFrame:
    """The task definitions and decision rules as they are written in the code."""
    b = protocol.BOOTSTRAP
    rows = [
        ("sentiment labels", " / ".join(config.SENTIMENTS)),
        ("aspect labels", " / ".join(config.ASPECTS)),
        ("primary metric", "Macro-F1 per task (fixed label list, zero_division=0)"),
        ("secondary metrics", "balanced accuracy, accuracy, joint accuracy (both labels right)"),
        ("selection CV", f"{len(split.CV_SEEDS)} x {split.N_SPLITS} grouped stratified folds, seeds {split.CV_SEEDS}"),
        ("acceptance rule", f"mean paired gain >= {experiments.MIN_DELTA} and a gain in >= {experiments.MIN_WINS} of 15 folds"),
        ("baselines", ", ".join(protocol.BASELINES)),
        ("expected test range", f"fresh CV seeds {protocol.FRESH_SEEDS}, never used for a choice"),
        ("test intervals", f"{b['level']:.0%} percentile bootstrap, {b['resamples']} resamples, near-duplicate groups as unit"),
        ("success on the test", "beat both dummies (paired interval > 0) and not below the reference"),
    ]
    return pd.DataFrame(rows, columns=["setting", "value"])


# ---- 2. ingestion and validation --------------------------------------------------------------

def source_fingerprint() -> pd.DataFrame:
    actual = ingest.fingerprint(config.RAW_FILE)
    rows = [dict(field=k, published=str(config.SOURCE[k]), local=str(actual[k]), same=config.SOURCE[k] == actual[k])
            for k in ("size_bytes", "md5", "sha256")]
    table = pd.DataFrame(rows)
    check("source file matches the published fingerprint", bool(table.same.all()), config.SOURCE["doi"])
    return table


def read_source() -> pd.DataFrame:
    return pd.read_excel(config.RAW_FILE, sheet_name=0, dtype=object, engine="openpyxl")


def _visible(t: str, width: int = 90) -> str:
    """Text with line breaks and invisible characters made visible, shortened to ``width``."""
    out = "".join("⏎" if ch == "\n" else f"⟨U+{ord(ch):04X}⟩" if unicodedata.category(ch) in ("Cf", "Cc") else ch
                  for ch in t)
    return out if len(out) <= width else out[: width - 1] + "…"


def repair_examples(raw: pd.DataFrame) -> pd.DataFrame:
    """The shortest real example of each repair step, before and after the step (preferring the typical
    case: both labels glued to the text, an invisible character removed)."""
    steps = [("strip_label_suffix", lambda t: validate.strip_label_suffix(t)[0],
              lambda t: validate.strip_label_suffix(t)[2] is not None),
             ("strip_translation", validate.strip_translation, lambda t: True),
             ("clean_whitespace", validate.clean_whitespace,
              lambda t: any(unicodedata.category(ch) == "Cf" for ch in t))]
    current = raw["Review"].tolist()
    rows = []
    for name, fn, typical in steps:
        after = [fn(t) for t in current]
        changed = [i for i, (a, b) in enumerate(zip(current, after)) if a != b]
        i = min(changed, key=lambda k: (not typical(current[k]), len(current[k]), k))
        rows.append(dict(step=name, rows_changed=len(changed), rid=i, before=_visible(current[i]),
                         after=_visible(after[i])))
        current = after
    return pd.DataFrame(rows)


def lineage_table(result: validate.ValidationResult) -> pd.DataFrame:
    return pd.DataFrame(result.lineage["steps"]).set_index("step")


def verify_validation(result: validate.ValidationResult) -> None:
    published = json.loads(config.LINEAGE_FILE.read_text(encoding="utf-8"))
    lin = result.lineage
    check("validation reconciles", lin["raw_rows"] == lin["validated_rows"] + sum(lin["excluded_by_reason"].values()),
          f"{lin['raw_rows']} = {lin['validated_rows']} kept + {sum(lin['excluded_by_reason'].values())} excluded")
    check("validation steps equal the published lineage", lin["steps"] == published["steps"], _rel(config.LINEAGE_FILE))
    got = frame_sha256(result.data)
    check("validated data byte-identical to the published hash", got == published["output"]["sha256"], f"SHA-256 {got[:12]}…")


def quality_table(result: validate.ValidationResult) -> pd.DataFrame:
    raw, val = result.profile_raw["rows_with"], result.profile_validated["rows_with"]
    table = pd.DataFrame({"raw": raw, "validated": val})
    table.index.name = "rows with"
    return table


def partition_overview(dev: pd.DataFrame) -> pd.DataFrame:
    """Rows and near-duplicate groups per partition, from the published (text-free) split file."""
    assignment = pd.read_csv(config.SPLIT_FILE)
    table = assignment.groupby("partition").agg(rows=("rid", "size"), groups=("group_id", "nunique"))
    table["share"] = (table.rows / table.rows.sum()).round(3)
    published = assignment[assignment.partition == "dev"].drop(columns="partition").sort_values("rid").reset_index(drop=True)
    got = dev[list(published.columns)].sort_values("rid").reset_index(drop=True)
    check("the development data is exactly the published development partition", got.equals(published),
          f"{len(dev)} rows: rid, group and CV folds equal to {_rel(config.SPLIT_FILE)}")
    return table.loc[["dev", "test"]]


# ---- 3. EDA (development partition only) ------------------------------------------------------

def label_counts(dev: pd.DataFrame) -> pd.DataFrame:
    table = pd.crosstab(dev.aspect, dev.sentiment, margins=True, margins_name="total")
    return table.loc[[*config.ASPECTS, "total"], [*config.SENTIMENTS, "total"]]


def _bars(ax, labels, values, color=BLUE, fmt="{:.1%}"):
    y = np.arange(len(labels))
    ax.barh(y, values, height=0.55, color=color)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    for yi, v in zip(y, values):
        ax.text(v, yi, " " + fmt.format(v), va="center", color=INK2, fontsize=8.5)


def plot_labels(dev: pd.DataFrame):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.3), gridspec_kw=dict(width_ratios=[1, 1, 1.25]))
    for ax, col, labels, title in ((axes[0], "sentiment", config.SENTIMENTS, "Sentiment share"),
                                   (axes[1], "aspect", config.ASPECTS, "Aspect share")):
        share = dev[col].value_counts(normalize=True).reindex(labels)
        _bars(ax, [LABEL_EN[k] for k in labels], share.to_numpy())
        ax.set_xlim(0, share.max() * 1.3)
        ax.set_title(title)
    joint = pd.crosstab(dev.aspect, dev.sentiment).loc[list(config.ASPECTS), list(config.SENTIMENTS)]
    _heatmap(axes[2], joint, "Aspect x sentiment (rows of dev)")
    fig.suptitle(f"Development partition, n = {len(dev)}", x=0.01, ha="left", color=INK2, fontsize=9.5)
    fig.tight_layout()
    return fig


def _heatmap(ax, frame: pd.DataFrame, title: str, normalise_rows: bool = False):
    values = frame.to_numpy(dtype=float)
    shade = values / values.sum(axis=1, keepdims=True) if normalise_rows else values / values.max()
    # vector cells with a surface-coloured gap (no embedded raster image)
    ax.pcolormesh(shade, cmap=_cmap(), vmin=0, vmax=1, edgecolors=SURFACE, linewidth=2)
    ax.set_xticks(np.arange(frame.shape[1]) + 0.5, [LABEL_EN.get(c, c) for c in frame.columns])
    ax.set_yticks(np.arange(frame.shape[0]) + 0.5, [LABEL_EN.get(i, i) for i in frame.index])
    ax.invert_yaxis()
    ax.tick_params(length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for (i, j), v in np.ndenumerate(values):
        ax.text(j + 0.5, i + 0.5, f"{v:g}", ha="center", va="center", fontsize=8.5,
                color="#ffffff" if shade[i, j] > 0.55 else INK)
    ax.set_title(title)


def length_table(dev: pd.DataFrame) -> pd.DataFrame:
    words = dev.text.str.split().str.len()
    chars = dev.text.str.len()
    table = pd.DataFrame({"rows": dev.groupby("sentiment").size(),
                          "words median": words.groupby(dev.sentiment).median(),
                          "words p90": words.groupby(dev.sentiment).quantile(0.9),
                          "chars median": chars.groupby(dev.sentiment).median(),
                          "<= 3 words": (words <= 3).groupby(dev.sentiment).mean().round(3)})
    return table.loc[list(config.SENTIMENTS)]


def plot_lengths(dev: pd.DataFrame):
    import matplotlib.pyplot as plt

    words = dev.text.str.split().str.len().clip(upper=60)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2), gridspec_kw=dict(width_ratios=[3, 5]))
    for ax, col, labels in ((axes[0], "sentiment", config.SENTIMENTS), (axes[1], "aspect", config.ASPECTS)):
        ax.boxplot([words[dev[col] == k] for k in labels], tick_labels=[LABEL_EN[k] for k in labels], widths=0.45,
                   medianprops=dict(color=BLUE, linewidth=2), boxprops=dict(color=INK2), whiskerprops=dict(color=INK2),
                   capprops=dict(color=INK2), flierprops=dict(marker="o", markersize=3, markerfacecolor=MUTED,
                                                              markeredgecolor="none", alpha=0.5))
        ax.set_title(f"Words per review by {col} (clipped at 60)")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("words")
    fig.tight_layout()
    return fig


def signal_rates(dev: pd.DataFrame) -> pd.DataFrame:
    """Share of reviews with a negation particle (with and without ما) and with an emoji, per sentiment."""
    flags = pd.DataFrame({
        "% with a negation particle": dev.text.map(eda._has_negation),
        "% without counting ما": dev.text.map(lambda t: eda._has_negation(t, text.NEGATIONS - {"ما"})),
        "% with an emoji": dev.text.str.contains(eda.EMOJI_RE),
    })
    return (flags.groupby(dev.sentiment).mean().loc[list(config.SENTIMENTS)] * 100).round(1)


def distinctive_terms(dev: pd.DataFrame, k: int = 6) -> pd.DataFrame:
    rows = []
    for task, labels in (("sentiment", config.SENTIMENTS), ("aspect", config.ASPECTS)):
        grams = eda.distinctive_ngrams(dev.text, dev[task], k=k)
        rows += [dict(task=task, label=lab, terms="، ".join(grams[lab])) for lab in labels]
    return pd.DataFrame(rows).set_index(["task", "label"])


def multi_aspect(dev: pd.DataFrame) -> pd.DataFrame:
    m = eda.multi_aspect_estimate(dev)
    counts = {f"{k} aspect cue(s)" if k < 3 else "3+ aspect cues": m["cue_counts"].get(k, 0) for k in range(4)}
    rows = list(counts.items()) + [
        ("reviews with cues for >= 2 aspects", f"{m['multi_cue_pct']:.1f}%"),
        ("labelled aspect among the cued ones (multi-cue reviews)", f"{m['multi_cue_label_among_cues_pct']:.1f}%"),
        ("single cue agrees with the label (sanity check)", f"{m['single_cue_agrees_with_label_pct']:.1f}%"),
    ]
    return pd.DataFrame(rows, columns=["measure", "value"])


# ---- 4. data preparation ----------------------------------------------------------------------

NORMALISATION_EXAMPLES = (
    "القهوةةة لذيييذة جداً 😍😍",
    "ما عجبني الأكل، كان بارد!!",
    "سعر الكوب ١٨ ريال والخدمة ممتــازة",
    "الجلسات في الدور الثاني مو مريحة",
)


def normalisation_table(examples=NORMALISATION_EXAMPLES) -> pd.DataFrame:
    """Each model-side text option of ``text.py`` applied alone to example reviews."""
    options = {
        "light (LIGHT)": lambda t: text.normalize(t, **text.LIGHT),
        "full (+ unify letters)": lambda t: text.normalize(t, **text.FULL),
        "symbols separate": lambda t: text.normalize(t, symbols="separate"),
        "digits token": lambda t: text.normalize(t, digits="token"),
        "negation marked": text.mark_negation,
        "stopwords removed": text.remove_stopwords,
    }
    rows = {name: [fn(t) for t in examples] for name, fn in options.items()}
    return pd.DataFrame(rows, index=pd.Index(examples, name="original")).T


# ---- 5. feature engineering -------------------------------------------------------------------

def _analyzer(representation: str):
    return build_pipeline(dict(model="logreg", representation=representation)).named_steps["features"].build_analyzer()


def char_ngram_overlap(pairs) -> pd.DataFrame:
    """Shared word tokens versus shared character 2-5-grams (the model's ``char25`` analyzer)."""
    words, chars = _analyzer("word11"), _analyzer("char25")
    rows = []
    for a, b in pairs:
        ca, cb = set(chars(a)), set(chars(b))
        longest = sorted({g.strip() for g in ca & cb if len(g.strip()) >= 3}, key=lambda g: (-len(g), g))[:3]
        rows.append(dict(pair=f"{a} / {b}", shared_word_tokens=len(set(words(a)) & set(words(b))),
                         shared_char_ngrams=len(ca & cb), jaccard_char=round(len(ca & cb) / len(ca | cb), 2),
                         longest_shared="، ".join(longest)))
    return pd.DataFrame(rows)


def representation_sizes(dev: pd.DataFrame) -> pd.DataFrame:
    """Vocabulary size and non-zero features per review for each candidate representation.

    Descriptive only (fitted on the whole development partition to show the size of each space);
    the models learn their vocabulary inside each training fold."""
    rows = []
    for rep in REPRESENTATIONS:
        feats = build_pipeline(dict(model="logreg", representation=rep)).named_steps["features"]
        X = feats.fit_transform(dev.text)
        rows.append(dict(representation=rep, features=X.shape[1], nonzero_per_review=round(X.getnnz(axis=1).mean(), 1)))
    return pd.DataFrame(rows).set_index("representation")


# ---- 2.4 / 6. split and leakage checks --------------------------------------------------------

def split_audit(validated: pd.DataFrame, result: split.SplitResult) -> dict[str, pd.DataFrame]:
    """Compare a split rebuilt from the validated data with the published one and return only aggregate
    tables (group sizes, leakage counts, class shares per partition), so the full data can be dropped."""
    published = pd.read_csv(config.SPLIT_FILE)
    check("split rebuilt from the validated data equals the published split",
          result.assignment.reset_index(drop=True).equals(published), _rel(config.SPLIT_FILE))
    pairs = pd.read_csv(config.NEAR_DUPLICATES_FILE, keep_default_na=False)
    check("near-duplicate pairs equal the published list", result.pairs.reset_index(drop=True).equals(pairs),
          f"{len(pairs)} pairs")
    counts = split.leakage_checks(validated, result.assignment, result.pairs)
    check("leakage checks are all zero", not any(counts.values()), f"{len(counts)} counts")
    sizes = result.assignment.groupby("group_id").size().value_counts().sort_index().rename("groups").to_frame()
    sizes.index.name = "rows in group"
    sizes["rows"] = sizes.index * sizes.groups
    return dict(groups=sizes.T, leakage=pd.Series(counts, name="count").to_frame(),
                proportions=split.proportions(validated, result.assignment).round(2))


# ---- 7. pipelines -----------------------------------------------------------------------------

def frozen_specs() -> dict[str, dict[str, dict]]:
    """Final and reference configurations per task as frozen in ``reports/protocol.json``."""
    p = json.loads(protocol.PROTOCOL_JSON.read_text(encoding="utf-8"))
    return {role: {task: p["tasks"][task][role] for task in TASKS} for role in ("final", "reference")}


def _vectorizers(step) -> list[tuple[str, object]]:
    return list(step.transformer_list) if hasattr(step, "transformer_list") else [("", step)]


def in_fold_vocabulary(dev: pd.DataFrame, spec: dict, task: str, column: str = split.CV_COLUMNS[0],
                       fold: int = 0) -> pd.DataFrame:
    """Fit the pipeline on the training rows of one fold and count the validation fold's n-grams
    that its vocabulary has never seen."""
    train, valid = dev[column] != fold, dev[column] == fold
    model = build_pipeline(spec).fit(dev.text[train], dev[task][train])
    normalised = model.named_steps["normalize"].transform(dev.text[valid])
    rows = []
    for name, vec in _vectorizers(model.named_steps["features"]):
        analyze = vec.build_analyzer()
        grams = set().union(*(analyze(t) for t in normalised))
        unseen = grams - set(vec.vocabulary_)
        rows.append(dict(vectorizer=name or spec["representation"], train_rows=int(train.sum()),
                         valid_rows=int(valid.sum()), learned_vocabulary=len(vec.vocabulary_),
                         idf_weights=len(vec.idf_), valid_ngrams=len(grams), unseen_in_training=len(unseen),
                         unseen_share=round(len(unseen) / len(grams), 3)))
    return pd.DataFrame(rows).set_index("vectorizer")


# ---- 8-9. baselines, CV and feature-engineering decisions --------------------------------------

def baseline_table() -> pd.DataFrame:
    log = experiments.read_log()
    rows = []
    for task in TASKS:
        for name in protocol.BASELINES:
            r = log.loc[log.exp_id == experiments.exp_id(task, name)].iloc[0]
            rows.append(dict(task=task, configuration=name, model=r.model, cv_macro_f1=r.cv_macro_f1_mean,
                             sd=r.cv_macro_f1_std, balanced_accuracy=r.balanced_acc_mean, accuracy=r.accuracy_mean))
    return pd.DataFrame(rows).set_index(["task", "configuration"])


def rerun_cv(dev: pd.DataFrame, items) -> pd.DataFrame:
    """Re-run the 3 x 5 CV of ``(task, spec)`` items and compare every fold with the logged row."""
    log = experiments.read_log()
    rows = []
    for task, spec in items:
        res = experiments.cv_scores(spec, dev, task)
        logged = experiments.fold_scores(log, experiments.exp_id(task, spec["name"]))
        diff = float(np.abs(np.array(res["f1"]) - np.array(logged)).max())
        rows.append(dict(task=task, configuration=spec["name"], logged_mean=float(np.mean(logged)),
                         rerun_mean=float(np.mean(res["f1"])), largest_fold_difference=diff,
                         fit_seconds=round(res["seconds"], 1)))
        # the log stores fold scores rounded to 6 decimals
        check(f"{task}: CV of {spec['name']} reproduces the logged 15 fold scores", diff <= 1e-6, f"largest difference {diff:.1e}")
    return pd.DataFrame(rows).set_index(["task", "configuration"])


def _decision(accepted: bool) -> str:
    return "accepted" if accepted else "rejected"


def ablation_table(task: str) -> pd.DataFrame:
    """Axis 1 and axes 2-6 as decided from the log by the code (nothing is run)."""
    log = experiments.read_log()
    rows = [dict(axis="Axis 1 - representation", candidate=s["candidate"], against=s["winner_before"],
                 mean_gain=s["mean_delta"], folds_won=s["wins"], decision=_decision(s["accepted"]))
            for s in experiments.decide_chain(log, task, experiments.AXIS1_CHAIN)]
    axes, _ = experiments.resolve_axes(task)
    rows += [dict(axis=r["axis"], candidate=r["candidate"], against=r["winner_before"], mean_gain=r["mean_delta"],
                  folds_won=r["wins"], decision=_decision(r["accepted"])) for r in axes]
    return pd.DataFrame(rows)


def plot_ablation(tables: dict[str, pd.DataFrame]):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(tables), figsize=(12, 4.6), sharey=False)
    for ax, (task, t) in zip(axes, tables.items()):
        y = np.arange(len(t))
        colors = [BLUE if d == "accepted" else AXIS for d in t.decision]
        ax.barh(y, t.mean_gain, height=0.55, color=colors)
        ax.axvline(0, color=INK2, linewidth=0.8)
        ax.axvline(experiments.MIN_DELTA, color=ORANGE, linewidth=1.2)
        ax.set_yticks(y, [c.replace("logreg_", "") for c in t.candidate])
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        span = t.mean_gain.abs().max()
        for yi, (g, w, d) in enumerate(zip(t.mean_gain, t.folds_won, t.decision)):
            # labels start right of the threshold line (or left of a negative bar) so they never cross it
            x = max(g, experiments.MIN_DELTA) + span * 0.04 if g >= 0 else g - span * 0.04
            ax.text(x, yi, f"{w}/15 {d}", va="center", ha="left" if g >= 0 else "right", color=INK2, fontsize=8)
        ax.set_xlim(-span * 1.9, span * 1.9)
        ax.set_title(f"{task}: mean paired Macro-F1 gain vs current winner")
        ax.set_xlabel(f"gain (orange line: acceptance threshold {experiments.MIN_DELTA}; also needs >= "
                      f"{experiments.MIN_WINS}/15 folds)")
    fig.tight_layout()
    return fig


def decision_path(task: str) -> list[str]:
    """Reference -> representation winner -> class weighting -> tuned final, as logged."""
    log = experiments.read_log()
    path = [protocol.REFERENCE, experiments.axis1_winner(log, task)]
    _, m3 = experiments.resolve_axes(task)
    path.append(m3["name"])
    path.append(experiments.resolve_m4(task)["final"]["name"])
    return list(dict.fromkeys(path))


def plot_decision_path():
    import matplotlib.pyplot as plt

    log = experiments.read_log()
    fig, axes = plt.subplots(1, len(TASKS), figsize=(12, 3.8))
    for ax, task in zip(axes, TASKS):
        names = decision_path(task)
        scores = np.array([experiments.fold_scores(log, experiments.exp_id(task, n)) for n in names])
        x = np.arange(len(names))
        for fold in scores.T:
            ax.plot(x, fold, color=AXIS, linewidth=0.8, zorder=1)
        ax.scatter(np.repeat(x, scores.shape[1]), scores.ravel(), s=16, color=BLUE, edgecolor=SURFACE,
                   linewidth=1, zorder=2)
        means = scores.mean(axis=1)
        ax.scatter(x, means, marker="_", s=500, color=INK, linewidth=2, zorder=3)
        for xi, m in zip(x, means):
            ax.text(xi + 0.12, m, f"{m:.3f}", va="center", color=INK, fontsize=8.5)
        ax.set_xticks(x, [n.replace("logreg_", "") for n in names], rotation=12)
        ax.set_title(f"{task}: 15 fold scores along the decision path")
        ax.set_ylabel("CV Macro-F1")
        ax.grid(axis="x", visible=False)
    fig.text(0.01, 0.0, "Grey lines join the same fold across configurations (paired comparison); black bar = mean.",
             color=INK2, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


# ---- 10. model selection and the frozen protocol ----------------------------------------------

def model_selection(task: str) -> dict:
    """Stage A (families), stage B (regularisation grid) and tf-idf options, decided from the log."""
    r = experiments.resolve_m4(task)
    families = pd.DataFrame([dict(stage="family", candidate=f["candidate"], against=f["vs"], mean_gain=f["mean_delta"],
                                  folds_won=f["wins"], decision=_decision(f["accepted"])) for f in r["families"]]
                            + [dict(stage="tf-idf option", candidate=s["candidate"], against=s["vs"],
                                    mean_gain=s["mean_delta"], folds_won=s["wins"], decision=_decision(s["accepted"]))
                               for s in r["tfidf_steps"]])
    grid = pd.DataFrame([dict(value=t["value"], mean=t["mean"], sd=t["std"], se=t["std"] / np.sqrt(len(t["folds"])))
                         for t in r["tuning"]])
    gate = r["gate"]
    summary = dict(task=task, family=r["family"], param=r["param"], one_se_pick=r["one_se"], used=r["picked"],
                   complexity_gate=("not needed" if gate is None else
                                    f"{gate['mean_delta']:+.4f}, {gate['wins']}/{gate['n']} -> {_decision(gate['accepted'])}"),
                   final=r["final"]["name"])
    return dict(table=families, grid=grid, summary=summary)


def plot_c_grid(selection: dict[str, dict]):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(selection), figsize=(12, 3.4))
    for ax, (task, s) in zip(axes, selection.items()):
        g, summ = s["grid"], s["summary"]
        x = np.arange(len(g))
        ax.errorbar(x, g["mean"], yerr=g.se, color=BLUE, ecolor=BLUE, elinewidth=1.2, capsize=3, marker="o",
                    markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5)
        best = int(g["mean"].idxmax())
        ax.axhline(g["mean"][best] - g.se[best], color=ORANGE, linewidth=1.2)
        for xi, row in g.iterrows():
            tags = [t for t, v in (("one-SE pick", summ["one_se_pick"]), ("used", summ["used"])) if row.value == v]
            if tags:
                ax.annotate(" + ".join(tags), (xi, row["mean"]), xytext=(0, 12), textcoords="offset points",
                            ha="center", color=INK, fontsize=8.5, fontweight="bold")
        ax.set_xticks(x, [f"{v:g}" for v in g.value])
        ax.set_xlabel(f"{summ['param']} (left = stronger regularisation)")
        ax.set_ylabel("CV Macro-F1 (mean ± 1 SE)")
        ax.set_title(f"{task}: logistic-regression {summ['param']} grid")
        ax.grid(axis="x", visible=False)
        ax.margins(y=0.25)
    fig.text(0.01, 0.0, "Orange line: best mean minus one standard error; the simplest value above it is the one-SE pick.",
             color=INK2, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def rebuild_protocol(dev: pd.DataFrame) -> dict:
    """Rebuild the protocol from code and the development partition and compare it with the frozen one
    (the comparison ``final --check`` makes before it fits the models); nothing is written."""
    stored = json.loads(protocol.PROTOCOL_JSON.read_text(encoding="utf-8"))
    rebuilt = protocol.build_protocol(dev, rerun=False)
    try:
        final.check_fingerprint(rebuilt, stored)
        ok, detail = True, f"fingerprint {rebuilt['fingerprint'][:16]}… = frozen"
    except final.ProtocolMismatch as err:
        ok, detail = False, str(err)
    check("protocol rebuilt from code and development data equals the frozen protocol", ok, detail)
    return final.with_frozen_results(rebuilt, stored)


def frozen_configurations(p: dict) -> pd.DataFrame:
    log = experiments.read_log()
    rows = []
    for task, t in p["tasks"].items():
        for role in ("final", "reference"):
            s = t[role]
            r = log.loc[log.exp_id == experiments.exp_id(task, s["name"])].iloc[0]
            rows.append(dict(task=task, role=role, configuration=s["name"], model=s["model"],
                             features=f"{s['representation']} TF-IDF", normalisation=s["normalisation"],
                             class_weight=s["class_weight"], params=json.dumps(s["params"]),
                             cv_macro_f1=f"{r.cv_macro_f1_mean:.4f} ± {r.cv_macro_f1_std:.4f}"))
    return pd.DataFrame(rows).set_index(["task", "role"])


def expected_ranges(p: dict) -> pd.DataFrame:
    rows = [dict(task=task, role=role, configuration=t[role]["name"], **{k: t["expected_range"][role][k]
                 for k in ("mean", "sd", "low", "high")}) for task, t in p["tasks"].items() for role in ("final", "reference")]
    j = p["joint_expected_range"]
    rows.append(dict(task="both", role="joint accuracy", configuration="final + final",
                     **{k: j[k] for k in ("mean", "sd", "low", "high")}))
    return pd.DataFrame(rows).set_index(["task", "role"]).round(4)


def certainty_levels(p: dict) -> pd.DataFrame:
    rows = []
    for task, t in p["tasks"].items():
        c = t["certainty"]
        for sel, fresh in zip(c["selection_levels"], c["fresh_levels"]):
            rows.append(dict(task=task, level=sel["level"], starts_at=c["thresholds"].get(sel["level"], 0.0),
                             selection_share=sel["share"], selection_accuracy=sel["accuracy"],
                             fresh_seed_share=fresh["share"], fresh_seed_accuracy=fresh["accuracy"]))
    return pd.DataFrame(rows).set_index(["task", "level"]).round(3)


def fit_final_models(dev: pd.DataFrame, p: dict) -> dict:
    """Fit the frozen final, reference and dummy pipelines on the whole development partition, as the
    one-shot evaluation did, and compare the final models with the recorded probability digests."""
    models = {task: final.fit_models(dev, p["tasks"][task], task) for task in TASKS}
    stored = json.loads(final.DIGESTS_FILE.read_text(encoding="utf-8"))
    runtime = final.runtime_versions()
    differ = [f"{k} {runtime.get(k)} (recorded {v})" for k, v in stored["versions"].items() if runtime.get(k) != v]
    for task in TASKS:
        # exact comparison only under the recorded versions, the rule `final --check` applies
        name = f"{task}: final model reproduces the recorded development-probability digest"
        digest = final.dev_digest(models[task]["final"], dev["text"])
        if not differ:
            check(name, digest == stored["digests"][task], f"SHA-256 {digest[:12]}…")
        else:
            skip(name, f"compared only under the recorded versions; here {', '.join(differ)}. "
                       "The CV fold and protocol checks above do not depend on this")
    return models


def plot_cv_summary(p: dict):
    """CV Macro-F1 (mean ± sd over the 15 selection folds) from the dummies to the frozen final model."""
    import matplotlib.pyplot as plt

    log = experiments.read_log()
    fig, axes = plt.subplots(1, len(TASKS), figsize=(12, 2.9))
    for ax, task in zip(axes, TASKS):
        names = [*protocol.BASELINES, p["tasks"][task]["final"]["name"]]
        rows = [log.loc[log.exp_id == experiments.exp_id(task, n)].iloc[0] for n in names]
        means = np.array([r.cv_macro_f1_mean for r in rows])
        sds = np.array([r.cv_macro_f1_std for r in rows])
        y = np.arange(len(names))
        ax.barh(y, means, height=0.55, color=[AXIS, AXIS, ORANGE, BLUE], xerr=sds, error_kw=dict(ecolor=INK2, lw=1))
        ax.set_yticks(y, [n.replace("logreg_", "") for n in names])
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        for yi, (m, sd) in enumerate(zip(means, sds)):
            ax.text(m + sd + 0.01, yi, f"{m:.3f}", va="center", color=INK, fontsize=8.5)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("CV Macro-F1 (mean ± sd, 15 folds)")
        ax.set_title(f"{task}: baselines, reference and frozen final")
    fig.tight_layout()
    return fig


# ---- 11. error analysis (development out-of-fold predictions) ---------------------------------

def per_class_tables(oof: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for task, o in oof.items():
        t = experiments.per_class_table(o, experiments.LABELS[task]).assign(task=task)
        frames.append(t)
        print(f"{task}: out-of-fold Macro-F1 {t.f1.mean():.4f}, accuracy {(o.y == o.pred).mean():.4f}, "
              f"errors {(o.y != o.pred).sum()} of {len(o)}")
    return pd.concat(frames).set_index(["task", "label"]).round(3)


def plot_confusions(matrices: dict[str, pd.DataFrame], suptitle: str):
    import matplotlib.pyplot as plt

    widths = [m.shape[1] for m in matrices.values()]
    fig, axes = plt.subplots(1, len(matrices), figsize=(12, 3.6), gridspec_kw=dict(width_ratios=widths))
    for ax, (title, m) in zip(axes, matrices.items()):
        _heatmap(ax, m, title, normalise_rows=True)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
    fig.suptitle(suptitle + " (cell shade = share of the true class)", x=0.01, ha="left", color=INK2, fontsize=9.5)
    fig.tight_layout()
    return fig


def error_categories() -> pd.DataFrame:
    cats = pd.read_csv(experiments.ERROR_CATEGORIES_FILE, keep_default_na=False)
    table = pd.crosstab(cats.category, cats.task).reindex(list(experiments.CATEGORY_NAMES), fill_value=0)
    table = table[list(dict.fromkeys(cats.task))]
    shares = table / table.sum()
    out = pd.concat({"reviews": table, "share": shares.round(2)}, axis=1)
    out.index = [experiments.CATEGORY_NAMES[k] for k in out.index]
    for task, d in experiments.gate_decision(cats).items():
        print(f"{task}: lexical variety {d['share']:.0%} of {d['n']} errors, "
              f"{'the largest' if d['largest'] else 'not the largest'} category -> "
              f"{'gate opens' if d['opens'] else 'gate stays closed'}")
    return out


def plot_error_categories(table: pd.DataFrame):
    import matplotlib.pyplot as plt

    share = table["share"]
    fig, ax = plt.subplots(figsize=(12, 3.6))
    y = np.arange(len(share))
    h = 0.36
    for k, (task, color) in enumerate(zip(share.columns, (BLUE, ORANGE))):
        ax.barh(y + (k - 0.5) * (h + 0.04), share[task], height=h, color=color, label=f"{task} (n={int(table['reviews'][task].sum())})")
    ax.axvline(experiments.GATE_SHARE, color=INK2, linewidth=1)
    ax.set_yticks(y, share.index)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"share of the categorised error sample (vertical line: gate threshold {experiments.GATE_SHARE:.0%})")
    ax.set_title("Manual categories of a seeded sample of out-of-fold errors")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def error_examples(oof: pd.DataFrame, dev: pd.DataFrame, n: int = 4, max_chars: int = 80, seed: int = 0) -> pd.DataFrame:
    """A few short out-of-fold errors (seeded sample of the errors with at most ``max_chars`` characters)."""
    errors = oof[oof.y != oof.pred].merge(dev[["rid", "text"]], on="rid")
    short = errors[errors.text.str.len() <= max_chars].sample(n=n, random_state=seed)
    return short.assign(text=short.text.map(lambda t: _visible(t, max_chars)))[["rid", "y", "pred", "text"]].rename(
        columns={"y": "gold", "pred": "predicted"}).reset_index(drop=True)


def top_features(pipeline, k: int = 8) -> pd.DataFrame:
    """Highest-weighted features per class of a fitted logistic-regression pipeline («» marks a
    character n-gram, · a word boundary inside it)."""
    names = pipeline.named_steps["features"].get_feature_names_out()

    def show(name: str) -> str:
        kind, _, gram = name.rpartition("__")
        if (kind or pipeline.named_steps["features"].analyzer) in ("char25", "char_wb"):
            return "«" + gram.replace(" ", "·") + "»"
        return gram

    clf = pipeline.named_steps["clf"]
    rows = {label: "  ".join(show(names[i]) for i in np.argsort(clf.coef_[c])[::-1][:k])
            for c, label in enumerate(clf.classes_)}
    return pd.Series(rows, name=f"top {k} features").to_frame()


# ---- 12. final results (saved reports only) ---------------------------------------------------

def final_test_report(p: dict) -> dict:
    """``reports/final_test.json`` after checking that it belongs to the frozen protocol rebuilt above."""
    r = json.loads(final.FINAL_JSON.read_text(encoding="utf-8"))
    entries = json.loads(final.ACCESS_LOG.read_text(encoding="utf-8"))
    card = json.loads(predict.CARD_FILE.read_text(encoding="utf-8"))
    check("saved test results come from the frozen protocol",
          r["fingerprint"] == final.FROZEN_FINGERPRINT == p["fingerprint"] == card["fingerprint"],
          f"{_rel(final.FINAL_JSON)}, fingerprint {r['fingerprint'][:16]}…")
    check("saved test results come from the one logged test read",
          len(entries) == 1 and r["access"] == entries[0] and r["access"]["run"] == 1,
          f"run 1, {r['access']['time']}, commit {r['access']['commit']}")
    return r


ROLE_NAMES = {"final": "final", "reference": "reference", "dummy_most_frequent": "dummy (most frequent)",
              "dummy_stratified": "dummy (stratified)"}


def heldout_model_table(r: dict, p: dict) -> pd.DataFrame:
    rows = []
    for task, t in r["tasks"].items():
        for role in final.ROLES:
            m, ci = t["metrics"][role], t["bootstrap"]["macro_f1"][role]
            name = p["tasks"][task][role]["name"] if role in ("final", "reference") else role
            rows.append(dict(task=task, model=ROLE_NAMES[role], configuration=name, macro_f1=m["macro_f1"],
                             ci_95=f"[{ci['low']:.3f}, {ci['high']:.3f}]", balanced_accuracy=m["balanced_accuracy"],
                             accuracy=m["accuracy"]))
    return pd.DataFrame(rows).set_index(["task", "model"]).round(4)


def paired_table(r: dict) -> pd.DataFrame:
    rows = []
    for task, t in r["tasks"].items():
        d = t["decision"]
        for other, pr in t["bootstrap"]["paired"].items():
            verdict = d["reference"] if other == "reference" else ("beats" if d["beats_dummies"][other] else "does not beat")
            rows.append(dict(task=task, final_minus=ROLE_NAMES[other], observed=round(pr["observed"], 4),
                             ci_95=f"[{pr['low']:.3f}, {pr['high']:.3f}]", share_le_0=pr["share_le_0"], decision=verdict,
                             task_success="yes" if d["success"] else "no"))
    return pd.DataFrame(rows).set_index(["task", "final_minus"])


def plot_test_results(r: dict, p: dict):
    """Test Macro-F1 with its bootstrap interval for every model, next to the expected range that was
    computed from fresh CV seeds before the test was read."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(TASKS), figsize=(12, 3.1))
    for ax, task in zip(axes, TASKS):
        t = r["tasks"][task]
        y = np.arange(len(final.ROLES))
        for yi, role in zip(y, final.ROLES):
            if role in ("final", "reference"):
                rng = t["expected_range"][role]
                ax.barh(yi, rng["high"] - rng["low"], left=rng["low"], height=0.5, color=RAMP[0],
                        label="expected range (fresh CV seeds)" if role == "final" else None)
            ci = t["bootstrap"]["macro_f1"][role]
            m = t["metrics"][role]["macro_f1"]
            color = BLUE if role == "final" else ORANGE if role == "reference" else MUTED
            ax.plot([ci["low"], ci["high"]], [yi, yi], color=color, linewidth=2)
            ax.scatter([m], [yi], s=50, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3)
            ax.text(ci["high"] + 0.015, yi, f"{m:.3f}", va="center", color=INK, fontsize=8.5)
        ax.set_yticks(y, [ROLE_NAMES[k] for k in final.ROLES])
        ax.invert_yaxis()
        ax.set_xlim(0, 1.05)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("test Macro-F1 (dot) with 95% bootstrap interval (line)")
        ax.set_title(f"{task}: held-out test, read once")
        ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def heldout_per_class(r: dict) -> pd.DataFrame:
    rows = []
    for task, t in r["tasks"].items():
        cis = t["bootstrap"]["per_class_f1"]["final"]
        for c in t["metrics"]["final"]["per_class"]:
            ci = cis[c["label"]]
            rows.append(dict(task=task, label=c["label"], precision=c["precision"], recall=c["recall"], f1=c["f1"],
                             ci_95=f"[{ci['low']:.3f}, {ci['high']:.3f}]", test_rows=c["support"],
                             note="too small for a claim" if c["support"] < final.SMALL_CELL else ""))
    return pd.DataFrame(rows).set_index(["task", "label"]).round(3)


def heldout_confusions(r: dict) -> dict[str, pd.DataFrame]:
    out = {}
    for task, t in r["tasks"].items():
        labels = experiments.LABELS[task]
        out[f"{task} (test, final model)"] = pd.DataFrame(t["metrics"]["final"]["confusion"], index=labels, columns=labels)
    return out


def heldout_levels(r: dict, p: dict) -> pd.DataFrame:
    rows = []
    for task, t in r["tasks"].items():
        fresh = {lv["level"]: lv for lv in p["tasks"][task]["certainty"]["fresh_levels"]}
        for lv in t["levels"]:
            rows.append(dict(task=task, level=lv["level"], test_rows=lv["n"], test_share=lv["share"],
                             test_accuracy=lv["accuracy"], fresh_seed_share=fresh[lv["level"]]["share"],
                             fresh_seed_accuracy=fresh[lv["level"]]["accuracy"]))
    return pd.DataFrame(rows).set_index(["task", "level"]).round(3)


def plot_levels(levels: pd.DataFrame):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(TASKS), figsize=(12, 3.0))
    for ax, task in zip(axes, TASKS):
        t = levels.loc[task]
        x = np.arange(len(t))
        w = 0.34
        ax.bar(x - w / 2 - 0.02, t.fresh_seed_accuracy, width=w, color=ORANGE, label="fresh-seed CV (before the test)")
        ax.bar(x + w / 2 + 0.02, t.test_accuracy, width=w, color=BLUE, label="held-out test")
        for xi, (acc, share) in enumerate(zip(t.test_accuracy, t.test_share)):
            ax.text(xi + w / 2 + 0.02, acc + 0.02, f"{acc:.2f}\n{share:.0%} of test", ha="center", color=INK, fontsize=8)
        ax.set_xticks(x, list(t.index))
        ax.set_ylim(0, 1.4)
        ax.set_ylabel("accuracy")
        ax.grid(axis="x", visible=False)
        ax.set_title(f"{task}: accuracy per certainty level")
        ax.legend(loc="upper right", fontsize=8, ncols=2)
    fig.tight_layout()
    return fig


def joint_cells(r: dict) -> pd.DataFrame:
    j = r["joint"]
    print(f"joint accuracy (both labels right): {j['accuracy']:.4f}; expected range "
          f"[{j['expected_range']['low']:.3f}, {j['expected_range']['high']:.3f}] -> {j['position']}")
    cells = pd.DataFrame(r["cells"])
    cells["note"] = np.where(cells.n < final.SMALL_CELL, "too small for a claim", "")
    table = cells.set_index(["aspect", "sentiment"]).round(3)
    return table.loc[[(a, s) for a in config.ASPECTS for s in config.SENTIMENTS if (a, s) in table.index]]


# ---- inference --------------------------------------------------------------------------------

def predict_examples(models: dict, texts) -> pd.DataFrame:
    """Score new reviews with ``predict.predict`` (the interface's input checks and certainty words)."""
    rows = []
    for t in texts:
        try:
            r = predict.predict(t, models=models)
        except predict.InvalidReview as err:
            rows.append(dict(review=t, aspect="—", aspect_certainty="—", sentiment="—", sentiment_certainty="—",
                             note=f"refused: {err.code}"))
            continue
        second = [f"{name}: {res.second_label}" for name, res in (("aspect", r.aspect), ("sentiment", r.sentiment))
                  if res.second_label]
        rows.append(dict(review=t, aspect=r.aspect.label, aspect_certainty=r.aspect.level_word,
                         sentiment=r.sentiment.label, sentiment_certainty=r.sentiment.level_word,
                         note=("runner-up " + "; ".join(second)) if second else ""))
    return pd.DataFrame(rows)
