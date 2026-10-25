"""Command line: ``python -m arabic_reviews <step>``."""

from __future__ import annotations

import argparse
import json

import pandas as pd

from . import config


def cmd_ingest() -> None:
    from .ingest import ingest

    manifest = ingest()
    print(f"verified {manifest['filename']} sha256={manifest['sha256']} -> {config.MANIFEST_FILE.name}")


def cmd_validate() -> None:
    from .ingest import verify_file
    from .reports import write_outputs
    from .validate import run_validation

    fp = verify_file(config.RAW_FILE, config.SOURCE)
    raw = pd.read_excel(config.RAW_FILE, sheet_name=0, dtype=object, engine="openpyxl")
    result = run_validation(raw)
    hashes = write_outputs(result, fp["sha256"])
    print(json.dumps({"lineage": result.lineage["excluded_by_reason"], "sha256": hashes}, indent=1))


def cmd_split() -> None:
    import hashlib

    from .reports import write_split_outputs
    from .split import make_split

    data = pd.read_csv(config.VALIDATED_FILE, dtype={"text": str}, keep_default_na=False)
    validated_sha = hashlib.sha256(config.VALIDATED_FILE.read_bytes()).hexdigest()
    hashes = write_split_outputs(data, make_split(data), validated_sha)
    print(json.dumps({"sha256": hashes}, indent=1))


def cmd_eda() -> None:
    from .eda import run_eda

    run_eda()
    print(f"wrote {config.EDA_REPORT.relative_to(config.ROOT)} and {config.FIGURES_DIR.relative_to(config.ROOT)}/")


def cmd_experiments() -> None:
    from .experiments import (ABLATIONS_FILE, AXIS1_CHAIN, TASKS, axes_markdown, axis_markdown,
                              read_log, resolve_axes, run_pending)
    from .split import load_dev

    dev = load_dev()
    before = len(read_log())
    run_pending(dev)
    results = {task: resolve_axes(task, dev) for task in TASKS}
    ABLATIONS_FILE.write_text(
        "# Feature-engineering ablations (development CV only)\n\n"
        + axis_markdown(read_log(), "Axis 1 - representation", AXIS1_CHAIN) + "\n"
        + axes_markdown(results) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(f"{len(read_log()) - before} experiment(s) appended; {len(read_log())} rows in reports/experiments.csv")


def cmd_models() -> None:
    from .experiments import (ERRORS_FILE, MODELS_FILE, TASKS, error_markdown, m4_markdown, oof_predictions,
                              read_log, resolve_m4)
    from .split import load_dev

    dev = load_dev()
    before = len(read_log())
    results = {task: resolve_m4(task, dev) for task in TASKS}
    MODELS_FILE.write_text(m4_markdown(results), encoding="utf-8", newline="\n")
    scratch = config.ROOT / ".local" / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    parts = ["# Out-of-fold error analysis (development partition)\n"]
    for task, r in results.items():
        oof = oof_predictions(r["final"], dev, task)
        parts.append(error_markdown(task, r["final"], oof))
        oof.merge(dev[["rid", "text"]], on="rid").query("y != pred").to_csv(
            scratch / f"oof_errors_{task}.csv", index=False, encoding="utf-8")
    ERRORS_FILE.write_text("\n".join(parts), encoding="utf-8", newline="\n")
    print(f"{len(read_log()) - before} experiment(s) appended; wrote {MODELS_FILE.name} and {ERRORS_FILE.name}")


COMMANDS = {
    "ingest": cmd_ingest, "validate": cmd_validate, "split": cmd_split, "eda": cmd_eda,
    "experiments": cmd_experiments, "models": cmd_models,
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="arabic_reviews")
    parser.add_argument("step", choices=sorted(COMMANDS))
    COMMANDS[parser.parse_args(argv).step]()


if __name__ == "__main__":
    main()
