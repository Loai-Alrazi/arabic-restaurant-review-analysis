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
    from .experiments import (ERROR_CATEGORIES_FILE, ERRORS_FILE, MODELS_FILE, TASKS, categories_markdown,
                              error_markdown, m4_markdown, oof_predictions, read_log, resolve_m4)
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
    cats = pd.read_csv(ERROR_CATEGORIES_FILE, keep_default_na=False)
    parts.append(categories_markdown(cats, {task: r["final"]["name"] for task, r in results.items()}))
    ERRORS_FILE.write_text("\n".join(parts), encoding="utf-8", newline="\n")
    print(f"{len(read_log()) - before} experiment(s) appended; wrote {MODELS_FILE.name} and {ERRORS_FILE.name}")


def cmd_protocol() -> None:
    from .protocol import PROTOCOL_JSON, PROTOCOL_MD, build_protocol, write_protocol

    protocol = build_protocol()
    write_protocol(protocol)
    print(f"fingerprint {protocol['fingerprint']}; wrote {PROTOCOL_MD.name} and {PROTOCOL_JSON.name}")


def cmd_final(rerun_reason: str | None = None, check_only: bool = False) -> None:
    from .final import run_final

    r = run_final(rerun_reason=rerun_reason, check_only=check_only)
    if check_only:
        print(f"fingerprint {r['fingerprint']} matches the frozen protocol; models fit; test partition not read")
        state = {None: "stored for later comparisons", True: "identical to the stored ones",
                 False: "DIFFERENT from the stored ones (library or code change)"}[r["digests_unchanged"]]
        print(f"development-probability digests of the saved models: {state}")
        return
    for task, t in r["tasks"].items():
        print(f"{task}: test Macro-F1 {t['metrics']['final']['macro_f1']:.4f} ({t['position']['final']} the expected "
              f"range); versus the reference: {t['decision']['reference']}; success: {t['decision']['success']}")
    print(f"joint accuracy {r['joint']['accuracy']:.4f}; run {r['access']['run']} logged in reports/test_access_log.json")


def cmd_v2_experiments() -> None:
    from .v2 import RESULTS_MD, run_experiments

    r = run_experiments()
    print(f"selection: {r['selection']['chosen']}; adopted: {r['adopted']}; released sentiment configuration: "
          f"{r['released']}; wrote {RESULTS_MD.relative_to(config.ROOT).as_posix()}")


def cmd_v2_build(verify: bool = False) -> None:
    from .v2 import CARD_FILE, PROTOCOL_FILE
    from .v2_release import build

    r = build(verify=verify)
    if verify:
        print("recomputed results equal reports/v2/results.json")
    print(f"version-2 fingerprint {r['fingerprint']}; sentiment configuration {r['released']}; wrote "
          f"{PROTOCOL_FILE.relative_to(config.ROOT).as_posix()}, {CARD_FILE.relative_to(config.ROOT).as_posix()} "
          "and models/v2/*.joblib (local); the held-out test was not read")


COMMANDS = {
    "ingest": cmd_ingest, "validate": cmd_validate, "split": cmd_split, "eda": cmd_eda,
    "experiments": cmd_experiments, "models": cmd_models, "protocol": cmd_protocol, "final": cmd_final,
    "v2-experiments": cmd_v2_experiments, "v2-build": cmd_v2_build,
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="arabic_reviews")
    parser.add_argument("step", choices=sorted(COMMANDS))
    parser.add_argument("--check", action="store_true",
                        help="final: verify the protocol and fit the models without reading the test partition")
    parser.add_argument("--rerun-reason", help="final: methodological reason for reading the test partition again")
    parser.add_argument("--verify", action="store_true",
                        help="v2-build: recompute every version-2 result first and require it to equal results.json")
    args = parser.parse_args(argv)
    if args.verify and args.step != "v2-build":
        parser.error("--verify applies to the v2-build step only")
    if args.step == "final" and args.check and args.rerun_reason is not None:
        parser.error("--check never reads the test partition, so it takes no --rerun-reason")
    if args.step == "final":
        cmd_final(args.rerun_reason, args.check)
    elif args.step == "v2-build":
        if args.check or args.rerun_reason is not None:
            parser.error("--check and --rerun-reason apply to the final step only")
        cmd_v2_build(args.verify)
    elif args.check or args.rerun_reason is not None:
        parser.error("--check and --rerun-reason apply to the final step only")
    else:
        COMMANDS[args.step]()


if __name__ == "__main__":
    main()
