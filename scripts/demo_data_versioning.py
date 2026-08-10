"""Demonstrate DVC data *versioning* -- the second half of requirement #3.

`dvc add` on its own only shows DVC storing a large file outside git. The brief
asks to track "datasets **and data changes**", which means showing that a past
version of the data can be recovered exactly, in step with the code that
produced it.

This script does that end to end and prints evidence at every step:

    1. tag the current raw data as `data-v1`
    2. simulate a new batch of customers arriving, producing v2
    3. `dvc add` -- watch the content hash in the .dvc pointer change
    4. commit v2 on its own branch and tag it `data-v2`
    5. show the pipeline noticing its input is stale
    6. time-travel: `git checkout data-v1 && dvc checkout`, and verify the
       file on disk is byte-for-byte the original again
    7. return to main with v1 data restored

v2 is committed on a **branch**, never on main. main keeps v1 data, so the
metrics and reports already committed there stay consistent with the data that
produced them.

Usage
-----
    python scripts/demo_data_versioning.py
    python scripts/demo_data_versioning.py --reset   # remove tags/branch, start over
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_CSV = PROJECT_ROOT / "data/raw/customer_churn_dataset-training-master.csv"
TEST_CSV = PROJECT_ROOT / "data/raw/customer_churn_dataset-testing-master.csv"
TRAIN_DVC = TRAIN_CSV.with_suffix(".csv.dvc")

TAG_V1 = "data-v1"
TAG_V2 = "data-v2"
BRANCH_V2 = "experiment/data-v2"

NEW_BATCH_ROWS = 20_000
SEED = 42


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def run(*args: str, check: bool = True) -> str:
    """Run a command in the project root and return its stdout."""
    result = subprocess.run(
        args,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"command failed: {' '.join(args)}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
    return (result.stdout or result.stderr).strip()


def step(number: str, title: str) -> None:
    print(f"\n{'=' * 74}\n  STEP {number}  {title}\n{'=' * 74}")


def fingerprint() -> dict[str, object]:
    """Content hash (from the .dvc pointer) plus what is actually on disk."""
    pointer = yaml.safe_load(TRAIN_DVC.read_text(encoding="utf-8"))["outs"][0]
    with TRAIN_CSV.open("r", encoding="utf-8") as fh:
        rows = sum(1 for _ in fh) - 1  # minus header
    return {
        "md5": pointer["md5"],
        "recorded_size": pointer["size"],
        "actual_size": TRAIN_CSV.stat().st_size,
        "rows": rows,
    }


def show(label: str, fp: dict[str, object]) -> None:
    print(
        f"  {label:<12} md5={fp['md5']}  rows={fp['rows']:,}  "
        f"bytes={fp['actual_size']:,}"
    )


def tag_exists(tag: str) -> bool:
    return bool(run("git", "tag", "--list", tag))


def current_branch() -> str:
    return run("git", "rev-parse", "--abbrev-ref", "HEAD")


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------
def reset() -> None:
    """Undo a previous demo run so it can be repeated from scratch."""
    print("Resetting the demo...")
    if current_branch() != "main":
        run("git", "checkout", "main")
    for tag in (TAG_V1, TAG_V2):
        if tag_exists(tag):
            run("git", "tag", "-d", tag)
            print(f"  deleted tag {tag}")
    if run("git", "branch", "--list", BRANCH_V2):
        run("git", "branch", "-D", BRANCH_V2)
        print(f"  deleted branch {BRANCH_V2}")
    run("dvc", "checkout", "--force")
    print("  restored v1 data via dvc checkout")
    print("\nReset complete. Run the script again without --reset.")


# ---------------------------------------------------------------------------
# the demo
# ---------------------------------------------------------------------------
def preflight() -> None:
    step("0", "Preflight")

    if not TRAIN_CSV.is_file():
        sys.exit(f"raw data missing: {TRAIN_CSV}\nRun `dvc pull` or fetch from Kaggle.")

    dirty = run("git", "status", "--porcelain")
    if dirty:
        sys.exit(
            "working tree is not clean -- commit or stash first:\n"
            + dirty
            + "\n\n(the demo checks out git tags, which needs a clean tree)"
        )

    if current_branch() != "main":
        sys.exit(f"please run this from main (currently on {current_branch()})")

    if tag_exists(TAG_V2):
        sys.exit(
            f"tag {TAG_V2} already exists -- this demo has been run before.\n"
            "Re-run with --reset to start over."
        )

    print(f"  branch:     {current_branch()}")
    print("  tree:       clean")
    print(f"  dvc:        {run('dvc', '--version')}")


def run_demo() -> None:
    # -- 1. tag the current data -------------------------------------------
    step("1", "Tag the current dataset as v1")
    v1 = fingerprint()
    show("v1", v1)
    if not tag_exists(TAG_V1):
        run("git", "tag", "-a", TAG_V1, "-m", "raw churn dataset, original Kaggle files")
    print(f"\n  tagged {TAG_V1} at {run('git', 'rev-parse', '--short', 'HEAD')}")

    # -- 2. simulate new data arriving -------------------------------------
    step("2", "A new batch of customers arrives (v2)")
    header = TRAIN_CSV.read_text(encoding="utf-8").split("\n", 1)[0].strip()
    columns = header.split(",")

    max_id = int(pd.read_csv(TRAIN_CSV, usecols=["CustomerID"])["CustomerID"].max())
    incoming = pd.read_csv(TEST_CSV).sample(n=NEW_BATCH_ROWS, random_state=SEED).copy()

    # Fresh identifiers so the batch reads as genuinely new customers rather
    # than duplicates of existing ones.
    incoming["CustomerID"] = range(max_id + 1, max_id + 1 + len(incoming))
    incoming = incoming[columns]

    # Appended as text rather than via a pandas read-modify-write. A full
    # round-trip would reformat all 440k existing rows -- the NaN row forces
    # every numeric column to float64, so integers would come back as "30.0" --
    # and the diff would look like the whole file changed rather than grew.
    with TRAIN_CSV.open("rb+") as fh:
        fh.seek(-1, 2)
        needs_newline = fh.read(1) != b"\n"
    with TRAIN_CSV.open("a", newline="", encoding="utf-8") as fh:
        if needs_newline:
            fh.write("\n")
        incoming.to_csv(fh, header=False, index=False)

    print(f"  appended {len(incoming):,} customers from a differently-distributed batch")
    print(f"  {v1['rows']:,} rows -> {fingerprint()['rows']:,} rows")
    print("  (existing rows untouched -- appended as text, not rewritten)")

    # -- 3. re-add and watch the hash change -------------------------------
    step("3", "dvc add -- the pointer's content hash changes")
    print(run("dvc", "add", str(TRAIN_CSV.relative_to(PROJECT_ROOT))))
    v2 = fingerprint()
    show("v1 (before)", v1)
    show("v2 (after)", v2)
    assert v1["md5"] != v2["md5"], "hash did not change -- dvc add failed"
    print("\n  the 23 MB file changed; git only sees a ~130-byte pointer change:")
    print(run("git", "diff", "--stat", str(TRAIN_DVC.relative_to(PROJECT_ROOT))))

    # -- 4. commit v2 on its own branch ------------------------------------
    step("4", "Commit v2 on a branch and tag it")
    run("git", "checkout", "-b", BRANCH_V2)
    run("git", "add", str(TRAIN_DVC.relative_to(PROJECT_ROOT)))
    run(
        "git",
        "commit",
        "-m",
        f"data: append a {NEW_BATCH_ROWS:,}-customer batch (v2)",
        "-m",
        "Simulates a new intake with a different behavioural profile. Committed "
        "on a branch so main keeps v1 and its matching metrics.",
    )
    run("git", "tag", "-a", TAG_V2, "-m", f"raw dataset plus {NEW_BATCH_ROWS} customers")
    run("dvc", "push")
    print(f"  committed on {BRANCH_V2}, tagged {TAG_V2}, pushed data to the DVC remote")

    # -- 5. the pipeline notices ------------------------------------------
    step("5", "The pipeline sees its input is stale")
    print(run("dvc", "status", check=False) or "  (no changes reported)")

    # -- 6. time-travel back to v1 ----------------------------------------
    step("6", "Time-travel: git checkout data-v1 && dvc checkout")
    run("git", "checkout", TAG_V1)
    print(f"  git checkout {TAG_V1}  ->  detached HEAD")
    print(run("dvc", "checkout", "--force"))

    restored = fingerprint()
    show("v1 original", v1)
    show("on disk now", restored)
    assert restored["md5"] == v1["md5"], "dvc checkout did not restore v1"
    assert restored["rows"] == v1["rows"], "row count mismatch after checkout"
    print("\n  ✓ the 23 MB file was restored byte-for-byte from the DVC cache")

    # -- 7. and forward again ---------------------------------------------
    step("7", "Forward again to v2, then home to main")
    run("git", "checkout", TAG_V2)
    run("dvc", "checkout", "--force")
    show("on disk now", fingerprint())
    print(f"  ✓ v2 restored ({NEW_BATCH_ROWS:,} extra rows back)")

    run("git", "checkout", "main")
    run("dvc", "checkout", "--force")
    final = fingerprint()
    show("back on main", final)
    assert final["md5"] == v1["md5"], "main should hold v1 data"

    # -- summary -----------------------------------------------------------
    print(f"\n{'=' * 74}\n  SUMMARY\n{'=' * 74}")
    print(f"  {TAG_V1}: {v1['rows']:>7,} rows  md5 {v1['md5']}")
    print(f"  {TAG_V2}: {v2['rows']:>7,} rows  md5 {v2['md5']}")
    print("\n  Two dataset versions, each pinned to a git ref. `git checkout <tag>`")
    print("  followed by `dvc checkout` moves code and data together.")
    print("\n  main is back on v1, so the committed metrics still match their data.")
    print(f"  Inspect v2 any time with:  git checkout {BRANCH_V2} && dvc checkout")
    print("  Undo this demo entirely:   python scripts/demo_data_versioning.py --reset")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reset", action="store_true", help="undo a previous run")
    if parser.parse_args().reset:
        reset()
        return

    preflight()
    try:
        run_demo()
    except BaseException:
        # The demo checks out git tags, so a failure partway through can leave
        # the repository on a detached HEAD with the wrong data on disk. Always
        # put it back before surfacing the error.
        print("\n" + "!" * 74)
        print("  demo failed -- restoring the repository to main with v1 data")
        print("!" * 74)
        run("git", "checkout", "main", check=False)
        run("dvc", "checkout", "--force", check=False)
        raise


if __name__ == "__main__":
    main()
