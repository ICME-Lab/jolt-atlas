#!/usr/bin/env python3
"""Validate an immutable Atlas source archive on the retained Linux host."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="+")
    parser.add_argument("--toolchain", default="1.95.0")
    parser.add_argument("--pass-name", default="initial")
    parser.add_argument("--integration-only", action="store_true")
    parser.add_argument("--ordinary-check-only", action="store_true")
    parser.add_argument("--workspace-checks", action="store_true")
    parser.add_argument("--core-tests-only", action="store_true")
    parser.add_argument("--ordinary-backends-only", action="store_true")
    parser.add_argument("--fixed-tables-only", action="store_true")
    args = parser.parse_args()
    root = Path("/root/atlas-consolidation")
    env = os.environ.copy()
    env.update(PATH="/root/.cargo/bin:" + env["PATH"],
               CARGO_TARGET_DIR="/root/atlas-consolidation/target",
               RUSTFLAGS="-D warnings", RAYON_NUM_THREADS="8",
               CARGO_BUILD_JOBS="16", CARGO_PROFILE_DEV_DEBUG="0",
               CARGO_PROFILE_TEST_DEBUG="0", RUST_TEST_THREADS="4")
    for name in args.names:
        lock = (root / "cargo-validation.lock").open("a")
        fcntl.flock(lock, fcntl.LOCK_EX)
        archive = root / f"{name}.tar.gz"
        source = root / f"{name}-{args.pass_name}"
        source.mkdir(exist_ok=False)
        with tarfile.open(archive) as bundle:
            bundle.extractall(source)
        patch = root / f"{name}.patch"
        if patch.stat().st_size:
            subprocess.run(["git", "apply", str(patch)], cwd=source, check=True)
        manifest = json.loads((root / f"{name}.manifest.json").read_text())
        for filename, expected in manifest["files"].items():
            assert hashlib.sha256((source / filename).read_bytes()).hexdigest() == expected, filename
        logs = root / "logs" / f"{name}-{args.pass_name}"
        logs.mkdir(exist_ok=False)
        record = dict(archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                      patch_sha256=hashlib.sha256(patch.read_bytes()).hexdigest(),
                      source_head=manifest["head"], authenticated_files=len(manifest["files"]),
                      source=str(source), workers=8, toolchain=args.toolchain,
                      environment={k: env[k] for k in ["RUSTFLAGS", "CARGO_TARGET_DIR", "RAYON_NUM_THREADS"]},
                      commands=[])
        clean = ["cargo", f"+{args.toolchain}", "clean"]
        for package in ["common", "atlas-onnx-tracer", "joltworks", "jolt-atlas-core", "jolt-atlas"]:
            clean += ["-p", package]
        with (logs / "clean.log").open("w") as output:
            subprocess.run(clean, cwd=source, env=env, stdout=output,
                           stderr=subprocess.STDOUT, check=True)
        record["local_crates_cleaned"] = True
        commands = [["fmt", "--all", "--check"]]
        if name == "baseline":
            commands = [["check", "-p", "jolt-atlas-core", "--all-targets", "--features", "zk"]]
        elif name == "correctness":
            commands.extend([
                ["clippy", "--workspace", "--all-targets"],
                ["check", "-p", "jolt-atlas-core", "--all-targets", "--features", "zk"],
                ["clippy", "-p", "jolt-atlas-core", "--all-targets", "--features", "zk"],
                ["test", "--workspace", "--lib"],
                ["test", "-p", "jolt-atlas-core", "--lib", "--features", "zk"],
            ])
        elif name == "ordinary":
            for features in [[], ["--no-default-features"],
                             ["--no-default-features", "--features", "zk"]]:
                commands.append(["test", "-p", "joltworks", "--lib"] + features)
                commands.append(["test", "-p", "jolt-atlas-core", "--lib"] + features)
            commands.append(["test", "-p", "atlas-onnx-tracer", "--no-default-features", "--lib"])
            commands.extend([
                ["clippy", "--workspace", "--all-targets"],
                ["clippy", "-p", "jolt-atlas-core", "--all-targets", "--features", "zk"],
            ])
        else:
            commands.extend([
                ["test", "-p", "joltworks", "--lib"],
                ["test", "-p", "joltworks", "--lib", "--features", "zk"],
                ["clippy", "-p", "joltworks", "--all-targets", "--features", "zk"],
            ])
        if args.integration_only:
            commands = [["check", "-p", "jolt-atlas-core", "--all-targets", "--features", "zk"],
                        ["clippy", "--workspace", "--all-targets"],
                        ["clippy", "--workspace", "--all-targets", "--features", "jolt-atlas-core/zk"]]
        if args.ordinary_check_only:
            commands = [["fmt", "--all", "--check"],
                        ["check", "-p", "jolt-atlas-core", "--no-default-features", "--lib"],
                        ["check", "-p", "jolt-atlas-core", "--no-default-features", "--tests"],
                        ["check", "-p", "joltworks", "--no-default-features", "--features", "zk", "--tests"],
                        ["clippy", "--workspace", "--all-targets"]]
        if args.workspace_checks:
            commands.extend([["clippy", "--workspace", "--all-targets"],
                             ["clippy", "--workspace", "--all-targets", "--features", "jolt-atlas-core/zk"]])
        if args.core_tests_only:
            commands = [["test", "-p", "jolt-atlas-core", "--lib", "--features", "zk"]]
        if args.ordinary_backends_only:
            commands = [["fmt", "--all", "--check"],
                        ["check", "-p", "jolt-atlas-core", "--no-default-features", "--lib"],
                        ["clippy", "--workspace", "--all-targets"],
                        ["test", "-p", "joltworks", "--lib", "--no-default-features", "--features", "affine-msm"],
                        ["clippy", "-p", "jolt-atlas-core", "--lib", "--no-default-features", "--features", "affine-msm,check-fixed-tables"]]
        if args.fixed_tables_only:
            commands = [["test", "-p", "jolt-atlas-core", "--lib", "--no-default-features", "--features", "affine-msm,check-fixed-tables", "fixed_tables_match_every_reference_entry"]]
        for i, command in enumerate(commands):
            cmd = ["cargo", f"+{args.toolchain}"] + command
            started = time.time()
            with (logs / f"{i:02d}.log").open("w") as output:
                try:
                    result = subprocess.run(cmd, cwd=source, env=env, stdout=output,
                                            stderr=subprocess.STDOUT, timeout=2700)
                    code = result.returncode
                except subprocess.TimeoutExpired:
                    code = 124
            record["commands"].append(dict(command=cmd, exit_code=code,
                                           seconds=time.time() - started,
                                           log=f"{i:02d}.log"))
            (logs / "summary.json").write_text(json.dumps(record, indent=2) + "\n")
            print(name, i, code, round(time.time() - started, 2), flush=True)
        lock.close()


if __name__ == "__main__":
    main()
