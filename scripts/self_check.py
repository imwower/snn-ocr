#!/usr/bin/env python3
"""Repository self-check utility for snn_ocr.

This script validates symbol availability/signatures, runs smoke tests, and
flags non-standard imports. Missing items are written to missing.json so
contributors can react quickly.
"""
from __future__ import annotations

import ast
import importlib
import inspect
import json
import math
import pkgutil
import sys
import sysconfig
from dataclasses import dataclass
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory
from typing import Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = REPO_ROOT / "snn_ocr"
MISSING_PATH = REPO_ROOT / "missing.json"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

@dataclass(frozen=True)
class SymbolSpec:
    name: str
    params: int | None = None


MODULE_MATRIX: Sequence[Tuple[str, Sequence[SymbolSpec]]] = (
    ("bitfont", (SymbolSpec("get_bitmap", 2),)),
    ("pgm", (SymbolSpec("save_pgm", 2),)),
    ("render", (SymbolSpec("render_text", 13),)),
    ("synth", (SymbolSpec("generate_dataset", 5),)),
    (
        "spikes",
        (
            SymbolSpec("encode_ttfs", 5),
            SymbolSpec("encode_poisson", 6),
        ),
    ),
    (
        "lif",
        (
            SymbolSpec("LIF", 4),
            SymbolSpec("conv2d_spike", 6),
            SymbolSpec("conv1d_spike", 6),
        ),
    ),
    ("otc", (SymbolSpec("compress_height", 6),)),
    ("seq", (SymbolSpec("dwconv1d_spike", 2),)),
    (
        "ctc",
        (
            SymbolSpec("ctc_loss", 3),
            SymbolSpec("beam_search", 5),
        ),
    ),
    ("train", (SymbolSpec("run_epoch", 19),)),
    (
        "eval",
        (
            SymbolSpec("cer", 2),
            SymbolSpec("wer", 2),
        ),
    ),
    ("cli", (SymbolSpec("main", 1),)),
    ("utils", ()),
)


def build_stdlib_index() -> set[str]:
    """Best-effort list of standard library modules (top-level names only)."""
    modules = set(sys.builtin_module_names)
    stdlib_names = getattr(sys, "stdlib_module_names", None)
    if stdlib_names is not None:
        modules |= stdlib_names
    else:
        stdlib_path = sysconfig.get_path("stdlib")
        if stdlib_path:
            for _, name, _ in pkgutil.walk_packages([stdlib_path]):
                modules.add(name.split(".", 1)[0])
    modules.add("__future__")
    return modules


STD_LIB_MODULES = build_stdlib_index()


def is_stdlib_or_local(module_name: str) -> bool:
    """Return True when module_name is part of stdlib or snn_ocr package."""
    if not module_name:
        return True
    root = module_name.split(".", 1)[0]
    if root == "snn_ocr":
        return True
    return root in STD_LIB_MODULES


def scan_imports() -> List[Tuple[str, str]]:
    """Collect non-standard imports within snn_ocr/ for enforcement."""
    offenders: List[Tuple[str, str]] = []
    for py_file in PKG_ROOT.rglob("*.py"):
        try:
            source = py_file.read_text(encoding="utf-8")
        except FileNotFoundError:
            continue
        try:
            tree = ast.parse(source, filename=str(py_file))
        except SyntaxError as exc:
            offenders.append(
                (f"syntax-error:{py_file.name}", f"{py_file.relative_to(REPO_ROOT)}:{exc.lineno}")
            )
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if not is_stdlib_or_local(alias.name):
                        offenders.append(
                            (
                                alias.name,
                                f"{py_file.relative_to(REPO_ROOT)}:{node.lineno}",
                            )
                        )
            elif isinstance(node, ast.ImportFrom):
                if node.level and not node.module:
                    # Relative import such as "from . import x" -> always local.
                    continue
                if node.level:
                    continue
                module_name = node.module or ""
                if module_name.startswith("snn_ocr"):
                    continue
                if not is_stdlib_or_local(module_name):
                    offenders.append(
                        (
                            module_name,
                            f"{py_file.relative_to(REPO_ROOT)}:{node.lineno}",
                        )
                    )
    # Deduplicate while preserving order
    unique: Dict[Tuple[str, str], None] = {}
    for offender in offenders:
        unique.setdefault(offender, None)
    return list(unique.keys())


def _param_count(obj) -> int | None:
    if obj is None:
        return None
    if not callable(obj):
        return None
    try:
        signature = inspect.signature(obj)
    except (TypeError, ValueError):
        return None
    count = 0
    for param in signature.parameters.values():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        count += 1
    return count


def check_modules() -> Tuple[
    List[Tuple[str, bool, List[Dict[str, object]]]],
    List[Dict[str, str]],
]:
    """Attempt to import every matrix entry and capture missing symbols/signatures."""
    results: List[Tuple[str, bool, List[Dict[str, object]]]] = []
    missing: List[Dict[str, str]] = []
    for module, symbols in MODULE_MATRIX:
        dotted = f"snn_ocr.{module}"
        try:
            imported = importlib.import_module(dotted)
            module_ok = True
            mod_error = ""
        except Exception as exc:  # pragma: no cover - surfaced via TODO log
            imported = None
            module_ok = False
            mod_error = str(exc)
            missing.append({"module": module, "symbol": "", "reason": mod_error})
        symbol_results: List[Dict[str, object]] = []
        for spec in symbols:
            present = bool(module_ok and hasattr(imported, spec.name))
            signature_ok = True
            actual_params = None
            reason = None
            if present and spec.params is not None:
                actual_params = _param_count(getattr(imported, spec.name, None))
                signature_ok = actual_params == spec.params
                if not signature_ok:
                    reason = f"signature mismatch expected {spec.params}, got {actual_params}"
            elif not present:
                reason = "symbol missing" if module_ok else "module import failed"
            if not present or not signature_ok:
                missing.append({
                    "module": module,
                    "symbol": spec.name,
                    "reason": reason or "unknown",
                })
            symbol_results.append(
                {
                    "name": spec.name,
                    "present": present,
                    "signature_ok": signature_ok,
                    "expected": spec.params,
                    "actual": actual_params,
                }
            )
        results.append((module, module_ok, symbol_results))
    return results, missing


def write_missing(missing: Sequence[Dict[str, str]]) -> None:
    """Persist missing symbol report for downstream tooling."""
    with MISSING_PATH.open("w", encoding="utf-8") as handle:
        json.dump(list(missing), handle, ensure_ascii=False, indent=2)


def print_matrix(matrix: Sequence[Tuple[str, bool, Sequence[Dict[str, object]]]]) -> None:
    """Render a simple ASCII table describing symbol availability."""
    print("模块存在矩阵：")
    header = f"{'模块':<10} {'可导入':<6} 缺失/异常"
    print(header)
    print("-" * len(header))
    for module, module_ok, symbol_results in matrix:
        issues: List[str] = []
        if not module_ok:
            issues.append("<module>")
        for result in symbol_results:
            if not result["present"]:
                issues.append(result["name"])
            elif not result["signature_ok"]:
                expected = result.get("expected")
                actual = result.get("actual")
                issues.append(f"{result['name']}(sig {expected}->{actual})")
        missing_col = ", ".join(issues) if issues else "-"
        status = "OK" if module_ok else "NO"
        print(f"{module:<10} {status:<6} {missing_col}")


def print_todos(missing: Sequence[Dict[str, str]]) -> None:
    todos = [item for item in missing if item.get("symbol")]
    if not todos:
        return
    print("TODO 提示：")
    for item in todos:
        symbol = item["symbol"]
        module = item["module"]
        print(f"TODO: 实现 {module}.{symbol}")


def _smoke_ctc() -> List[Dict[str, str]]:
    failures: List[Dict[str, str]] = []
    try:
        from snn_ocr import ctc
    except Exception as exc:  # pragma: no cover - import failure recorded
        failures.append({"module": "ctc", "symbol": "smoke:import", "reason": str(exc)})
        return failures

    empty_target_logits = [
        [3.5, 0.1, 0.2],
        [3.0, 0.5, 0.1],
        [2.8, 0.2, 0.2],
    ]
    repeated_logits = [
        [0.2, 2.5, 0.1],
        [0.1, 2.2, 0.3],
        [2.4, 0.2, 0.4],
        [0.2, 2.4, 0.3],
    ]
    repeated_target = [1, 1]
    rng = Random(13)
    rng_logits: List[List[float]] = []
    for _ in range(128):
        frame = [rng.uniform(-0.5, 0.5) for _ in range(5)]
        frame[0] += 0.3
        rng_logits.append(frame)
    rng_target = [1, 2, 3]
    cases = [
        {
            "name": "empty-target",
            "logits": empty_target_logits,
            "target": [],
            "expect_greedy": "",
            "beam_opts": {"beam": 2},
        },
        {
            "name": "double-letter",
            "logits": repeated_logits,
            "target": repeated_target,
            "expect_greedy": "AA",
            "expect_beam": "AA",
            "beam_opts": {"beam": 5},
        },
        {
            "name": "random-noise",
            "logits": rng_logits,
            "target": rng_target,
            "min_beam_len": len(rng_target),
            "beam_opts": {"beam": 8, "len_norm": True, "ins_penalty": 0.05},
        },
    ]

    for case in cases:
        name = case["name"]
        try:
            logits = case["logits"]
            target = case["target"]
            loss = ctc.ctc_loss(logits, target, blank=0)
            if not math.isfinite(loss) or loss < 0.0:
                raise AssertionError("loss not finite")
            _, grad = ctc.ctc_loss_with_grad(logits, target, blank=0)
            if len(grad) != len(logits):
                raise AssertionError("grad length mismatch")
            greedy = ctc.greedy_decode(logits, blank=0)
            beam = ctc.beam_search(logits, blank=0, **case.get("beam_opts", {}))
            if case.get("expect_greedy") is not None and greedy != case["expect_greedy"]:
                raise AssertionError(f"greedy '{greedy}' != '{case['expect_greedy']}'")
            if case.get("expect_beam") is not None and beam != case["expect_beam"]:
                raise AssertionError(f"beam '{beam}' != '{case['expect_beam']}'")
            if case.get("min_beam_len") is not None and len(beam) < case["min_beam_len"]:
                raise AssertionError("beam result too short")
        except Exception as exc:  # pragma: no cover - recorded via missing.json
            failures.append({"module": "ctc", "symbol": f"smoke:{name}", "reason": str(exc)})
    return failures


def _smoke_lif() -> List[Dict[str, str]]:
    failures: List[Dict[str, str]] = []
    try:
        from snn_ocr import lif as lif_mod
    except Exception as exc:  # pragma: no cover - import failure recorded
        failures.append({"module": "lif", "symbol": "smoke:import", "reason": str(exc)})
        return failures
    try:
        neuron = lif_mod.LIF(alpha=0.9, v_th=1.0)
        spikes_high = []
        for _ in range(12):
            _, spike = neuron.step(0.8)
            spikes_high.append(spike)
        rate_high = sum(spikes_high) / len(spikes_high)
        if rate_high <= 0.5:
            raise AssertionError("high-current spike rate too low")
        neuron.reset()
        spikes_low = []
        for _ in range(12):
            _, spike = neuron.step(0.1)
            spikes_low.append(spike)
        if any(spikes_low):
            raise AssertionError("low-current should not spike")
    except Exception as exc:  # pragma: no cover - recorded via missing.json
        failures.append({"module": "lif", "symbol": "smoke:lif-rate", "reason": str(exc)})
    return failures


def _smoke_pgm() -> List[Dict[str, str]]:
    failures: List[Dict[str, str]] = []
    try:
        from snn_ocr import pgm as pgm_mod
    except Exception as exc:  # pragma: no cover - import failure recorded
        failures.append({"module": "pgm", "symbol": "smoke:import", "reason": str(exc)})
        return failures
    pattern = [
        [0, 255, 0, 255],
        [255, 0, 255, 0],
    ]
    try:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp) / "smoke.pgm"
            pgm_mod.save_pgm(tmp_path, pattern)
            loaded = pgm_mod.load_pgm(tmp_path)
            if loaded != pattern:
                raise AssertionError("PGM roundtrip mismatch")
    except Exception as exc:  # pragma: no cover - recorded via missing.json
        failures.append({"module": "pgm", "symbol": "smoke:roundtrip", "reason": str(exc)})
    return failures


def run_smoke_tests() -> List[Dict[str, str]]:
    failures: List[Dict[str, str]] = []
    failures.extend(_smoke_ctc())
    failures.extend(_smoke_lif())
    failures.extend(_smoke_pgm())
    return failures


def main() -> int:
    matrix, missing = check_modules()
    print_matrix(matrix)

    smoke_failures = run_smoke_tests()
    if smoke_failures:
        print(f"Smoke 测试：失败 {len(smoke_failures)} 项")
    else:
        print("Smoke 测试：通过")
    missing.extend(smoke_failures)

    offenders = scan_imports()
    if offenders:
        print(f"非标准库依赖 {len(offenders)} 条")
        for name, location in offenders:
            print(f"- {name} @ {location}")
            missing.append({"module": "imports", "symbol": name, "reason": location})
    else:
        print("非标准库依赖 0 条")

    print_todos(missing)
    write_missing(missing)
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
