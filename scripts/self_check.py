#!/usr/bin/env python3
"""Repository self-check utility for snn_ocr.

This script validates that snn_ocr only relies on the Python standard library,
and that the key public entry points remain importable. Missing symbols are
reported in both stdout and missing.json so contributors can track TODOs.
"""
from __future__ import annotations

import ast
import importlib
import json
import pkgutil
import sys
import sysconfig
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = REPO_ROOT / "snn_ocr"
MISSING_PATH = REPO_ROOT / "missing.json"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODULE_MATRIX: Sequence[Tuple[str, Sequence[str]]] = (
    ("bitfont", ("get_bitmap",)),
    ("pgm", ("save_pgm",)),
    ("render", ("render_text",)),
    ("synth", ("generate_dataset",)),
    ("spikes", ("encode_ttfs", "encode_poisson")),
    ("lif", ("LIF", "conv2d_spike", "conv1d_spike")),
    ("otc", ("compress_height",)),
    ("seq", ("dwconv1d_spike",)),
    ("ctc", ("ctc_loss", "beam_search")),
    ("train", ("run_epoch",)),
    ("eval", ("cer", "wer")),
    ("cli", ("main",)),
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


def check_modules() -> Tuple[
    List[Tuple[str, bool, List[Tuple[str, bool]]]], List[Dict[str, str]]
]:
    """Attempt to import every matrix entry and capture missing symbols."""
    results: List[Tuple[str, bool, List[Tuple[str, bool]]]] = []
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
        symbol_results: List[Tuple[str, bool]] = []
        for symbol in symbols:
            symbol_ok = bool(module_ok and hasattr(imported, symbol))
            symbol_results.append((symbol, symbol_ok))
            if not symbol_ok:
                reason = "module import failed" if not module_ok else "not defined"
                missing.append({"module": module, "symbol": symbol, "reason": reason})
        results.append((module, module_ok, symbol_results))
    return results, missing


def write_missing(missing: Sequence[Dict[str, str]]) -> None:
    """Persist missing symbol report for downstream tooling."""
    with MISSING_PATH.open("w", encoding="utf-8") as handle:
        json.dump(list(missing), handle, ensure_ascii=False, indent=2)


def print_matrix(matrix: Sequence[Tuple[str, bool, Sequence[Tuple[str, bool]]]]) -> None:
    """Render a simple ASCII table describing symbol availability."""
    print("模块存在矩阵：")
    header = f"{'模块':<10} {'可导入':<6} 缺失符号"
    print(header)
    print("-" * len(header))
    for module, module_ok, symbol_results in matrix:
        missing_symbols = []
        if not module_ok:
            missing_symbols.append("<module>")
        missing_symbols.extend(
            symbol for symbol, ok in symbol_results if not ok
        )
        missing_col = ", ".join(missing_symbols) if missing_symbols else "-"
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


def main() -> int:
    matrix, missing = check_modules()
    print_matrix(matrix)
    write_missing(missing)
    print_todos(missing)
    offenders = scan_imports()
    if offenders:
        print(f"第三方依赖扫描：{len(offenders)} 条")
        for name, location in offenders:
            print(f"- {name} @ {location}")
        return 1
    print("第三方依赖扫描：0 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
