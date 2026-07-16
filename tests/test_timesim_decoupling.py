# tests/test_timesim_decoupling.py
"""解耦契约 §2.2-1（进回归门）：cost_eval/timesim/** 与内存仿真模块双向 import 禁令。
共享白名单只有 specs / schedule / opdag（上游 IR）——契约 §2.2-2。"""
import ast
import glob
import os

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cost_eval")
MEM_MODULES = {"mem_timeline", "structure_mem", "static_mem"}


def _imported_names(src: str) -> set[str]:
    """import 语句涉及的全部名字段集合（模块点分段 + from-import 的符号名）。"""
    toks: set[str] = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            for a in node.names:
                toks.update(a.name.split("."))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                toks.update(node.module.split("."))
            for a in node.names:
                toks.add(a.name)
    return toks


def _read(p):
    with open(p, "r", encoding="utf-8") as fh:
        return fh.read()


def test_linter_helper_catches_violations():
    """linter 自检：三种走私写法都要被抓到。"""
    assert "mem_timeline" in _imported_names("from ..mem_timeline import build_1f1b")
    assert "mem_timeline" in _imported_names("from cost_eval import mem_timeline")
    assert "mem_timeline" in _imported_names("import cost_eval.mem_timeline as mt")


def test_timesim_never_imports_mem_simulator():
    files = glob.glob(os.path.join(ROOT, "timesim", "**", "*.py"), recursive=True)
    assert files, "timesim 包不存在？"
    for p in files:
        bad = _imported_names(_read(p)) & MEM_MODULES
        assert not bad, f"{p} 走私 import 内存模块: {bad}"


def test_mem_simulator_never_imports_timesim():
    for mod in MEM_MODULES:
        p = os.path.join(ROOT, f"{mod}.py")
        assert "timesim" not in _imported_names(_read(p)), f"{p} 反向 import timesim"
