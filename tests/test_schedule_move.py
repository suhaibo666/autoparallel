"""schedule.py 搬家（spec §2.2 契约2）：调度代数移入中立模块，mem_timeline re-export 兼容。
验收 = ①新模块可独立 import 且不依赖任何内存模块;②旧 import 路径同一对象(is);③行为逐值一致。"""
import ast
import inspect

# 搬家名单（8 个定义全量，含私有 _1f1b_from_warmup——它也在 mem_timeline 的 re-export
# 元组里，删掉即静默破坏兼容契约，故必须一并盯住）。两个测试共用，单点维护。
_MOVED_NAMES = ("Event", "_1f1b_from_warmup", "build_1f1b", "interleaved_warmup",
                "build_interleaved_1f1b", "get_schedule_table",
                "interleaved_virtual_order", "chunk_layer_ids")


def test_schedule_module_standalone():
    import cost_eval.schedule as sch
    for name in _MOVED_NAMES:
        assert hasattr(sch, name), name


def test_schedule_has_no_mem_imports():
    import cost_eval.schedule as sch
    src = inspect.getsource(sch)
    names = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            for a in node.names:
                names.update(a.name.split("."))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.update(node.module.split("."))
    assert not names & {"mem_timeline", "structure_mem", "static_mem"}


def test_mem_timeline_reexports_same_objects():
    import cost_eval.schedule as sch
    import cost_eval.mem_timeline as mt
    for name in _MOVED_NAMES:
        assert getattr(mt, name) is getattr(sch, name), name


def test_schedule_behavior_spotcheck():
    from cost_eval.schedule import build_1f1b, interleaved_virtual_order
    evs = build_1f1b(stage=0, pp=2, m=4)          # warmup=1
    kinds = [(e.kind, e.mb) for e in evs]
    assert kinds == [("FWD", 0), ("FWD", 1), ("BWD", 0), ("FWD", 2), ("BWD", 1),
                     ("FWD", 3), ("BWD", 2), ("BWD", 3)]
    order = interleaved_virtual_order(stage=0, pp=2, m=4, v=2, group_size=2)
    assert len(order) == 2 * 4 * 2 and order[0][0] == "FWD"
