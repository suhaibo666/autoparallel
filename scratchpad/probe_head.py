# -*- coding: utf-8 -*-
"""lm_head 段(vocab 投影)与 loss 段的**真源走查**探针。"""
import importlib.util
import sys

sys.path.insert(0, ".")
spec = importlib.util.spec_from_file_location("pc", "scratchpad/probe_components.py")
pc = importlib.util.module_from_spec(spec)
sys.argv = ["x", "none"]
spec.loader.exec_module(pc)

from cost_eval.opdag.extractor import extract_cell
from cost_eval.opdag.module_resolver import ResolvedSpec

flags = pc.cell_flags()
flags.update({
    # `Linear.__init__` 派生量(gpt_model.py:252-258 的 output_layer 构造点逐字)
    "skip_weight_param_allocation": False, "has_bias": False, "skip_add_bias": False,
    "input_size": 4096, "output_size": 129280,
})
dag = extract_cell(
    pc.MF, "pynative/layers/linear.py", "Linear",
    ResolvedSpec(cell="Linear", submodules={}), flags,
    recurse=True, subcell_specs={}, cross_file=True,
    runtime_predicates=pc.RUNTIME_PREDICATES, host_call_allow=pc.HOST_ALLOW,
    input_axes={"input_": ("seq_length", "micro_batch_size", "hidden_size")},
    strict=True)
pc.show("Linear(lm_head vocab 投影)", dag, True)
