"""PROBE (read-only): MTP layer saves under mHC (dedup / dangling-stream sanity)."""
import os, sys, warnings
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R)
warnings.simplefilter("ignore")
from cost_eval.llm_config import LLMConfig, to_dimtable
from cost_eval.build_llm import build_llm_spec
from cost_eval.shape_eval import resolve_tensor
from cost_eval.parallel_model import ParallelModel
from cost_eval.specs import ParallelConfig
MiB = 2 ** 20

cfg = LLMConfig(num_layers=2, hidden_size=4096, num_attention_heads=32, vocab_size=32000,
                seq_length=4096, batch_size=1, attn_type="gqa",
                ffn_hidden_size=8192, mtp_num_layers=1,
                residual_variant="mhc", num_residual_streams=4, use_fused_mhc=True)
spec = build_llm_spec(cfg)
pm = ParallelModel(ParallelConfig(), spec.dims.n_layers, 1)
for key, layer in spec.layer_specs.items():
    if "mtp" not in key:
        continue
    seen, tot = {}, 0.0
    for op in layer.ops:
        for s in op.saves:
            if s.name in seen:
                continue
            r = resolve_tensor(s, spec.dims, pm)
            seen[s.name] = (r.local_numel * r.dtype_bytes / MiB, op.name)
    print("layer:", key, " unique saves:", len(seen))
    for n, (mb, own) in sorted(seen.items(), key=lambda kv: -kv[1][0])[:14]:
        print("   %9.2f MiB  %-22s by=%s" % (mb, n, own))
    print("   total = %.1f MiB" % sum(v[0] for v in seen.values()))
    # dangling / producer check
    produced = {op.output.name for op in layer.ops}
    dang = sorted({t.name for op in layer.ops for t in op.inputs
                   if t.name not in produced and not t.is_weight})
    print("   inputs with no producer in-layer:", dang)
