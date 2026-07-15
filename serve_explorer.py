"""交互式内存实验台 v2（本地服务，stdlib 零依赖）→ http://127.0.0.1:8765

v2（2026-07-11，修用户四条）:
① 模型结构可配:attn(mla/gqa/mha)、层数、dense:MoE 比例(first_k_dense)、专家数/topk/头数/seq——
  走 `LLMConfig → build_llm_spec`（Tier-1: mha/gqa/mla × dense/moe）,不再锁死 DSv3 preset;
② 并行度自由输入（数字框,无上限）+ **合法性校验**（tp|heads、ep|experts、ep|dp·cp·tp、S|cp、pp≤层数…）,
  非法给中文明确报错;
③ 模型结构 op-DAG 图回归:左侧竖向逐层图,**由仿真器实际计算用的 resolved op 图生成**（tp/ep/cp 切分后
  字节,配置一改图上字节即变;静态 mindformers 提取版仍作交叉验证）,层组可展开/折叠;
④ memory timeline 回归为**按时间顺序的堆叠面积图**（可点,点到事件→右侧各桶）。PP 每 stage 标签页。
"""
import dataclasses
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

sys.stdout.reconfigure(encoding="utf-8")
from cost_eval.presets import deepseek_v3, deepseek_v4
from cost_eval.build_llm import build_llm_spec
from cost_eval.structure_mem import estimate_structure_memory, estimate_select_memory
from cost_eval.specs import ParallelConfig, OptimizerSpec, HardwareSpec, RecomputeSpec, SwapSpec
from cost_eval.report import Evaluator

MiB = 2 ** 20
BK = ["persistent", "act_live", "kept_frag", "gather_buf", "grad_buf", "recomp_scratch",
      "bwd_scratch", "bwd_working_set", "swap_buf", "workspace", "optstep", "framework"]
_CP_METHODS = ("colossal", "ulysses", "ring", "hybrid")
# select 选择器:**单一来源** = 转换器 _SELECT_MODULE_OPS（2026-07-14 review P1.5:此前双维护
# 导致口径漂移——serve 多 "qkv" 而转换器没有,GQA yaml select 静默漏选）。
from cost_eval.configs.from_mindformers import _SELECT_MODULE_OPS as _SEL_MODULE_OPS
_SEL_ATTN = set(_SEL_MODULE_OPS["self_attention"])
_SEL_MLP = set(_SEL_MODULE_OPS["mlp"])

# ── 模型预设（结构字段自 HF config.json,2026-07-11 抓取;dims=对 LLMConfig 的覆盖）─────────
# ui: 预设填充到页面输入框的值;dims: 服务端构建 LLMConfig 时 replace 的全尺寸维度。
# base: "v3"=deepseek_v3 缩层基座 / "v4"=deepseek_v4（dsv4_hybrid,compress_ratios 逐层按 0/4/128
# 循环近似——HF 全列表未逐层抓取,已标注）。
PRESETS = {
    "custom": {
        "label": "Custom（自定义）", "base": "v3",
        "source": "自定义结构（改任意字段即自动切到此项;维度全由下方输入框决定）",
        "ui": {"attn": "mla", "layers": 8, "dense_k": 1, "experts": 8, "topk": 4,
               "heads": 8, "kv_groups": 8, "seq": 4096, "batch": 1, "mtp": 0,
               "hidden": 1792, "ffn": 3072, "moe_ffn": 1024, "q_lora": 1536, "kv_lora": 512,
               "qk_nope": 128, "qk_rope": 64, "v_head": 192, "vocab": 129280},
        "dims": {},
    },
    "dsv3_mini": {
        "label": "DSv3-mini（仓库锚点）", "base": "v3",
        "source": "仓库缩层配置(真机验证锚点 12473)",
        "ui": {"attn": "mla", "layers": 8, "dense_k": 1, "experts": 8, "topk": 4,
               "heads": 8, "kv_groups": 8, "seq": 4096, "batch": 1, "mtp": 0,
               "hidden": 1792, "ffn": 3072, "moe_ffn": 1024, "q_lora": 1536, "kv_lora": 512,
               "qk_nope": 128, "qk_rope": 64, "v_head": 192, "vocab": 129280},
        "dims": {},
    },
    "dsv3_671b": {
        "label": "DeepSeek-V3 671B", "base": "v3",
        "source": "HF deepseek-ai/DeepSeek-V3 config.json",
        "ui": {"attn": "mla", "layers": 61, "dense_k": 3, "experts": 256, "topk": 8,
               "heads": 128, "kv_groups": 128, "seq": 4096, "batch": 1, "mtp": 1,
               "hidden": 7168, "ffn": 18432, "moe_ffn": 2048, "q_lora": 1536, "kv_lora": 512, "qk_nope": 128, "qk_rope": 64, "v_head": 128, "vocab": 129280},
        "dims": {"hidden_size": 7168, "ffn_hidden_size": 18432, "moe_ffn_hidden_size": 2048,
                 "q_lora_rank": 1536, "kv_lora_rank": 512, "qk_nope_head_dim": 128,
                 "qk_rope_head_dim": 64, "v_head_dim": 128, "head_dim": 192,
                 "vocab_size": 129280, "moe_shared_expert_num": 1, "moe_shared_ffn_hidden_size": 2048},
    },
    "dsv32_exp": {
        "label": "DeepSeek-V3.2-Exp", "base": "v3",
        "source": "HF deepseek-ai/DeepSeek-V3.2-Exp config.json;DSA indexer(64/128/topk2048) 预估计——"
                  "op 图基于 mindformers training_graph 静态图 DSA 代码,无真机锚点,待 pynative DSA 落地重校准",
        "ui": {"attn": "dsa", "layers": 61, "dense_k": 3, "experts": 256, "topk": 8,
               "heads": 128, "kv_groups": 128, "seq": 4096, "batch": 1, "mtp": 1,
               "hidden": 7168, "ffn": 18432, "moe_ffn": 2048, "q_lora": 1536, "kv_lora": 512, "qk_nope": 128, "qk_rope": 64, "v_head": 128, "vocab": 129280},
        "dims": {"hidden_size": 7168, "ffn_hidden_size": 18432, "moe_ffn_hidden_size": 2048,
                 "q_lora_rank": 1536, "kv_lora_rank": 512, "qk_nope_head_dim": 128,
                 "qk_rope_head_dim": 64, "v_head_dim": 128, "head_dim": 192,
                 "dsa_indexer_n_heads": 64, "dsa_indexer_head_dim": 128, "dsa_indexer_topk": 2048,
                 "vocab_size": 129280, "moe_shared_expert_num": 1, "moe_shared_ffn_hidden_size": 2048},
    },
    "dsv4_flash": {
        "label": "DeepSeek-V4-Flash", "base": "v4",
        "source": "HF deepseek-ai/DeepSeek-V4-Flash config.json;compress_ratios 逐层按 0/4/128 循环近似",
        "ui": {"attn": "dsv4_hybrid", "layers": 43, "dense_k": 1, "experts": 256, "topk": 6,
               "heads": 64, "kv_groups": 1, "seq": 4096, "batch": 1, "mtp": 1,
               "hidden": 4096, "ffn": 12288, "moe_ffn": 2048, "q_lora": 1024, "kv_lora": 512, "qk_nope": 128, "qk_rope": 64, "v_head": 192, "vocab": 129280},
        "dims": {"hidden_size": 4096, "moe_ffn_hidden_size": 2048,
                 "q_lora_rank": 1024, "o_lora_rank": 1024, "o_groups": 8,
                 "dsa_indexer_n_heads": 64, "dsa_indexer_head_dim": 128, "dsa_indexer_topk": 512,
                 "vocab_size": 129280, "moe_shared_expert_num": 1, "moe_shared_ffn_hidden_size": 2048},
    },
    "dsv4_pro": {
        "label": "DeepSeek-V4-Pro", "base": "v4",
        "source": "HF deepseek-ai/DeepSeek-V4-Pro config.json;compress_ratios 逐层按 0/4/128 循环近似",
        "ui": {"attn": "dsv4_hybrid", "layers": 61, "dense_k": 1, "experts": 384, "topk": 6,
               "heads": 128, "kv_groups": 1, "seq": 4096, "batch": 1, "mtp": 1,
               "hidden": 7168, "ffn": 18432, "moe_ffn": 3072, "q_lora": 1536, "kv_lora": 512, "qk_nope": 128, "qk_rope": 64, "v_head": 192, "vocab": 129280},
        "dims": {"hidden_size": 7168, "moe_ffn_hidden_size": 3072,
                 "q_lora_rank": 1536, "o_lora_rank": 1024, "o_groups": 16,
                 "dsa_indexer_n_heads": 64, "dsa_indexer_head_dim": 128, "dsa_indexer_topk": 1024,
                 "vocab_size": 129280, "moe_shared_expert_num": 1, "moe_shared_ffn_hidden_size": 3072},
    },
    "glm5": {
        "label": "GLM-5 (zai-org)", "base": "v3",
        "source": "HF zai-org/GLM-5 config.json(glm_moe_dsa);DSA indexer(32/128/topk2048) 预估计——"
                  "op 图基于 mindformers training_graph 静态图 DSA 代码,无真机锚点,待 pynative DSA 落地重校准",
        "ui": {"attn": "dsa", "layers": 78, "dense_k": 3, "experts": 256, "topk": 8,
               "heads": 64, "kv_groups": 64, "seq": 4096, "batch": 1, "mtp": 0,
               "hidden": 6144, "ffn": 12288, "moe_ffn": 2048, "q_lora": 2048, "kv_lora": 512, "qk_nope": 192, "qk_rope": 64, "v_head": 256, "vocab": 154880},
        "dims": {"hidden_size": 6144, "ffn_hidden_size": 12288, "moe_ffn_hidden_size": 2048,
                 "q_lora_rank": 2048, "kv_lora_rank": 512, "qk_nope_head_dim": 192,
                 "qk_rope_head_dim": 64, "v_head_dim": 256, "head_dim": 256,
                 "dsa_indexer_n_heads": 32, "dsa_indexer_head_dim": 128, "dsa_indexer_topk": 2048,
                 "vocab_size": 154880, "moe_shared_expert_num": 1, "moe_shared_ffn_hidden_size": 2048},
    },
}


def _i(p, k, d):
    try:
        return int(p.get(k, d))
    except (TypeError, ValueError):
        return -1


def parse_pp_split(s, pp, N):
    """「pp 层分配」文本 → **transformer 层分配**（不含伪层）。对应 mindformers `num_layer_list`
    口径:用户只输 transformer 层数/stage（如 "5,5,6,6,6,6,5,4"）——embedding/head/MTP **不占配额**,
    由 eval_config 在 cfg 构建后归置(embedding→stage0、head+mtp→末 stage,2026-07-11 修:此前在此处
    补伪层拿不到 mtp 数,V4 带 MTP 时和差 1 报错)。返回 (errors, tuple|None)。空串 → None(均匀切)。"""
    s = (s or "").strip()
    if not s:
        return [], None
    try:
        parts = [int(x) for x in s.replace("，", ",").split(",")]
    except ValueError:
        return [f"pp 层分配 {s!r} 解析失败（应为逗号分隔整数,如 3,5）"], None
    if len(parts) != pp:
        return [f"pp 层分配段数({len(parts)}) 必须 == pp({pp})"], None
    if sum(parts) != N:
        return [f"pp 层分配之和({sum(parts)}) 必须 == 可切分总层数 transformer+mtp({N})"], None
    if any(x < 1 for x in parts):
        return [f"pp 层分配每段须 ≥1:{parts}"], None
    return [], tuple(parts)


def stage_decoder_layers(pp, N, mtp, pp_split):
    """stage → decoder 层 id 集（1..N,评估器口径;mtp 层不参与 select,忽略）。
    与 ParallelModel._layer_to_stage 的切分规则严格一致:pp_split(用户 transformer+mtp 配额)
    或均匀切(mid=N+mtp,per=mid//pp,remainder 归末)。"""
    mid = N + max(0, mtp)
    out = {s: set() for s in range(pp)}
    if pp_split:                                  # 用户配额(不含伪层)
        pre = 0
        for s, cnt in enumerate(pp_split):
            for i in range(pre + 1, pre + cnt + 1):   # 全局可切层 1..mid
                if i <= N:                             # >N 的是 mtp,忽略
                    out[s].add(i)
            pre += cnt
    else:
        base, rem = divmod(mid, pp)                # 与 ParallelModel 同:前 rem 个 stage 各多 1
        i = 0
        for s in range(pp):
            for _ in range(base + (1 if s < rem else 0)):
                if i + 1 <= N:
                    out[s].add(i + 1)
                i += 1
    return out


def parse_stage_select(s, pp, N, mtp, pp_split):
    """「per-stage 选重」文本 → {layer_id: set(op 子串)}（2026-07-14,用户口径:按 stage 配置）。
    语法 `s<i>[-<j>]: 模式; ...`,模式 = none | self_attention | mlp | both(≈full,真机退化端 0.991)
    | 任意 op 名子串(逗号分)。stage→层映射与当前 pp 切分(含 pp_split/mtp)严格一致。
    返回 (errors, dict|None)。空串 → None。"""
    s = (s or "").strip()
    if not s:
        return [], None
    smap = stage_decoder_layers(pp, N, mtp, pp_split)
    out = {}
    for part in s.replace("；", ";").split(";"):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            return [f"per-stage 选重 {part!r} 缺 `:`（格式 s0:both 或 s1-3:self_attention）"], None
        rng, mode = (x.strip() for x in part.split(":", 1))
        if not rng.startswith("s"):
            return [f"per-stage 选重段 {rng!r} 须以 s 开头（如 s0 / s2-5）"], None
        body = rng[1:]
        try:
            a, b = (int(x) for x in body.split("-")) if "-" in body else (int(body), int(body))
        except ValueError:
            return [f"per-stage 选重 stage 范围 {rng!r} 解析失败"], None
        if not (0 <= a <= b < pp):
            return [f"per-stage 选重 stage 范围 {rng!r} 越界（0..{pp-1}）"], None
        ml = mode.strip().lower()
        if ml in ("none", "no", ""):
            continue
        if ml == "both":
            ops = _SEL_ATTN | _SEL_MLP           # ≈full(真机退化端 0.991)
        elif ml == "self_attention":
            ops = set(_SEL_ATTN)
        elif ml == "mlp":
            ops = set(_SEL_MLP)
        else:
            ops = {x.strip() for x in mode.replace("，", ",").split(",") if x.strip()}
        for st in range(a, b + 1):
            for lid in smap[st]:
                out.setdefault(lid, set()).update(ops)
    return [], (out or None)


def parse_select_cfg(s, N):
    """「细粒度选重」文本 → {layer_id(1-based): set(op 子串)}。对应 mindformers
    `select_module: {cell: [ranges]}` 口径(0-indexed decoder 层,同 from_mindformers +1 偏移):
      格式  `pattern: 层范围; pattern: 层范围`
      pattern = cell 名(self_attention/mlp,展开为该 cell 的 op 集)或任意 op 名子串(flash/e_fc1/ln2…)
      层范围 = `0-3,6`（逗号分段,`a-b` 闭区间）
    返回 (errors, dict|None)。空串 → None。"""
    s = (s or "").strip()
    if not s:
        return [], None
    out = {}
    for part in s.replace("；", ";").split(";"):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            return [f"细粒度选重 {part!r} 缺 `:`（格式 pattern: 层范围,如 self_attention:0-3）"], None
        pat, rng = (x.strip() for x in part.split(":", 1))
        ops = (_SEL_ATTN if pat == "self_attention" else
               (_SEL_MLP if pat == "mlp" else {pat}))
        if not pat:
            return ["细粒度选重 pattern 为空"], None
        for seg in rng.replace("，", ",").split(","):
            seg = seg.strip()
            if not seg:
                continue
            try:
                a, b = (int(x) for x in seg.split("-")) if "-" in seg else (int(seg), int(seg))
            except ValueError:
                return [f"细粒度选重层范围 {seg!r} 解析失败（a-b 或单层号,0-indexed）"], None
            if not (0 <= a <= b < N):
                return [f"细粒度选重层范围 {seg!r} 越界（0-indexed decoder 层,0..{N-1}）"], None
            for l0 in range(a, b + 1):
                out.setdefault(l0 + 1, set()).update(ops)   # +1: 评估器层 id(embedding=0)
    return [], (out or None)


def _is_stage_seg_key(k):
    """段 key 是否为 stage 定位（s0 / s1-2）——'s' 后接 数字 或 数字-数字。统一入口据此自动识别坐标系。"""
    if not k.startswith("s"):
        return False
    b = k[1:]
    parts = b.split("-")
    return b != "" and len(parts) in (1, 2) and all(pt.isdigit() for pt in parts)


def parse_recompute_cfg(s, pp, N, mtp, pp_split):
    """统一「细粒度重算」入口（用户报告 #2：per-stage 与 mf 层号两框合一,自动识别、互斥）。
    一个文本框、按每段 key 自动判定坐标系（两种写法**不可混用**）：
      - key 形如 `s0` / `s1-2`（s+stage 号）→ **按 PP stage**（`parse_stage_select`，stage→层映射跟当前切分）。
      - key 是 cell/op 名（self_attention/mlp/flash…）→ **按绝对层号 0-indexed**（`parse_select_cfg`，mf 口径）。
    二者下游都产出同一个 `{layer_id: set(op 子串)}`（此前它们本就互斥、殊途同归）。空串→([], None)。"""
    s = (s or "").strip()
    if not s:
        return [], None
    segs = [x.strip() for x in s.replace("；", ";").split(";") if x.strip()]
    keys = []
    for seg in segs:
        if ":" not in seg:
            return [f"细粒度重算 {seg!r} 缺 `:`（按 stage：s0:both / 按层号：mlp:0-3）"], None
        keys.append(seg.split(":", 1)[0].strip().lower())
    stage_like = [_is_stage_seg_key(k) for k in keys]
    if all(stage_like):
        return parse_stage_select(s, pp, N, mtp, pp_split)      # 全 stage 写法
    if any(stage_like):
        return ["细粒度重算不可混用 stage(s0:…) 与层号(mlp:0-3) 两种写法——请统一为其中一种"], None
    return parse_select_cfg(s, N)                                # 全层号写法（mf 口径）


def parse_and_validate(p):
    """query dict → (errors:list[str], cfg:LLMConfig|None, pc_args:dict|None)。全部校验先行、报中文。"""
    errs = []
    N = _i(p, "layers", 8); B = _i(p, "batch", 1); S = _i(p, "seq", 4096)
    heads = _i(p, "heads", 8); kvg = _i(p, "kv_groups", heads)
    dense_k = _i(p, "dense_k", 1); E = _i(p, "experts", 8); topk = _i(p, "topk", 4)
    dp = _i(p, "dp", 2); tp = _i(p, "tp", 1); ep = _i(p, "ep", 1)
    pp = _i(p, "pp", 1); cp = _i(p, "cp", 1); vpp = _i(p, "vpp", 1)
    mtp = _i(p, "mtp", 0)
    mbs_raw = (p.get("mbs") or "").strip()
    mbs = _i(p, "mbs", 0) if mbs_raw else 0          # 0=auto(=pp)
    if mbs_raw and mbs < 1:
        errs.append("microbatch 数必须是 ≥1 的整数(或留空=auto)")
    if vpp < 1:
        errs.append("vpp 必须是 ≥1 的整数")
    if mtp < 0:
        errs.append("mtp 层数必须是 ≥0 的整数")
    T = N + max(0, mtp)      # 可切分总层数 = transformer + MTP（2026-07-11 用户口径:mtp 计入切分）
    attn = p.get("attn", "mla"); method = p.get("method", "colossal")
    rmode = p.get("recompute", "None"); sel = p.get("select", "attn")
    # 结构维度（custom/微调:UI 传入即覆盖;未传(-1)则用预设 dims/基座默认）
    _DIMF = {"hidden": "hidden_size", "ffn": "ffn_hidden_size", "moe_ffn": "moe_ffn_hidden_size",
             "q_lora": "q_lora_rank", "kv_lora": "kv_lora_rank", "qk_nope": "qk_nope_head_dim",
             "qk_rope": "qk_rope_head_dim", "v_head": "v_head_dim", "vocab": "vocab_size"}
    dim_over = {}
    for uik, field in _DIMF.items():
        v = _i(p, uik, -1)
        if p.get(uik) is not None and v < 1:
            errs.append(f"{uik} 必须是 ≥1 的整数")
        elif v >= 1:
            dim_over[field] = v
    if "qk_nope_head_dim" in dim_over and "qk_rope_head_dim" in dim_over:
        dim_over["head_dim"] = dim_over["qk_nope_head_dim"] + dim_over["qk_rope_head_dim"]

    for name, v, lo in [("layers", N, 1), ("batch", B, 1), ("seq", S, 1), ("heads", heads, 1),
                        ("dp_shard", dp, 1), ("tp", tp, 1), ("ep", ep, 1), ("pp", pp, 1), ("cp", cp, 1),
                        ("dense_k", dense_k, 0), ("experts", E, 0), ("topk", topk, 1), ("kv_groups", kvg, 1)]:
        if v < lo:
            errs.append(f"{name} 必须是 ≥{lo} 的整数")
    if errs:
        return errs, None, None
    if attn not in ("mla", "gqa", "mha", "dsa", "dsv4_hybrid"):
        errs.append(f"attn 结构 {attn!r} 不支持（mla/gqa/mha/dsa/dsv4_hybrid）")
    preset = p.get("preset", "dsv3_mini")
    if preset not in PRESETS:
        errs.append(f"未知模型预设 {preset!r}")
    if method not in _CP_METHODS:
        errs.append(f"cp 算法 {method!r} 不支持（{'/'.join(_CP_METHODS)}）")
    if dense_k > N:
        errs.append(f"dense 层数({dense_k}) 不能超过总层数({N})")
    has_moe = dense_k < N and E > 0
    # ── 并行合法性（镜像 ParallelModel/resolve 的规则,先行报错）──
    if heads % tp:
        errs.append(f"tp({tp}) 必须整除注意力头数 heads({heads})")
    if attn in ("gqa", "mha") and kvg % tp:
        errs.append(f"tp({tp}) 必须整除 KV 组数 kv_groups({kvg})（gqa 的 KV 头也按 tp 切）")
    if attn in ("gqa", "mha") and heads % kvg:
        errs.append(f"kv_groups({kvg}) 必须整除 heads({heads})")
    if S % cp:
        errs.append(f"cp({cp}) 必须整除序列长 seq({S})")
    if pp > T:
        errs.append(f"pp({pp}) 不能超过可切分总层数 transformer+mtp({T})")
    if has_moe:
        if E % ep:
            errs.append(f"ep({ep}) 必须整除专家数 experts({E})")
        if topk > E:
            errs.append(f"topk({topk}) 不能超过专家数 experts({E})")
    elif ep > 1:
        errs.append(f"无 MoE 层（dense_k={dense_k} ≥ layers 或 experts=0），ep({ep}) 无意义,请设 1")
    if (dp * cp * tp) % ep:
        errs.append(f"ep({ep}) 必须整除 dp_shard·cp·tp={dp*cp*tp}（专家在该区内分片,ParallelModel 规则）")
    if rmode not in ("None", "full", "select", "custom"):
        errs.append(f"recompute {rmode!r} 不支持")
    # 重算层范围（sel_layers, a-b）：对 **full/select/custom 均生效**（空=全部层 1-N）。
    # 此前仅 custom 消费 → full/select 改层范围结构图不变（用户报告 #1，2026-07-15 修）。
    sel_ops_raw = [s.strip() for s in p.get("sel_ops", "").split(",") if s.strip()]
    lr_raw = p.get("sel_layers", "").strip()
    lr = lr_raw or f"1-{N}"
    try:
        a, b = (int(x) for x in lr.split("-")) if "-" in lr else (int(lr), int(lr))
    except ValueError:
        a = b = -1
    # pp 层分配（mindformers num_layer_list 口径）——提前解析：细粒度重算按 stage 写法需其做 stage→层映射。
    e_pp, pp_split = parse_pp_split(p.get("pp_split", ""), pp, T)
    errs += e_pp
    # 细粒度重算（统一入口,用户报告 #2：per-stage 与层号两写法合一,自动识别、互斥）：非空即优先于
    # 图上勾选/select 模块。`sel_stage` 保留为**隐藏兼容别名**（旧查询串/yaml）,sel_cfg 空时回落读取。
    e_sel, sel_cfg = parse_recompute_cfg(
        p.get("sel_cfg", "").strip() or p.get("sel_stage", ""), pp, N, mtp, pp_split)
    errs += e_sel
    # 非空层范围：full/select/custom 均校验（越界/倒序即报错，不静默忽略）
    if lr_raw and rmode in ("full", "select", "custom") and not (1 <= a <= b <= N):
        errs.append(f"重算层范围 {lr_raw!r} 非法（1-{N} 内的 a-b）")
    if rmode == "custom" and sel_cfg is None and not sel_ops_raw:
        errs.append("custom 重算需至少勾选一个 op（图上点 ↻）或填「细粒度重算」文本")
    if errs:
        return errs, None, None

    # 预设基座（dims 覆盖 = HF config.json 的全尺寸维度）+ UI 字段最终覆盖。
    pr = PRESETS[preset]
    base = (deepseek_v4(N) if (pr["base"] == "v4" or attn == "dsv4_hybrid") else deepseek_v3(N))
    if pr["dims"]:
        base = dataclasses.replace(base, **pr["dims"])
    if dim_over:                               # UI 维度最终覆盖（custom / 预设微调）
        base = dataclasses.replace(base, **dim_over)
    # dsa（DSv3.2/GLM-5 的 MLA+lightning indexer 稀疏注意力）:真实 attn_type=dsa（layers/dsa.py，
    # **预估计**——基于 training_graph 静态图 DSA 代码,无真机锚点,待 pynative DSA 落地重校准）。
    # indexer 三维未由预设/UI 提供时按 DSv3.2-Exp 默认(64/128/2048)补齐,避免 fail-loud 拦住 custom。
    _attn_map = {"mha": "gqa"}
    dsa_fill = {}
    if attn == "dsa":
        if base.dsa_indexer_n_heads <= 0:
            dsa_fill["dsa_indexer_n_heads"] = 64
        if base.dsa_indexer_head_dim <= 0:
            dsa_fill["dsa_indexer_head_dim"] = 128
        if base.dsa_indexer_topk <= 0:
            dsa_fill["dsa_indexer_topk"] = 2048
    cfg = dataclasses.replace(
        base, num_layers=N, batch_size=B, seq_length=S,
        attn_type=_attn_map.get(attn, attn),
        **dsa_fill,
        num_attention_heads=heads,
        # dsv4_hybrid 的 num_query_groups 用基座值（MLA 系惰性=1）;mla/mha/dsa=heads;gqa=kv_groups。
        num_query_groups=(base.num_query_groups if attn == "dsv4_hybrid"
                          else (heads if attn in ("mla", "mha", "dsa") else kvg)),
        first_k_dense_replace=dense_k,
        num_moe_experts=(E if has_moe else None),
        moe_router_topk=topk,
        mtp_num_layers=max(0, mtp))
    # （细粒度重算已在上方统一入口 parse_recompute_cfg 解析,含 per-stage 写法——用户报告 #2 合并,
    #   此处不再单独处理 per-stage；stage→层映射用的是**归置前**的用户配额 pp_split，口径不变。）
    # pp 层分配 → 含伪层的 layers_per_stage:embedding→stage0、head+MTP→末 stage（不占用户配额;
    # mtp 数只有 cfg 构建后可知——V4 预设 num_nextn_predict_layers=1）。
    if pp_split is not None:
        # mtp 计入用户配额(和==N+mtp,MTP 位于层序列末端、落在末段配额里)→ 只补 embedding/head 伪层。
        full = list(pp_split)
        full[0] += 1
        full[-1] += 1
        pp_split = tuple(full)
    pc_args = dict(dp=dp, tp=tp, ep=(ep if has_moe else 1), pp=pp, cp=cp, method=method,
                   rmode=rmode, sel=sel, N=N, sel_ops=sel_ops_raw, sel_range=(a, b),
                   sel_cfg=sel_cfg, pp_split=pp_split, vpp=vpp, mbs=mbs)
    return [], cfg, pc_args


def _dtype_lbl(b):
    return {1: "int8(1B)", 2: "bf16(2B)", 4: "fp32(4B)"}.get(b, f"{b}B")


def _shape_info(tref, rt, dims, dtype_b):
    """符号 shape × 每维数值 × 切分除数 → 计算说明（让 MiB 可追溯）。
    tref=原始 TensorRef(符号 shape+shard 标注);rt=ResolvedTensor(切分后 numel,真值)。"""
    from cost_eval.shape_eval import eval_expr

    def _dim(d):
        s = str(d)
        return f"({s})" if ("+" in s or "*" in s) else s   # 复合维加括号防歧义
    sym = "·".join(_dim(d) for d in tref.shape) if tref is not None else "?"
    vals = None
    div = 1
    axes = ""
    if tref is not None:
        try:
            vs = [int(round(eval_expr(str(d), dims))) for d in tref.shape]
            vals = "×".join(str(v) for v in vs)
            full = 1
            for v in vs:
                full *= v
            # 除数 = 全量/切分后（真值,来自 resolve）;轴标注只列 shard 声明（cp 是否真切由除数体现）
            div = max(1, round(full / rt.local_numel)) if rt.local_numel else 1
            axes = "·".join(sorted(set(tref.shard.values())))
        except Exception:
            pass
    calc = f"({sym})" + (f"={vals}" if vals else "")
    if div > 1:
        calc += f" ÷{div}" + (f"({axes})" if axes else "")
    calc += f" ·{_dtype_lbl(dtype_b)}"
    return calc


def graph_json(layers, norm_dtype, spec, dims, recompute=None):
    """一个 stage 的 ResolvedLayer 列表 → 逐层 op-DAG JSON（含 shape 计算说明）。

    per-op saves 按 norm-fp32 口径调整（与仿真器一致）。**重算感知**（2026-07-15 用户报告：
    结构图激活此前无论重算如何都不变）：传入 `recompute`（RecomputeSpec）后，逐 op 按**仿真器
    同口径**（`mem_timeline.py:651-660` 的 FWD pin）标注是否被重算（saves 反向重物化、前向不存）：
      - 层头 `act_mib` = 该层 **stored 激活总量**（与仿真器逐字节同口径）：
        None→`activation_saves`（全量存）/ full→`checkpoint_input`（仅层入口边界）/
        select→`estimate_select_memory.act_live_pinned`（非选中 saves ∪ 层入口边界）。
      - 每 op：`recomp`（bool，被重算=saves 不存）、stored `act_mib`（只计**存下**的 saves）、
        `recomp_mib`（被重算省下的激活量）；每条 acts 明细带 `stored` 标志。
      - `recompute=None`（缺省，如无重算路径）→ pinned_names=None → 全 stored、`recomp=none`，
        `act_mib` == 旧 `activation_saves`，**逐字节复现旧行为**。
    原始 OpSpec（符号 shape/shard）与 resolved op 按序对齐——resolve 保序遍历,zip 安全。"""
    from cost_eval.specs import RecomputeSpec
    rc = recompute if recompute is not None else RecomputeSpec()
    out = []
    for l in layers:
        lid = l.layer_id
        sm = estimate_structure_memory(l.ops, norm_compute_dtype_bytes=norm_dtype)
        _optype = lambda op: getattr(op.type, "value", op.type)
        # 该层重算态 → 层头 stored 总量（与 mem_timeline FWD `saved=...` 逐字节同口径）。
        #   - none  ：`activation_saves`（全量 saves 常驻）。
        #   - full  ：`checkpoint_input`（仅保层入口锚点，层内所有 op 反向重物化）。
        #   - select：`act_live_pinned`（非选中 op saves ∪ 层入口锚点）。
        # per-op stored 判定则更朴素——**被重算的 op 前向不存任何 saves**（stored=not op_recomp）；
        # 层入口 checkpoint_input 是**层级重算锚点**（上一层输出，非本层某 op 的激活），单列 `entry_mib`
        # 于层头解释 stored 总量与 per-op 之差，不摊到某个被重算 op（否则「此 op 重算却仍显 X MiB」易误读）。
        if rc.is_full(lid):
            recomp_state = "full"
            layer_act = sm.checkpoint_input
        elif rc.is_select(lid):
            recomp_state = "select"
            layer_act = estimate_select_memory(
                l.ops, lambda op: rc.op_matches(lid, op.name, _optype(op)),
                norm_compute_dtype_bytes=norm_dtype).act_live_pinned
        else:
            recomp_state = "none"
            layer_act = sm.activation_saves
        entry_mib = round(sm.checkpoint_input / MiB, 2) if recomp_state != "none" else 0
        orig_ops = spec.get_layer(l.layer_type).ops
        ops, edges = [], []
        eseen = set()
        produced = {}
        for i, (op, oop) in enumerate(zip(l.ops, orig_ops)):
            for t in op.inputs:
                if t.name in produced and (produced[t.name], i) not in eseen:
                    eseen.add((produced[t.name], i))
                    edges.append([produced[t.name], i])
            produced[op.output.name] = i
            op_recomp = (recomp_state == "full") or (
                recomp_state == "select" and rc.op_matches(lid, op.name, _optype(op)))
            acts = []
            stored_b = recomp_b = 0
            for t, ot in zip(op.saves, oop.saves):
                is_norm_fp32 = (op.type == "norm" and "softmax" not in op.name.lower()
                                and norm_dtype > t.dtype_bytes)
                eff_dtype = norm_dtype if is_norm_fp32 else t.dtype_bytes
                b = t.local_numel * eff_dtype
                # 被重算的 op 前向不存任何 saves（反向重物化）；否则常驻。
                stored = not op_recomp
                if stored:
                    stored_b += b
                else:
                    recomp_b += b
                acts.append({"name": t.name, "mib": round(b / MiB, 2), "stored": stored,
                             "calc": _shape_info(ot, t, dims, eff_dtype)
                                     + (" ←norm 存 fp32 输入" if is_norm_fp32 else "")
                                     + ("" if stored else " ←重算:反向重物化,前向不存")})
            ops.append({"i": i, "name": op.name, "type": op.type,
                        "act_mib": round(stored_b / MiB, 2), "acts": acts,
                        "recomp": op_recomp, "recomp_mib": round(recomp_b / MiB, 2),
                        "param_mib": round(sum(t.local_numel * t.dtype_bytes for t in op.params) / MiB, 2),
                        "out": {"name": op.output.name,
                                "mib": round(op.output.local_numel * op.output.dtype_bytes / MiB, 2),
                                "sym": "·".join(str(d) for d in oop.output.shape),
                                "calc": _shape_info(oop.output, op.output, dims, op.output.dtype_bytes)},
                        "ins": [t.name for t in op.inputs],
                        "ws_mib": round(op.workspace_bytes / MiB, 1)})
        out.append({"id": l.layer_id, "type": l.layer_type,
                    "act_mib": round(layer_act / MiB, 1),                   # stored 总量（仿真器口径）
                    "full_act_mib": round(sm.activation_saves / MiB, 1),   # 无重算全量（对照）
                    "recomp": recomp_state,                                 # none|full|select
                    "entry_mib": entry_mib,                                 # 重算态层入口锚点（stored）
                    "param_mib": round(sum(o["param_mib"] for o in ops), 1),
                    "ops": ops, "edges": edges})
    return out


# ── yaml 导入 round-trip:非 UI 可表达的 extra 键(P1-17/§4.8 闭环,2026-07-15)────────────
# 手配路径这些键**缺省** → 复现历史固定假设(设备 64GiB / AdamW fp32 / swap 关 /
# dp_replicate=1 / reshard=default / offload 关 / prefetch=1),eval_config 输出逐字节不变。
# yaml 导入路径:`_bundle_to_fields` 把 bundle 的完整解析值(含这些 extra)回填成 UI/隐藏字段
# → 随 qs() 回传 → `_build_eval_specs` 按解析值构造 → **页面评估用完整 bundle,不是固定假设**。
def _x_int(p, k, default):
    """extra 整数:键缺省/空串 → default;非法 → default(不阻断评估)。"""
    v = p.get(k)
    if v is None or (isinstance(v, str) and not v.strip()):
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _x_flag(p, k, default):
    """extra 布尔:键缺省/空串 → default;'1'/true/on/yes → True;'0'/false/off/no → False。"""
    v = p.get(k)
    if v is None or (isinstance(v, str) and not v.strip()):
        return default
    s = str(v).strip().lower()
    if s in ("1", "true", "on", "yes"):
        return True
    if s in ("0", "false", "off", "no"):
        return False
    return default


def _build_eval_specs(p, pa):
    """query dict + parsed pa → (ParallelConfig, OptimizerSpec, HardwareSpec, SwapSpec)。

    非 UI 可表达的 extra 键(dp_replicate/reshard/cpu_offload/prefetch/opt_dtype/maxdev_gib)
    缺省=历史手配假设(见上方注释)→ 手配路径逐字节不变;yaml 导入回填后 → 页面评估用完整解析值。
    仅这些 extra 影响并行 dp_replicate/reshard/offload/prefetch、优化器 dtype、设备容量;其余口径不变。
    """
    dp, tp = pa["dp"], pa["tp"]
    mbs = pa["mbs"] or (pa["pp"] if pa["pp"] > 1 else 1)   # 用户显式 or auto=pp
    # 并行 extra(缺省 = ParallelConfig 默认 → 手配不变)
    dp_repl = _x_int(p, "dp_replicate", 1)
    reshard = (p.get("reshard") or "default")
    reshard = (reshard.strip() or "default") if isinstance(reshard, str) else "default"
    offload = _x_flag(p, "cpu_offload", False)
    prefetch = _x_int(p, "prefetch", 1)
    sp_ext = _x_flag(p, "sp", None)      # 隐藏字段:导入回填 bundle.sequence_parallel;缺省→历史推导
    seq_par = (dp > 1 or tp > 1) if sp_ext is None else sp_ext
    pc = ParallelConfig(
        dp_replicate=dp_repl, dp_shard=dp, tp=tp, ep=pa["ep"], pp=pa["pp"], cp=pa["cp"],
        sequence_parallel=seq_par, num_microbatches=mbs,
        context_parallel_method=pa["method"], interleave=pa["vpp"],
        reshard_after_forward=reshard, cpu_offload=offload, prefetch_depth=prefetch,
        layers_per_stage=(list(pa["pp_split"]) if pa["pp_split"] and pa["pp"] > 1 else None))
    # 优化器 dtype(缺省 fp32,历史手配假设):bf16 params 多存一份 compute 副本(state 14 vs 12)。
    opt_fp32 = str(p.get("opt_dtype", "fp32")).strip().lower() != "bf16"
    opt = OptimizerSpec.adamw(params_fp32=opt_fp32, grad_dtype_bytes=_x_int(p, "grad_bytes", 4))
    # 设备容量(缺省 64GiB,历史手配假设):UI 以 GiB 输入 → bytes。
    mg = p.get("maxdev_gib")
    if mg is None or (isinstance(mg, str) and not str(mg).strip()):
        maxdev = 64 * 2 ** 30
    else:
        try:
            maxdev = int(round(float(mg) * 2 ** 30))
        except (TypeError, ValueError):
            maxdev = 64 * 2 ** 30
    hw = HardwareSpec(max_device_memory=maxdev, framework_reserve=0)
    # swap:yaml 导入侧恒关(from_mindformers_dict 对 swap.enable fail-loud) → 手配同 SwapSpec()。
    return pc, opt, hw, SwapSpec()


def eval_config(p):
    errs, cfg, pa = parse_and_validate(p)
    if errs:
        return {"ok": False, "errors": errs}
    spec = build_llm_spec(cfg)
    d = spec.dims
    N = pa["N"]
    a, b = pa["sel_range"]   # 重算层范围（空 sel_layers → 1-N，见 parse_and_validate；full/select 亦生效）
    if pa["rmode"] == "full":
        rc = RecomputeSpec("full", full_layers=set(range(a, b + 1)))
    elif pa["rmode"] == "select":
        selset = _SEL_ATTN if pa["sel"] == "attn" else (_SEL_MLP if pa["sel"] == "mlp" else _SEL_ATTN | _SEL_MLP)
        rc = RecomputeSpec("select", select_ops={lid: set(selset) for lid in range(a, b + 1)})
    elif pa["sel_cfg"]:
        # 细粒度文本（mindformers select_module 口径,每 pattern 可不同层集）——非空即优先。
        rc = RecomputeSpec("select", select_ops=pa["sel_cfg"])
    elif pa["rmode"] == "custom":
        # 图上勾选:任意 op 名 × 单一层范围 —— 与 mindformers select_recompute（op 位置级）同口径。
        rc = RecomputeSpec("select", select_ops={lid: set(pa["sel_ops"]) for lid in range(a, b + 1)})
    else:
        rc = RecomputeSpec("None")
    # P1-02④（2026-07-14 review）：select 选择器**命中数校验**——错拼 op 子串此前静默空转
    # （层仍被标 select → kept_frag margin 生效 →「越错越贵」，真机应≈none）。判据按**每层
    # 选择器并集**（预设 cell 集刻意覆盖 MLA+GQA 两套 op 词汇,逐 selector 判会误杀）：某层
    # 的全部 selector 在该层 op 图零命中 → 报错并列出可用 op 名,不静默评估。
    if rc.mode == "select":
        for lid, sels in sorted(rc.select_ops.items()):
            if not (0 <= lid < len(spec.layer_pattern)):
                return {"ok": False, "errors": [f"选重层号 {lid} 超出层图范围(0..{len(spec.layer_pattern)-1})"]}
            lops = spec.layer_specs[spec.layer_pattern[lid]].ops
            hit = any(s.lower() in op.name.lower()
                      or s.lower() in str(getattr(op.type, "value", op.type)).lower()
                      for s in sels for op in lops)
            if not hit:
                return {"ok": False, "errors": [
                    f"选重 pattern {sorted(sels)} 在层 {lid}({spec.layer_pattern[lid]}) 的 op 图"
                    f"**零命中**——静默空转会错算(真机同样不生效)。该层可用 op 名: "
                    f"{', '.join(op.name for op in lops)}"]}
    # P1-17/§4.8 闭环:pc/opt/hw/swap 由 `_build_eval_specs` 统一构造——手配路径缺 extra 时
    # 复现历史固定假设(逐字节不变);yaml 导入回填 extra 后页面评估用完整解析 bundle 值。
    pc, opt, hw, swap = _build_eval_specs(p, pa)
    ev = Evaluator(spec, pc, opt, hw, rc, swap)
    rep = ev.evaluate(record_timeline=True)
    # resolved 图（与 evaluate 同口径重解析一次,拿逐 op 切分后字节）
    from cost_eval.parallel_model import ParallelModel
    from cost_eval.shape_eval import ShapeEval
    world = pc.dp_replicate * pc.dp_shard * pc.cp * pc.tp * pc.pp
    g = ShapeEval().resolve(spec, ParallelModel(pc, d.n_layers, world))
    norm_dtype = getattr(d, "norm_compute_dtype_bytes", 0)
    # 显示层数口径（2026-07-15 修）：embedding/lm_head 是**自动归置的伪层**（embedding→stage0、
    # head→末 stage），**不计入** stage 层数——用户配的 pp_split 只含可切分层（transformer+mtp），
    # 故显示数必须与配置一致（此前 len(lys) 把伪层也数进去，stage0/末 stage 各虚增 1，对不上配置）。
    _PSEUDO = {"embedding", "lm_head"}
    # P2-01 双 OOM 口径（closure-audit v5, 2026-07-15 §4.8）：除 allocated（`.oom`，真机 OOM 主判据）
    # 外，同时输出 reserved 口径。设备 HBM 真实约束是 reserved（allocated + HCCL 通信缓冲，见
    # report.reserved_estimate_bytes 的口径边界说明）。MAXDEV = eval_config 里 HardwareSpec 硬编码的
    # 64GiB 容量（rep 已按此值构造，其 reserved_oom/allocated_oom 亦用它，故逐 stage 判定与顶层一致）。
    # **口径边界（closure-audit v2 §F7，2026-07-15）**：reserved_oom 是**下界判定**——reserved 估计
    # 只含 allocated + HCCL，**不含 allocator pool 碎片**（真机 DSv4 reserved−allocated≈676-680 MiB
    # 里 HCCL 之外还有 ~277-281 MiB pool 分量未建模）。故 `reserved_oom=True` 是确定超容，但
    # `reserved_oom=False` **不保证**真实 reserved 不超（临界区可能已超）——字段名后附 `_lb`(lower-bound)
    # 语义供前端标注，勿把 False 当"安全"。
    MAXDEV = rep.max_device_memory   # = 64 * 2**30
    stages = []
    for sp in rep.per_stage:
        lys = g.stages.get(sp.stage, [])
        split_lys = [l for l in lys if l.layer_type not in _PSEUDO]   # 可切分层（=用户配额口径）
        rng = f"{lys[0].layer_type}(L{lys[0].layer_id})…{lys[-1].layer_type}(L{lys[-1].layer_id})" if lys else ""
        extras = [l.layer_type for l in lys if l.layer_type in _PSEUDO]   # 附着的伪层（embedding/head）
        resv_bytes = rep.reserved_estimate_bytes(sp.stage)   # 该 stage: allocated 峰值 + HCCL 缓冲
        stages.append({
            "stage": sp.stage, "peak": round(sp.peak_bytes / MiB, 1), "peak_event": sp.peak_event,
            "oom": sp.oom,   # allocated 口径（保留旧名兼容）
            "reserved_oom": resv_bytes > MAXDEV,   # reserved 口径：该 stage 的 reserved 估计超容
            # P2-01（Y4，2026-07-15）：reserved 估计现 = allocated + HCCL + **pool 碎片**（1.8%
            # 自 DSv4 单点标定，framework.allocator_pool_fragmentation）→ 已非纯下界、更接近真实
            # reserved（DSv4 估 16093 vs 真机 16092-16096）。但碎片率跨模型波动（另一 DSv3 run ~4-5%）
            # → 是**标定近似**、非严格上界：reserved_oom=False 仍不保证绝对安全（临界区留余量）。
            "reserved_oom_is_calibrated_estimate": True,   # 含 pool 碎片近似,非严格上界(也非纯下界)
            "reserved_mib": round(resv_bytes / MiB, 1),   # 该 stage reserved 估计（含 pool，MiB）
            "layers_desc": rng, "n_layers": len(split_lys),
            "extras": extras,   # 该 stage 附着的伪层（不计入 n_layers；embedding→stage0/head→末 stage）
            "graph": graph_json(lys, norm_dtype, spec, d, rc),
            "timeline": [{"event": s.event, "total": round(s.total_bytes / MiB, 1),
                          "buckets": {k: round(getattr(s.breakdown, k, 0) / MiB, 1) for k in BK
                                      if getattr(s.breakdown, k, 0)}} for s in sp.timeline],
        })
    worst_reserved = max(rep.reserved_estimate_bytes(sp.stage) for sp in rep.per_stage)   # 最紧 stage
    return {"ok": True, "world": world, "tightest": rep.tightest_stage,
            "device_peak": round(max(s["peak"] for s in stages), 1), "stages": stages,
            "hccl_mib": round(rep.hccl_reserved_bytes / MiB, 0),
            "allocated_oom": rep.allocated_oom,   # P2-01 顶层：任一 stage allocated 峰值超容
            "reserved_oom": rep.reserved_oom,     # P2-01 顶层：任一 stage reserved 估计超容（含 pool 近似）
            "reserved_oom_is_calibrated_estimate": True,  # 含 pool 碎片近似(1.8% 单点标定),非严格上界
            "reserved_margin_mib": round((MAXDEV - worst_reserved) / MiB, 1)}   # 容量−最紧 stage reserved（含 pool,可负）


def _sel_ops_to_text(select_ops):
    """RecomputeSpec.select_ops {lid: set} → 「细粒度选重」文本（逆向,mindformers 口径展示）。
    相同 op 集聚层、层压 ranges;op 集==cell 展开集则显示 cell 名。"""
    from cost_eval.configs.from_mindformers import _SELECT_MODULE_OPS
    by_ops = {}
    for lid, ops in (select_ops or {}).items():
        by_ops.setdefault(frozenset(ops), []).append(lid - 1)   # → 0-indexed decoder
    parts = []
    for ops, lids in by_ops.items():
        name = next((cell for cell, cops in _SELECT_MODULE_OPS.items() if frozenset(cops) == ops), None)
        if name is None:
            name = ",".join(sorted(ops))
        lids.sort()
        rngs, s0 = [], lids[0]
        for prev, cur in zip(lids, lids[1:] + [None]):
            if cur != prev + 1:
                rngs.append(f"{s0}-{prev}" if prev > s0 else f"{s0}")
                s0 = cur
        parts.append(f"{name}:{','.join(rngs)}")
    return "; ".join(parts)


def _mf_adapt(mf):
    """老式 mindformers yaml（`parallel_config`/`recompute_config`/`runner_config`,model.model_config
    嵌套,offset/pp_interleave_num 在 model 段）→ 转换器新式段。**只改内存中的 dict,不落盘**;
    新式 yaml 原样通过。返回 (mf, vpp)。"""
    mf = dict(mf)
    off = vpp = None
    m = mf.get("model")
    if isinstance(m, dict) and "model_config" in m:            # model.model_config 展平
        mc = dict(m["model_config"])
        off = mc.pop("offset", None)                           # pipeline 层偏移(老式放 model 段)
        vpp = mc.pop("pp_interleave_num", None)                # VPP 交错数
        mf["model"] = mc
    if "parallelism" not in mf and isinstance(mf.get("parallel_config"), dict):
        pcfg = mf["parallel_config"]
        # P0.3(2026-07-14 review):老式 `data_parallel` 语义随 `parallel.enable_parallel_optimizer`
        # (epo,mindformers/mindspore 默认 **False**)分流:epo=True → 权重/优化器沿 dp 切分(zero 类)
        # = data_parallel_shard;epo=False → 纯数据并行,权重逐 dp rank **复制** = data_parallel_replicate
        # (评估器持久态只 ÷fsdp_degree=dp_shard·cp,不 ÷dp_replicate,static_mem.py:31——建模正确)。
        # 此前无条件映射 dp_shard 会把 epo=False + dp>1 的持久内存静默低估 ~dp 倍。
        epo = bool((mf.get("parallel") or {}).get("enable_parallel_optimizer", False))
        dp = int(pcfg.get("data_parallel", 1) or 1)
        mf["parallelism"] = {
            "tensor_parallel": pcfg.get("model_parallel", 1),
            "pipeline_parallel": pcfg.get("pipeline_stage", 1),
            "expert_parallel": pcfg.get("expert_parallel", 1),
            "context_parallel": pcfg.get("context_parallel", 1),
            "data_parallel_shard": dp if epo else 1,
            "data_parallel_replicate": 1 if epo else dp,
            "sequence_parallel": bool(pcfg.get("use_seq_parallel", False)),
            "pipeline_parallel_microbatch_size": pcfg.get("micro_batch_num", 1),
        }
    # 老式 `recompute_config` 段（graph 模式:recompute:True/[per-stage 列表]/select_recompute）
    # **不支持、不转换**（2026-07-14 review P1.6,用户裁决:只支持 pynative 新式 `recompute:` 段）。
    # 不静默:do_POST 检测到该段会在 warnings 里明示"未转换,请在页面手动配置重算"。
    if "training" not in mf:
        mf["training"] = {"local_batch_size": (mf.get("runner_config") or {}).get("batch_size", 1)}
    # offset → parallelism.num_layer_list:嵌套(VPP per-chunk)按 stage 跨 chunk 求和;flat 走 offset;int 忽略。
    # P1-18(final review 2026-07-14):嵌套换算依赖 num_hidden_layers——yaml 依赖类内默认时此刻缺失,
    # 用 N=0 算出的列表恒错(sum≠N,671b 导入失败根因)。缺 N 时暂存 `_nested_offset`,
    # 由 `_materialize_nested_offset` 在页面兜底注入 N 后再换算。
    if isinstance(off, (list, tuple)) and off:
        par = mf.setdefault("parallelism", {})
        if isinstance(off[0], (list, tuple)):
            par["_nested_offset"] = [list(c) for c in off]
            _materialize_nested_offset(mf)          # N 在场则就地换算,缺场保持暂存
        else:
            par["offset"] = list(off)
    return mf, int(vpp or 1)


def _materialize_nested_offset(mf, warnings=None):
    """消费 `_nested_offset` 暂存：有 `num_hidden_layers` 则换算成 `num_layer_list`
    （base=N//(pp·v)，按 stage 跨 chunk 求和）；仍无 N 则丢弃并警告——绝不静默错算。"""
    par = mf.get("parallelism")
    if not isinstance(par, dict) or "_nested_offset" not in par:
        return
    N = int((mf.get("model") or {}).get("num_hidden_layers", 0) or 0)
    if N <= 0:
        if warnings is None:
            return                                   # 等下一次(兜底后)的物化机会
        par.pop("_nested_offset")
        warnings.append("offset(嵌套 VPP)未换算:num_hidden_layers 缺失且页面无兜底,已忽略该 offset")
        return
    off = par.pop("_nested_offset")
    pp = int(par.get("pipeline_parallel", 1) or 1)
    v = len(off)
    base = N // (pp * v)
    par["num_layer_list"] = [sum(base + int(off[c][s]) for c in range(v)) for s in range(pp)]


def _bundle_to_fields(b):
    """EvaluatorConfigBundle → UI 字段 dict（yaml 导入回填;只读转换,不落盘）。"""
    llm, pc, rc = b.llm, b.parallel, b.recompute
    f = {
        "attn": llm.attn_type, "layers": llm.num_layers,
        "dense_k": (llm.first_k_dense_replace or 0),
        "experts": (llm.num_moe_experts or 0), "topk": (llm.moe_router_topk or 1),
        "heads": llm.num_attention_heads, "kv_groups": llm.num_query_groups,
        "seq": llm.seq_length, "batch": llm.batch_size,
        "hidden": llm.hidden_size, "ffn": llm.ffn_hidden_size,
        "moe_ffn": (llm.moe_ffn_hidden_size or llm.ffn_hidden_size),
        "q_lora": (llm.q_lora_rank or 1), "kv_lora": (llm.kv_lora_rank or 1),
        "qk_nope": (llm.qk_nope_head_dim or 1), "qk_rope": (llm.qk_rope_head_dim or 1),
        "v_head": (llm.v_head_dim or llm.head_dim), "vocab": llm.vocab_size,
        "dp": pc.dp_shard, "tp": pc.tp, "ep": pc.ep, "pp": pc.pp, "cp": pc.cp,
        "method": pc.context_parallel_method,
        "mtp": int(getattr(llm, "mtp_num_layers", 0) or 0),
        "recompute": ("full" if rc.mode == "full" else ("custom" if rc.mode == "select" else "None")),
        "sel_cfg": (_sel_ops_to_text(rc.select_ops) if rc.mode == "select" else ""),
    }
    if getattr(pc, "layers_per_stage", None):
        lp = list(pc.layers_per_stage)
        lp[0] -= 1; lp[-1] -= 1        # 去 embedding/head 伪层(mtp 计入可切分层数,保留在配额里)
        f["pp_split"] = ",".join(str(x) for x in lp)
    # P1-17/§4.8 闭环(2026-07-15):非 UI 子集的**完整解析值** → UI/隐藏字段,页面评估按这些值算
    #（不再固定假设 64GiB/AdamW-fp32/dp_replicate=1/reshard=default/offload 关/prefetch=1）。
    # 这些键随 qs() 回传给 eval_config → `_build_eval_specs` 消费 → 真 round-trip。
    f["dp_replicate"] = pc.dp_replicate                       # 纯数据并行度(复制,不切分)
    f["reshard"] = pc.reshard_after_forward                   # always|never|default(gather 生命周期)
    f["cpu_offload"] = int(bool(pc.cpu_offload))              # 参数/优化器态卸载 CPU
    f["prefetch"] = getattr(pc, "prefetch_depth", 1)          # FSDP 参数预取深度
    f["sp"] = int(bool(pc.sequence_parallel))                 # 隐藏字段:序列并行(导入按解析值,不再推导)
    f["maxdev_gib"] = round(b.hardware.max_device_memory / (2 ** 30), 4)   # 设备容量(GiB)
    f["opt_dtype"] = "fp32" if b.optimizer.state_bytes_per_param == 12 else "bf16"   # 优化器 params dtype
    f["grad_bytes"] = b.optimizer.grad_dtype_bytes            # 反向 grad dtype 字节
    return f


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        u = urlparse(self.path)
        if u.path == "/api/parse_yaml":
            n = int(self.headers.get("Content-Length", 0) or 0)
            body = self.rfile.read(n).decode("utf-8", errors="replace")
            # 前端 POST JSON{yaml, defaults};兼容裸 yaml 文本(旧格式)。
            defaults = {}
            txt = body
            try:
                j = json.loads(body)
                if isinstance(j, dict) and "yaml" in j:
                    txt = j["yaml"]; defaults = j.get("defaults") or {}
            except ValueError:
                pass
            try:
                import yaml as _yaml
                mf = _yaml.safe_load(txt)
                if not isinstance(mf, dict):
                    raise ValueError("yaml 顶层须是映射(mindformers 训练配置)")
                legacy_rc = isinstance(mf.get("recompute_config"), dict) and "recompute" not in mf
                mf, vpp = _mf_adapt(mf)
                # 缺必需结构字段（该 yaml 依赖 mindformers 类内默认）→ **用页面当前值兜底并显式警告**,
                # 不阻断导入（评估器仍不猜任何默认,兜底值来自用户当前对话框,可见可改）。
                warnings = []
                if legacy_rc:
                    warnings.append("老式 recompute_config(graph 模式)不支持,已忽略——"
                                    "仅支持 pynative 新式 recompute 段;请在页面手动配置重算")
                m = mf.get("model")
                if isinstance(m, dict):
                    _REQ = {"num_hidden_layers": "layers", "num_attention_heads": "heads",
                            "hidden_size": "hidden", "vocab_size": "vocab", "seq_length": "seq"}
                    from cost_eval.configs.from_mindformers import _MODEL_KEY_ALIASES
                    alias_ok = {v for k, v in _MODEL_KEY_ALIASES.items() if k in m}
                    m = dict(m)
                    for mk, uik in _REQ.items():
                        if m.get(mk) is None and mk not in alias_ok and defaults.get(uik):
                            m[mk] = int(defaults[uik])
                            warnings.append(f"{mk} 缺失(yaml 依赖类内默认)→ 用页面当前值 {defaults[uik]} 兜底,请核对")
                    mf = dict(mf); mf["model"] = m
                # P1-18:嵌套 offset 在必需字段兜底后物化(兜底可能刚注入 num_hidden_layers)
                _materialize_nested_offset(mf, warnings)
                from cost_eval.configs.from_mindformers import from_mindformers_dict
                bundle = from_mindformers_dict(mf)
                fields = _bundle_to_fields(bundle)
                # P1-17/§4.8 闭环（2026-07-15）：**完整 round-trip**——页面评估按解析出的完整
                # bundle 算（dp_replicate/reshard/offload/prefetch、设备容量、优化器 dtype 均生效,
                # 不再固定假设）。上轮「回填为 UI 子集/需走 CLI」的收窄措辞已废止:页面即可完整评估。
                warnings.append(
                    "已按完整解析值回填并评估:dp_replicate/reshard/offload/prefetch、设备容量、"
                    "优化器 dtype 均生效(可在页面对应字段查看并覆盖);swap 段导入侧恒关")
                if bundle.parallel.dp_replicate > 1:
                    # P0.3:epo=False 的纯数据并行——权重/优化器逐 dp rank 复制不切分,评估器按
                    # dp_replicate 建模(持久态不 ÷dp,单卡峰值与 dp_shard=1 相同)。dp_replicate 现
                    # **随 round-trip 生效**(进 world/HCCL 域),页面 dp_replicate 字段单列、可见可改。
                    warnings.append(
                        f"纯数据并行 dp_replicate={bundle.parallel.dp_replicate}(权重/优化器逐 rank 复制"
                        "不切分):单卡峰值与 dp_shard=1 相同,world/HCCL 域已计入 dp_replicate(round-trip "
                        "生效);页面 dp 字段=dp_shard、dp_replicate 字段单列,勿混淆")
                if vpp > 1:
                    fields["vpp"] = vpp
                self._send(json.dumps({"ok": True, "fields": fields, "warnings": warnings},
                                      ensure_ascii=False))
            except Exception as e:
                self._send(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}, ensure_ascii=False))
            return
        self.send_response(404); self.end_headers()

    def _send(self, body, ctype="application/json"):
        b = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(200); self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            self._send(PAGE.replace("__PRESETS__", json.dumps(PRESETS, ensure_ascii=False)),
                       "text/html"); return
        if u.path == "/api/eval":
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                self._send(json.dumps(eval_config(q), ensure_ascii=False))
            except Exception as e:
                self._send(json.dumps({"ok": False, "errors": [f"{type(e).__name__}: {e}"]}, ensure_ascii=False))
            return
        self.send_response(404); self.end_headers()


PAGE = r"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>LLM 内存实验台 v2</title>
<style>
:root{--bg:#f5f6f8;--card:#fff;--ink:#161b24;--mut:#6a7280;--line:#e3e6ec;--blue:#2f4b7c;--saved:#c0392b;--mono:ui-monospace,Consolas,Menlo,monospace}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif}
.top{padding:14px 20px 10px;border-bottom:1px solid var(--line);background:var(--card)}
.eyebrow{font:600 11px/1 var(--mono);letter-spacing:.14em;text-transform:uppercase;color:var(--blue)}
h1{font-size:19px;margin:5px 0 8px}
.cfgrow{display:flex;gap:10px;flex-wrap:wrap;align-items:end;margin-top:6px}
.cfgrow .cap{font:600 10px/1 var(--mono);color:var(--mut);letter-spacing:.08em;text-transform:uppercase;width:64px;align-self:center}
.fld{display:flex;flex-direction:column;gap:2px}
.fld label{font:600 9.5px/1 var(--mono);letter-spacing:.05em;text-transform:uppercase;color:var(--mut)}
.fld input,.fld select{font:13px var(--mono);padding:4px 7px;border:1px solid var(--line);border-radius:6px;background:#fff;width:76px}
.fld select{width:auto;min-width:76px}
.kpi{margin-left:auto;background:#fafbfc;border:1px solid var(--line);border-radius:9px;padding:6px 14px;text-align:right}
.kpi .n{font:700 20px/1.2 var(--mono);color:var(--saved)}.kpi .t{color:var(--mut);font-size:10.5px}
.errbox{background:#fdf0ef;border:1px solid #ecc;border-radius:8px;color:#a33;font:12.5px/1.7 var(--mono);padding:8px 14px;margin-top:8px;display:none}
.wrap{padding:12px 20px 40px}
.tabs{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px}
.tab{padding:5px 13px;border:1px solid var(--line);border-radius:8px;background:#fff;cursor:pointer;font:600 12px var(--mono)}
.tab.on{background:var(--blue);color:#fff;border-color:var(--blue)}.tab.oom{border-color:#c0392b;color:#c0392b}.tab.on.oom{background:#c0392b;color:#fff}.tab.rsv{border-color:#e67e22;color:#e67e22}.tab.on.rsv{background:#e67e22;color:#fff}
.grid{display:grid;grid-template-columns:1fr 340px;gap:14px;align-items:start}
.maincol{display:flex;flex-direction:column;gap:14px;min-width:0}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden}
.card>header{padding:11px 15px;border-bottom:1px solid var(--line);font-weight:600;font-size:13.5px}
.gpane{max-height:56vh;overflow:auto;padding:10px 12px}
.lay{border:1px solid var(--line);border-radius:9px;margin-bottom:8px;overflow:hidden}
.lay>.hd{display:flex;align-items:center;gap:9px;padding:7px 11px;background:#fafbfc;cursor:pointer;font:12.5px var(--mono)}
.lay>.hd:hover{background:#f0f3f7}
.lay .car{color:var(--mut);width:10px}.lay .lt{font-weight:700}
.lay .am{margin-left:auto;color:var(--saved);font-weight:600}.lay .pm2{color:var(--mut);font-size:11px}
.lay .ops{display:none;padding:8px 12px 10px;position:relative}
.lay.open .ops{display:block}
.opn{position:relative;border-radius:7px;padding:6px 10px 5px;margin:0 0 4px 16px;color:#fff;cursor:pointer;font:11.5px/1.45 var(--mono)}
.opn:hover{outline:2px solid #111}
.opn .nm{font-weight:700}.opn .meta{opacity:.92;font-size:10.5px}
.opn .sv{color:#ffe08a;font-weight:700}.opn .tr{opacity:.75}
.opn::before{content:"";position:absolute;left:-11px;top:-6px;bottom:-6px;border-left:2px solid #ccd2da}
.opn:first-child::before{top:50%}.opn:last-child::before{bottom:50%}
.opn::after{content:"";position:absolute;left:-13px;top:50%;width:6px;height:6px;border-radius:50%;background:#aab2bd;transform:translateY(-50%)}
.cnode{cursor:pointer}.cnode.dim{opacity:.2}.cnode.hl rect{stroke:#111!important;stroke-width:3.2px!important}
.crcb{cursor:pointer}.crcb:hover circle{fill:#fff}
.edge{stroke:#c3c9d2;stroke-width:1.4;fill:none}
.edge.up{stroke:#2f4b7c;stroke-width:2.6}.edge.down{stroke:#e15759;stroke-width:2.6}.edge.dim{opacity:.25}
.tlpane{padding:10px 12px;max-height:60vh;overflow-y:auto}
.tlpane svg{display:block;width:100%;height:auto}
.tlblock{border:1px solid var(--line);border-radius:9px;margin-bottom:10px;overflow:hidden}
.tlblock.on{border-color:var(--blue);box-shadow:0 0 0 1px var(--blue)}
.tlblock>.th{display:flex;gap:10px;align-items:center;padding:6px 12px;background:#fafbfc;cursor:pointer;font:600 12px var(--mono)}
.tlblock>.th:hover{background:#f0f3f7}
.tlblock>.th .pk{margin-left:auto;color:var(--saved)}
.tlblock .tbody{padding:4px 8px 8px}
.desc{color:var(--mut);font-size:11.5px;margin:0 0 6px}
.dpane{max-height:calc(100vh - 240px);overflow:auto;padding:12px 14px}
.dpane .ph{color:var(--mut);font-size:12.5px}
.badge{display:inline-block;padding:2px 9px;border-radius:5px;font:700 12px/1.5 var(--mono);color:#fff}
.kv .k{color:var(--mut);font:600 10px/1 var(--mono);letter-spacing:.07em;text-transform:uppercase;margin-top:11px}
.kv .v{font:12px/1.5 var(--mono);word-break:break-all;margin-top:3px}
.tlist{list-style:none;padding:0;margin:3px 0 0}.tlist li{padding:2.5px 8px;margin-top:3px;border-radius:5px;background:#f2f4f7;font:11.5px/1.4 var(--mono);cursor:pointer}
.tlist li:hover{background:#e6ebf2}
.barrow{display:flex;align-items:center;gap:8px;padding:2.5px 0;font:11.5px var(--mono)}
.barrow .bl{width:112px;text-align:right;color:#333;flex-shrink:0}
.bartrack{flex:1;height:16px;background:#eef0f3;border-radius:4px;overflow:hidden}.barfill{height:100%}
.barrow .bv{width:76px;text-align:right;color:#555;flex-shrink:0;font-variant-numeric:tabular-nums}
.busy{opacity:.45;pointer-events:none}
.legend{font:10px var(--mono);color:var(--mut);padding:4px 15px 10px}.legend span{margin-right:9px;white-space:nowrap}
</style></head><body>
<div class="top">
  <div class="eyebrow">pynative-cost-evaluator · interactive v2</div>
  <h1>LLM 内存实验台 — 结构可配 · 切分自由 · op-DAG + timeline</h1>
  <div class="cfgrow"><span class="cap">模型结构</span>
    <div class="fld"><label>模型预设</label><select id="preset" name="preset" style="min-width:170px">
      <option value="custom">Custom（自定义）</option>
      <option value="dsv3_mini" selected>DSv3-mini（仓库锚点）</option>
      <option value="dsv3_671b">DeepSeek-V3 671B</option>
      <option value="dsv32_exp">DeepSeek-V3.2-Exp</option>
      <option value="dsv4_flash">DeepSeek-V4-Flash</option>
      <option value="dsv4_pro">DeepSeek-V4-Pro</option>
      <option value="glm5">GLM-5 (zai-org)</option>
    </select></div>
    <div class="fld"><label>yaml 导入</label><input type="file" id="yamlfile" accept=".yaml,.yml" style="font-size:11px;width:170px" title="选 mindformers 训练 yaml → 解析回填到对话框(不改动 yaml 文件本身)"></div>
    <div class="fld"><label>attn</label><select name="attn"><option value="mla" selected>MLA</option><option value="gqa">GQA</option><option value="mha">MHA</option><option value="dsa">DSA(预估计)</option><option value="dsv4_hybrid">DSv4-hybrid</option></select></div>
    <div class="fld"><label>layers</label><input name="layers" type="number" min="1" value="8"></div>
    <div class="fld"><label>dense 层数</label><input name="dense_k" type="number" min="0" value="1" title="前 K 层 dense,其余 MoE(first_k_dense_replace);=layers 则纯 dense"></div>
    <div class="fld"><label>experts</label><input name="experts" type="number" min="0" value="8"></div>
    <div class="fld"><label>topk</label><input name="topk" type="number" min="1" value="4"></div>
    <div class="fld"><label>heads</label><input name="heads" type="number" min="1" value="8"></div>
    <div class="fld"><label>kv_groups</label><input name="kv_groups" type="number" min="1" value="8" title="gqa 的 KV 组数(=heads 即 MHA)"></div>
    <div class="fld"><label>seq</label><input name="seq" type="number" min="1" value="4096"></div>
    <div class="fld"><label>batch</label><input name="batch" type="number" min="1" value="1"></div>
    <div class="fld"><label>mtp 层数</label><input name="mtp" type="number" min="0" value="0" title="MTP(num_nextn_predict_layers);计入可切分总层数(pp 分配的和=layers+mtp),位于层序列末端"></div>
  </div>
  <div class="cfgrow"><span class="cap">结构维度</span>
    <div class="fld"><label>hidden</label><input name="hidden" type="number" min="1" value="1792"></div>
    <div class="fld"><label>ffn</label><input name="ffn" type="number" min="1" value="3072"></div>
    <div class="fld"><label>moe_ffn</label><input name="moe_ffn" type="number" min="1" value="1024"></div>
    <div class="fld"><label>q_lora</label><input name="q_lora" type="number" min="1" value="1536"></div>
    <div class="fld"><label>kv_lora</label><input name="kv_lora" type="number" min="1" value="512"></div>
    <div class="fld"><label>qk_nope</label><input name="qk_nope" type="number" min="1" value="128"></div>
    <div class="fld"><label>qk_rope</label><input name="qk_rope" type="number" min="1" value="64"></div>
    <div class="fld"><label>v_head</label><input name="v_head" type="number" min="1" value="192"></div>
    <div class="fld"><label>vocab</label><input name="vocab" type="number" min="1" value="129280" style="width:90px"></div>
  </div>
  <div class="cfgrow"><span class="cap">并行切分</span>
    <div class="fld"><label>dp_shard</label><input name="dp" type="number" min="1" value="2"></div>
    <div class="fld"><label>tp</label><input name="tp" type="number" min="1" value="1"></div>
    <div class="fld"><label>ep</label><input name="ep" type="number" min="1" value="1"></div>
    <div class="fld"><label>pp</label><input name="pp" type="number" min="1" value="1"></div>
    <div class="fld"><label>pp 层分配</label><input name="pp_split" placeholder="如 3,5(空=均匀)" style="width:96px" title="每 stage 的可切分层数(transformer+mtp,mindformers num_layer_list 口径),段数=pp、和=layers+mtp;embedding/head 是伪层自动归 stage0/末 stage、不占配额也不计入显示层数"></div>
    <div class="fld"><label>vpp</label><input name="vpp" type="number" min="1" value="1" title="虚拟流水交错数(mindformers pp_interleave_num);>1 时每个物理 stage 持 vpp 个非连续 chunk(round-robin: 虚拟 stage=chunk*pp+rank),微批数需≥pp,更深 warmup→更多在飞激活。例:pp=2,vpp=2,8 层→stage0 持 chunk0(L1,2)+chunk2(L5,6),stage1 持 chunk1(L3,4)+chunk3(L7,8)"></div>
    <div class="fld"><label>microbatch</label><input name="mbs" placeholder="auto(=pp)" style="width:76px" title="流水微批数 num_microbatches;空=auto(pp>1 时取 pp)。m>pp 时 1F1B warmup/在飞深度随 m 分化"></div>
    <div class="fld"><label>cp</label><input name="cp" type="number" min="1" value="1"></div>
    <div class="fld"><label>cp 算法</label><select name="method"><option selected>colossal</option><option>ulysses</option><option>ring</option><option>hybrid</option></select></div>
    <div class="fld"><label>recompute</label><select name="recompute"><option value="None" selected>无</option><option value="full">full</option><option value="select">select(模块)</option><option value="custom">custom(图上选 op)</option></select></div>
    <div class="fld"><label>select 模块</label><select name="select"><option value="attn" selected>self_attn</option><option value="mlp">mlp</option><option value="both">both</option></select></div>
    <div class="fld"><label>重算层范围</label><input name="sel_layers" placeholder="1-8" style="width:64px" title="重算作用的层范围 a-b（1-N,含端点）——对 full / select / custom 均生效;空=全部层。例:full+「1-2」= 只前 2 层整层重算"></div>
    <div class="fld"><label>细粒度重算</label><input name="sel_cfg" placeholder="s0:both; s2-3:mlp   或   self_attention:0-3; flash:4-7" style="width:340px" title="统一入口,按段自动识别两种写法(不可混用),非空即优先于图上勾选/select 模块:&#10;① 按 PP stage —— s0:both; s1-2:self_attention; s3:none (stage→层跟当前 pp 切分)&#10;② 按绝对层号(mf select_module,0-indexed) —— self_attention:0-3; flash:4-7&#10;模式/pattern = none | self_attention | mlp | both(≈full) | 任意 op 名子串;分号分隔多条"></div>
    <input type="hidden" name="sel_ops" value="">
    <div class="kpi"><div class="n" id="kpeak">—</div><div class="t" id="kmeta">设备峰值</div></div>
  </div>
  <div class="cfgrow"><span class="cap">运行时/硬件</span>
    <div class="fld"><label>dp_replicate</label><input name="dp_replicate" type="number" min="1" value="1" title="纯数据并行度(权重/优化器逐 rank 复制、不切分);单卡峰值与 dp_shard=1 相同,进 world/HCCL 域。yaml 导入按解析值回填"></div>
    <div class="fld"><label>reshard</label><select name="reshard" title="reshard_after_forward_policy:default(PP 整体不 reshard,非 PP 除 output 均前向后即 reshard) / always(前向后即 reshard,反向 re-gather) / never(unsharded 权重驻留至本模块反向) —— 改 gather 生命周期(fsdp=dp_shard·cp>1 时生效)"><option value="default" selected>default</option><option value="always">always</option><option value="never">never</option></select></div>
    <div class="fld"><label>cpu_offload</label><select name="cpu_offload" title="参数/优化器状态卸载 CPU:开 → 该 stage 持久态=0、优化器 step 无设备瞬态"><option value="0" selected>关</option><option value="1">开</option></select></div>
    <div class="fld"><label>prefetch</label><input name="prefetch" type="number" min="0" value="1" title="FSDP 参数预取深度(prefetch_depth);0=无预取(单缓冲),≥1=下 N 层双缓冲。yaml 导入按解析值回填"></div>
    <div class="fld"><label>设备容量(GiB)</label><input name="maxdev_gib" type="number" min="1" step="1" value="64" title="设备 HBM 容量(HardwareSpec.max_device_memory);OOM 判据用它。yaml 导入按 context.max_device_memory 回填(缺省 54GiB),手配默认 64GiB"></div>
    <div class="fld"><label>优化器 dtype</label><select name="opt_dtype" title="AdamW params dtype:fp32(state=master+m+v=12B/param) / bf16(+compute 副本 2B=14B/param)。yaml 导入按 model.params_dtype 回填"><option value="fp32" selected>fp32</option><option value="bf16">bf16</option></select></div>
    <input type="hidden" name="sp" value="">
    <input type="hidden" name="grad_bytes" value="4">
  </div>
  <div class="cfgrow" id="rcrow" style="display:none"><span class="cap">重算 op</span><div id="rcchips" style="font:11.5px var(--mono);color:var(--mut)">（在左图 op 节点上点 <b>↻</b> 勾选;再点取消）</div></div>
  <div class="errbox" id="err"></div>
</div>
<div class="wrap">
  <div class="tabs" id="tabs"></div>
  <div class="grid">
    <div class="maincol">
      <div class="card"><header id="ghdr">模型结构 · op-DAG（点层展开）</header><div class="gpane" id="gpane"></div></div>
      <div class="card"><header id="tlhdr">内存时间线（FWD→BWD,按时间顺序）</header>
        <div class="tlpane"><p class="desc" id="tldesc"></p><div id="tl"></div></div>
        <div class="legend" id="leg"></div></div>
    </div>
    <div class="card"><header id="dhdr">详情</header><div class="dpane" id="detail"><p class="ph">悬停/点击左侧算子 → 算子详情（存的激活/上下游）；点 timeline → 该刻各桶。</p></div></div>
  </div>
</div>
<script>
const PRESETS=__PRESETS__;
const OPC={matmul:"#4e79a7",flash_attn:"#e15759",elementwise:"#b07aa1",norm:"#59a14f",rope:"#8cd17d",
  moe_router:"#f9a825",moe_gemm:"#2f4b7c",dispatch:"#76b7b2",combine:"#76b7b2",embedding:"#7cae60",
  dsa:"#e15759",csa:"#e15759",hca:"#e15759"};
const BKC={persistent:"#6b6b6b",act_live:"#4e79a7",kept_frag:"#c0392b",gather_buf:"#59a14f",grad_buf:"#f28e2b",recomp_scratch:"#b07aa1",bwd_scratch:"#e15759",bwd_working_set:"#8cd17d",swap_buf:"#76b7b2",workspace:"#bab0ac",optstep:"#ff9da7",framework:"#d7d7d7"};
const BKD={persistent:"参数+优化器状态",act_live:"存活激活",kept_frag:"B margin(保留-MoE碎片)",gather_buf:"FSDP all-gather",grad_buf:"梯度缓冲",recomp_scratch:"full重算重物化",bwd_scratch:"反向临时(loss fp32)",bwd_working_set:"无重算反向工作集",swap_buf:"激活swap",workspace:"算子workspace",optstep:"优化器step",framework:"框架"};
let cur=null,curStage=0,openLayers=new Set();
function esc(s){return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");}
function fmib(m){return m>=1024?(m/1024).toFixed(1)+" GiB":m.toFixed(0)+" MiB";}
function qs(){const o={};document.querySelectorAll(".top [name]").forEach(e=>o[e.name]=e.value);return o;}
let timer=null;
function refreshSoon(){clearTimeout(timer);timer=setTimeout(refresh,350);}
async function refresh(){
  document.querySelector(".wrap").classList.add("busy");
  let d;
  try{const r=await fetch("/api/eval?"+new URLSearchParams(qs()));d=await r.json();}
  catch(e){d={ok:false,errors:["服务不可达: "+e]};}
  document.querySelector(".wrap").classList.remove("busy");
  const eb=document.getElementById("err");
  if(!d.ok){eb.style.display="block";eb.innerHTML="✗ 配置不合法/评估失败：<br>· "+d.errors.map(esc).join("<br>· ");
    document.getElementById("kpeak").textContent="—";return;}
  eb.style.display="none";
  cur=d;curStage=d.tightest;openLayers=new Set();
  document.getElementById("kpeak").textContent=fmib(d.device_peak);
  document.getElementById("kmeta").textContent=`设备峰值 · world=${d.world} · hccl(reserved)+${d.hccl_mib}M · reserved余量 ${d.reserved_margin_mib}M`;
  document.getElementById("tabs").innerHTML=d.stages.map(s=>`<div class="tab ${s.stage===curStage?"on":""} ${s.oom?"oom":(s.reserved_oom?"rsv":"")}" data-s="${s.stage}">Stage ${s.stage} · ${fmib(s.peak)}${s.oom?" ⚠OOM":(s.reserved_oom?" ⚠reserved":"")}<span style="color:${s.stage===curStage?'#dde':'#999'};font-weight:400"> · ${s.n_layers}层${s.extras&&s.extras.length?" +"+s.extras.map(e=>e==="embedding"?"emb":e==="lm_head"?"head":e).join("+"):""}</span></div>`).join("");
  document.querySelectorAll(".tab").forEach(t=>t.addEventListener("click",()=>{curStage=+t.dataset.s;openLayers=new Set();drawStage();}));
  drawStage();
}
function drawStage(){
  document.querySelectorAll(".tab").forEach(t=>t.classList.toggle("on",+t.dataset.s===curStage));
  const st=cur.stages.find(s=>s.stage===curStage);
  drawGraph(st); drawTimeline(st);
  const ev=st.timeline, pi=ev.findIndex(z=>z.total===st.peak);
  showBuckets(ev[pi>=0?pi:0]);
}
/* ── ③ 模型结构 op-DAG（逐层可展开;真 SVG DAG:拓扑分层+连线箭头,分支/汇合可见）── */
function cellLayout(ops,edges){
  // 拓扑最长路分层:lv=深度(行,竖向);同 lv 并排(lane,横向)→ 分支支路可见。
  const succ={},ind={}; ops.forEach(o=>ind[o.i]=0);
  edges.forEach(([a,b])=>{(succ[a]=succ[a]||[]).push(b);ind[b]++;});
  const lv={}; ops.forEach(o=>lv[o.i]=0);
  const q=ops.filter(o=>ind[o.i]===0).map(o=>o.i), ind2=Object.assign({},ind);
  while(q.length){const u=q.shift();(succ[u]||[]).forEach(v=>{lv[v]=Math.max(lv[v],lv[u]+1);if(--ind2[v]===0)q.push(v);});}
  const lanes={}, pos={};
  ops.forEach(o=>{const l=lv[o.i];lanes[l]=lanes[l]||0;pos[o.i]={lv:l,lane:lanes[l]++};});
  return {pos,maxlv:Math.max(...ops.map(o=>lv[o.i]),0),maxlane:Math.max(...Object.values(lanes),1)};
}
function cellSvg(L){
  const NW=196,NH=48,CW=214,RH=68,MX=8,MY=8;
  const {pos,maxlv,maxlane}=cellLayout(L.ops,L.edges);
  const W=MX*2+(maxlane)*CW+NW-CW+CW, H=MY*2+maxlv*RH+NH;
  const X=i=>MX+pos[i].lane*CW, Y=i=>MY+pos[i].lv*RH;
  let s=`<svg width="${Math.max(W,420)}" height="${H}" viewBox="0 0 ${Math.max(W,420)} ${H}" xmlns="http://www.w3.org/2000/svg" font-family="var(--mono)" font-size="10.5">`;
  s+=`<defs><marker id="ar${L.id}" markerWidth="7" markerHeight="7" refX="6" refY="3" orient="auto"><path d="M0,0L6,3L0,6Z" fill="#9aa2ad"/></marker></defs>`;
  // 真实 edges:producer 底边中点 → consumer 顶边中点(贝塞尔+箭头)
  L.edges.forEach(([a,b])=>{
    const x1=X(a)+NW/2,y1=Y(a)+NH,x2=X(b)+NW/2,y2=Y(b);
    s+=`<path class="edge" id="ce_${L.id}_${a}_${b}" d="M${x1},${y1} C${x1},${y1+22} ${x2},${y2-22} ${x2},${y2}" marker-end="url(#ar${L.id})"/>`;});
  L.ops.forEach(o=>{
    const c=OPC[o.type]||"#8b93a0";
    const applied=o.recomp===true;                    // 后端权威:当前配置下此 op 被重算(saves 不存)
    const rcOn=applied||customOps.has(o.name);         // 虚线标记:已生效重算 或 交互勾选(custom)
    const x=X(o.i),y=Y(o.i);
    const tag=applied?`↻重算·不存${o.recomp_mib>0?" 省"+o.recomp_mib+"M":""}`
             :(o.act_mib>0?`💾 ${o.act_mib}M`:(o.param_mib>0?`⚙ ${o.param_mib}M`:"↻ transient"));
    const tagc=applied?"#e6c9f5":(o.act_mib>0?"#ffe08a":"#e8e8e8");
    const stroke=applied?"#d9a7ec":(o.act_mib>0?"#c0392b":"#8d949e");
    s+=`<g class="cnode ${rcOn?"rc":""}" data-l="${L.id}" data-i="${o.i}">`;
    s+=`<rect x="${x}" y="${y}" width="${NW}" height="${NH}" rx="7" fill="${c}" fill-opacity="${applied?0.62:0.88}" stroke="${stroke}" stroke-width="${(applied||o.act_mib>0)?2.5:1}"${rcOn?' stroke-dasharray="5 3"':''}/>`;
    s+=`<text x="${x+8}" y="${y+15}" fill="#fff" font-weight="bold">${esc(o.name)} <tspan font-weight="normal" fill-opacity=".85">${esc(o.type)}</tspan></text>`;
    s+=`<text x="${x+8}" y="${y+29}" fill="#f0f0f0" font-size="9.5">→${esc(o.out.name)}:(${esc(o.out.sym)})</text>`;
    s+=`<text x="${x+8}" y="${y+42}" fill="${tagc}" font-size="9.5">${esc(tag)}</text>`;
    s+=`<g class="crcb" data-op="${esc(o.name)}"><circle cx="${x+NW-13}" cy="${y+13}" r="9" fill="${customOps.has(o.name)?"#fff":"rgba(255,255,255,.28)"}"/><text x="${x+NW-13}" y="${y+17}" text-anchor="middle" font-weight="bold" fill="${customOps.has(o.name)?"#c0392b":"#fff"}">↻</text></g></g>`;});
  return s+`</svg>`;
}
function drawGraph(st){
  document.getElementById("ghdr").textContent=`模型结构 · Stage ${st.stage}（${st.n_layers} 可切分层${st.extras&&st.extras.length?" + "+st.extras.map(e=>e==="embedding"?"embedding":e==="lm_head"?"head":e).join("/")+"(伪层,不占配额)":""},点层展开 op-DAG）`;
  if(openLayers.size===0){const seen=new Set();st.graph.forEach(L=>{if(!seen.has(L.type)){seen.add(L.type);openLayers.add(L.id);}});}
  document.getElementById("gpane").innerHTML=st.graph.map((L,idx)=>{
    const open=openLayers.has(L.id);
    const opsH=open?`<div class="ops" style="display:block;overflow-x:auto">${cellSvg(L)}</div>`:"";
    const conn=idx<st.graph.length-1?`<div style="text-align:center;color:#9aa2ad;font:12px var(--mono);line-height:1">↓</div>`:"";
    const rcOnL=(L.recomp&&L.recomp!=="none");
    const amH=rcOnL
      ? `<span class="am" style="color:#8e44ad" title="重算态:仅存下方标注为「存」的激活;被重算 op 的 saves 前向不存(反向重物化)。层入口锚点=上一层输出,重算必须保留">↻${L.recomp==="full"?"全重算":"选择性重算"} · 存 ${L.act_mib} MiB${L.entry_mib>0?"（层入口锚点 "+L.entry_mib+"M）":""} / 全量 ${L.full_act_mib}</span>`
      : `<span class="am">激活 ${L.act_mib} MiB</span>`;
    return `<div class="lay ${open?"open":""}" data-l="${L.id}"><div class="hd" data-l="${L.id}"><span class="car">${open?"▾":"▸"}</span><span class="lt">L${L.id} ${esc(L.type)}</span><span class="pm2">${L.ops.length} ops · ${L.edges.length} edges</span>${amH}</div>${opsH}</div>${conn}`;
  }).join("");
  document.querySelectorAll(".lay>.hd").forEach(h=>h.addEventListener("click",()=>{const id=+h.dataset.l;openLayers.has(id)?openLayers.delete(id):openLayers.add(id);drawGraph(st);}));
  document.querySelectorAll(".cnode").forEach(el=>{
    el.addEventListener("mouseenter",()=>hiOp(st,+el.dataset.l,+el.dataset.i));
    el.addEventListener("click",()=>hiOp(st,+el.dataset.l,+el.dataset.i));
  });
  document.querySelectorAll(".crcb").forEach(b=>b.addEventListener("click",e=>{
    e.stopPropagation(); toggleRc(b.dataset.op);}));
}
/* ── ② 细粒度重算:任意 op 勾选(与 mindformers select_recompute 同口径) ── */
let customOps=new Set();
function toggleRc(op){
  customOps.has(op)?customOps.delete(op):customOps.add(op);
  document.querySelector("[name=sel_ops]").value=[...customOps].join(",");
  if(customOps.size&&document.querySelector("[name=recompute]").value!=="custom")
    document.querySelector("[name=recompute]").value="custom";
  syncChips(); refreshSoon();
}
function syncChips(){
  const row=document.getElementById("rcrow");
  const isC=document.querySelector("[name=recompute]").value==="custom";
  row.style.display=isC?"flex":"none";
  if(!isC)return;
  document.getElementById("rcchips").innerHTML=customOps.size
    ? [...customOps].map(o=>`<span style="display:inline-block;background:#eaf0f8;border:1px solid #c8d4e6;border-radius:6px;padding:2px 8px;margin:0 5px 3px 0;cursor:pointer" data-rm="${esc(o)}">↻ ${esc(o)} ✕</span>`).join("")
      +`<span style="color:#999">（点 chip 移除;作用层范围见上方「重算层范围」）</span>`
    : "（在左图 op 节点上点 <b>↻</b> 勾选;再点取消）";
  document.querySelectorAll("#rcchips [data-rm]").forEach(c=>c.addEventListener("click",()=>toggleRc(c.dataset.rm)));
}
function hiOp(st,lid,i){
  const L=st.graph.find(x=>x.id===lid), o=L.ops[i];
  const pred=L.edges.filter(e=>e[1]===i).map(e=>e[0]), succ=L.edges.filter(e=>e[0]===i).map(e=>e[1]);
  const keep=new Set([i,...pred,...succ]);
  document.querySelectorAll(".cnode").forEach(el=>{el.classList.remove("hl","dim");
    if(+el.dataset.l!==lid)return; const j=+el.dataset.i;
    if(j===i)el.classList.add("hl"); else if(!keep.has(j))el.classList.add("dim");});
  document.querySelectorAll(".edge").forEach(e=>e.classList.remove("up","down","dim"));
  L.edges.forEach(([a,b])=>{const e=document.getElementById(`ce_${lid}_${a}_${b}`);if(!e)return;
    if(b===i)e.classList.add("up"); else if(a===i)e.classList.add("down"); else e.classList.add("dim");});
  const c=OPC[o.type]||"#8b93a0";
  const actsH=o.acts.length?o.acts.map(a=>a.stored===false
      ?`<div style="margin-bottom:4px;opacity:.72"><span style="color:#8e44ad;font-weight:700">↻ ${a.mib} MiB 不存</span> = ${esc(a.name)} <span style="color:#555">${esc(a.calc)}</span></div>`
      :`<div style="margin-bottom:4px"><span style="color:#c0392b;font-weight:700">💾 ${a.mib} MiB</span> = ${esc(a.name)} <span style="color:#555">${esc(a.calc)}</span></div>`).join(""):'<span style="color:#999">（反向不存激活）</span>';
  const rcNote=o.recomp?`<div style="color:#8e44ad;margin-bottom:5px;font-weight:600">↻ 本 op 被重算：前向不存 saves、反向重物化${o.recomp_mib>0?"（省 "+o.recomp_mib+" MiB）":""}</div>`:"";
  const U=pred.length?pred.map(j=>`<li data-l="${lid}" data-g="${j}">↑ ${esc(L.ops[j].name)} (${esc(L.ops[j].type)})</li>`).join(""):'<li style="color:#999;cursor:default">（层输入）</li>';
  const D=succ.length?succ.map(j=>`<li data-l="${lid}" data-g="${j}">↓ ${esc(L.ops[j].name)} (${esc(L.ops[j].type)})</li>`).join(""):'<li style="color:#999;cursor:default">（层输出）</li>';
  document.getElementById("dhdr").textContent="算子详情";
  document.getElementById("detail").innerHTML=`<span class="badge" style="background:${c}">${esc(o.name)}</span> <span style="font:600 11px var(--mono);color:var(--blue)">${esc(o.type)} · L${lid} ${esc(L.type)}</span>
    <div class="kv">
    <div class="k">要存的激活（切分后）</div><div class="v">${rcNote}${actsH}</div>
    <div class="k">输出</div><div class="v">${esc(o.out.name)} · ${o.out.mib} MiB<br><span style="color:#555">${esc(o.out.calc)}</span></div>
    <div class="k">输入</div><div class="v">${o.ins.map(esc).join(", ")||"—"}</div>
    ${o.param_mib?`<div class="k">参数(持久,切分后)</div><div class="v">⚙ ${o.param_mib} MiB</div>`:""}
    ${o.ws_mib?`<div class="k">workspace</div><div class="v">${o.ws_mib} MiB</div>`:""}
    <div class="k">上游算子</div><ul class="tlist">${U}</ul>
    <div class="k">下游算子</div><ul class="tlist">${D}</ul></div>`;
  document.querySelectorAll("#detail li[data-g]").forEach(li=>li.addEventListener("click",()=>hiOp(st,+li.dataset.l,+li.dataset.g)));
}
/* ── ④ timeline 堆叠面积图（时间顺序,可点）── */
function drawTimeline(_){
  // 2026-07-11:全部 stage 的 timeline 同时展示(用户要求"都要展示");当前 stage 块高亮,
  // 块标题可点(联动切 stage/左图),每块图内可点事件看该刻各桶。
  document.getElementById("tlhdr").textContent=`内存时间线 · 全部 ${cur.stages.length} 个 stage（FWD→BWD,点图看该刻各桶）`;
  document.getElementById("tldesc").innerHTML=cur.stages.length>1?`蓝框 = 当前选中 stage（与左侧结构图联动;点任一块标题切换）`:``;
  document.getElementById("tl").innerHTML=cur.stages.map(st=>{
    return `<div class="tlblock ${st.stage===curStage?"on":""}" data-s="${st.stage}">
      <div class="th" data-s="${st.stage}">Stage ${st.stage} <span style="color:#888;font-weight:400">${esc(st.layers_desc)}</span><span class="pk">峰值 ${fmib(st.peak)} @ ${esc(st.peak_event)}${st.oom?" ⚠OOM":(st.reserved_oom?" ⚠reserved":"")}</span></div>
      <div class="tbody">${tlSvg(st)}</div></div>`;
  }).join("");
  cur.stages.forEach(st=>bindTl(st));
  document.getElementById("leg").innerHTML=Object.keys(BKC).map(k=>`<span><span style="display:inline-block;width:8px;height:8px;background:${BKC[k]};border-radius:2px"></span>${k}</span>`).join("");
  document.querySelectorAll(".tlblock>.th").forEach(h=>h.addEventListener("click",()=>{curStage=+h.dataset.s;openLayers=new Set();drawStage();}));
}
function tlSvg(st){
  const ev=st.timeline,n=ev.length;
  const W=640,H=210,PL=52,PR=10,PT=8,PB=40,pw=W-PL-PR,ph=H-PT-PB;
  const ymax=Math.max(...ev.map(e=>e.total))*1.06;
  const X=i=>PL+(n<2?0:i/(n-1)*pw), Y=v=>PT+ph-v/ymax*ph;
  let s=`<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" font-family="var(--mono)" font-size="9" id="tlsvg_${st.stage}">`;
  for(let k=0;k<4;k++){const v=ymax*k/3;s+=`<line x1="${PL}" y1="${Y(v)}" x2="${PL+pw}" y2="${Y(v)}" stroke="#ececec"/><text x="${PL-5}" y="${Y(v)+3}" text-anchor="end" fill="#999">${(v/1024).toFixed(0)}G</text>`;}
  let base=new Array(n).fill(0);
  Object.keys(BKC).forEach(k=>{
    const vals=ev.map(e=>e.buckets[k]||0); if(!vals.some(v=>v>0))return;
    const top=vals.map((v,i)=>base[i]+v);
    let pts=top.map((v,i)=>`${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join(" ");
    pts+=" "+base.slice().reverse().map((v,i)=>`${X(n-1-i).toFixed(1)},${Y(v).toFixed(1)}`).join(" ");
    s+=`<polygon points="${pts}" fill="${BKC[k]}" fill-opacity=".85"/>`; base=top;});
  s+=`<polyline points="${ev.map((e,i)=>`${X(i).toFixed(1)},${Y(e.total).toFixed(1)}`).join(" ")}" fill="none" stroke="#111" stroke-width="1.4"/>`;
  const pi=ev.findIndex(e=>e.total===st.peak);
  if(pi>=0)s+=`<circle cx="${X(pi)}" cy="${Y(ev[pi].total)}" r="3.5" fill="#c0392b"/><text x="${X(pi)+5}" y="${Y(ev[pi].total)-4}" fill="#c0392b" font-weight="bold">★${esc(ev[pi].event)}</text>`;
  const step=Math.max(1,Math.floor(n/10));
  for(let i=0;i<n;i+=step)s+=`<text x="${X(i)}" y="${PT+ph+13}" text-anchor="middle" fill="#999" transform="rotate(32 ${X(i)} ${PT+ph+13})">${esc(ev[i].event)}</text>`;
  s+=`<line id="cursor_${st.stage}" x1="-9" y1="${PT}" x2="-9" y2="${PT+ph}" stroke="#111" stroke-dasharray="3 3"/><rect x="${PL}" y="${PT}" width="${pw}" height="${ph}" fill="transparent" style="cursor:crosshair" id="tlhit_${st.stage}"/></svg>`;
  return s;
}
function bindTl(st){
  const ev=st.timeline,n=ev.length;
  const W=640,PL=52,PR=10,pw=W-PL-PR;
  const svg=document.getElementById(`tlsvg_${st.stage}`),hit=document.getElementById(`tlhit_${st.stage}`);
  if(!svg||!hit)return;
  function pick(evt){const r=svg.getBoundingClientRect();const fx=(evt.clientX-r.left)/r.width*W;
    let i=Math.round((fx-PL)/(n<2?1:pw/(n-1)));i=Math.max(0,Math.min(n-1,i));
    const c=document.getElementById(`cursor_${st.stage}`);
    const X=PL+(n<2?0:i/(n-1)*pw); c.setAttribute("x1",X);c.setAttribute("x2",X);
    document.getElementById("dhdr").textContent=`各桶开销 · Stage ${st.stage}`;
    showBuckets(ev[i]);}
  hit.addEventListener("click",e=>{e.stopPropagation();pick(e);});
  hit.addEventListener("mousemove",e=>{if(e.buttons)pick(e);});
}
function showBuckets(e){
  const bs=Object.entries(e.buckets).sort((a,b)=>b[1]-a[1]);const mx=Math.max(...bs.map(x=>x[1]),1);
  document.getElementById("dhdr").textContent=`各桶开销 · ${e.event}`;
  document.getElementById("detail").innerHTML=`<div style="font:600 12px var(--mono);margin-bottom:6px">事件 <b>${esc(e.event)}</b> · 总 ${fmib(e.total)}</div>`+
    bs.map(([k,v])=>`<div class="barrow"><span class="bl">${k}</span><span class="bartrack"><span class="barfill" style="width:${v/mx*100}%;background:${BKC[k]||'#ccc'}"></span></span><span class="bv">${v.toFixed(0)}·${(v/e.total*100).toFixed(0)}%</span></div><div style="font-size:10px;color:#999;margin:-2px 0 3px 120px">${BKD[k]||""}</div>`).join("")+
    `<p style="color:#999;font-size:11px;margin-top:10px">悬停左侧算子可切回算子详情。</p>`;
}
/* 运行时/硬件 extra → 手配默认(64GiB/AdamW-fp32/dp_replicate=1/reshard=default/offload 关/prefetch=1)。
   选模型预设=手配路径 → 复位这些 extra,不残留上次 yaml 导入的解析值(非导入路径保持现状)。*/
const RT_DEFAULTS={dp_replicate:"1",reshard:"default",cpu_offload:"0",prefetch:"1",maxdev_gib:"64",opt_dtype:"fp32",sp:"",grad_bytes:"4"};
function resetRuntimeExtras(){
  Object.entries(RT_DEFAULTS).forEach(([k,v])=>{const el=document.querySelector(`.top [name=${k}]`);if(el)el.value=v;});
}
/* 模型预设:选中即填充结构字段(HF config.json 值),用户仍可手改覆盖 */
function applyPreset(key){
  const pr=PRESETS[key]; if(!pr)return;
  Object.entries(pr.ui).forEach(([k,v])=>{const el=document.querySelector(`.top [name=${k}]`);if(el)el.value=v;});
  resetRuntimeExtras();      // 手配路径 → extra 回到固定假设(64GiB/AdamW-fp32/...)
  document.getElementById("kmeta").textContent="来源: "+pr.source;
}
document.getElementById("preset").addEventListener("change",e=>{applyPreset(e.target.value);syncChips();refreshSoon();});
/* yaml 导入:解析 mindformers 训练配置 → 回填对话框(不改 yaml 文件本身) */
document.getElementById("yamlfile").addEventListener("change",async e=>{
  const f=e.target.files[0]; if(!f)return;
  const txt=await f.text();
  let d; try{const r=await fetch("/api/parse_yaml",{method:"POST",body:JSON.stringify({yaml:txt,defaults:qs()})}); d=await r.json();}
  catch(err){d={ok:false,error:String(err)};}
  const eb=document.getElementById("err");
  if(!d.ok){eb.style.display="block";eb.innerHTML="✗ yaml 解析失败: "+esc(d.error);e.target.value="";return;}
  if(d.warnings&&d.warnings.length){eb.style.display="block";eb.innerHTML="⚠ yaml 导入警告:<br>· "+d.warnings.map(esc).join("<br>· ");}
  else eb.style.display="none";
  Object.entries(d.fields).forEach(([k,v])=>{const el=document.querySelector(`.top [name=${k}]`);if(el&&v!==null&&v!==undefined)el.value=v;});
  document.getElementById("preset").value="custom";
  document.getElementById("kmeta").textContent="来源: yaml 导入("+f.name+"),已按完整解析值回填并评估(dp_replicate/reshard/offload/prefetch/设备容量/优化器 dtype 均生效,可改;不写回文件)";
  e.target.value="";       // 允许重选同一文件
  syncChips();refreshSoon();
});
/* 结构字段被手改 → 已偏离预设 → 下拉自动跳 Custom（并行/重算字段不算偏离） */
const STRUCT_FIELDS=["attn","layers","dense_k","experts","topk","heads","kv_groups","seq","batch",
  "hidden","ffn","moe_ffn","q_lora","kv_lora","qk_nope","qk_rope","v_head","vocab"];
function markCustom(){const sel=document.getElementById("preset");if(sel.value!=="custom"){sel.value="custom";
  document.getElementById("kmeta").textContent="来源: "+PRESETS.custom.source;}}
document.querySelectorAll(".top [name]").forEach(e=>{if(e.id==="preset")return;
  e.addEventListener("change",()=>{if(STRUCT_FIELDS.includes(e.name))markCustom();syncChips();refreshSoon();});});
document.querySelectorAll(".top input[type=number]").forEach(e=>e.addEventListener("input",()=>{
  if(STRUCT_FIELDS.includes(e.name))markCustom();refreshSoon();}));
refresh();
</script></body></html>"""


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    srv = HTTPServer(("127.0.0.1", port), H)
    print(f"内存实验台 v2 → http://127.0.0.1:{port}   (Ctrl-C 退出)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
