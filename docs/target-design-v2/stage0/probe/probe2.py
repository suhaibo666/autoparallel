"""补测: 严格 guard 口径 / 配置规范化函数 / 间接层机制 / 去冗余 PySub 曲线 /
PolicyStateBinding 精确规模。"""
import ast
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.setrecursionlimit(20000)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import astprobe as ap  # noqa: E402
from run_all import (REPOS, FRAMEWORK_PREFIX, MODEL_INCLUDE, MODEL_EXCLUDE_SUBSTR, sel)  # noqa: E402

OUT = os.path.join(HERE, 'out')
W = sys.stdout.write

# --------------------------------------------------- (a) TIGHT guard 口径
# 去掉最松的模式; 并要求匹配到的标识符经由 config-ish 基座 或 是明确的并行度记号
TIGHT_GROUPS = {
    '并行度': [r'tensor_model_parallel', r'pipeline_model_parallel', r'expert_model_parallel',
            r'expert_tensor_parallel', r'context_parallel', r'sequence_parallel',
            r'data_parallel', r'^model_parallel', r'hierarchical_context_parallel',
            r'^tp$', r'^pp$', r'^ep$', r'^cp$', r'^dp$', r'^etp$',
            r'^tp_size$', r'^pp_size$', r'^ep_size$', r'^cp_size$', r'^dp_size$',
            r'world_size$', r'^tp_group$', r'^tp_comm', r'moe_extended_tp'],
    '重算': [r'recompute', r'gradient_checkpointing', r'activation_offload', r'cpu_offloading',
           r'^checkpoint_granularity', r'activation_checkpoint'],
    '融合/实现选择': [r'fusion', r'^fused_', r'_fused$', r'use_flash', r'flash_attn',
                r'attention_backend', r'_impl$', r'grouped_gemm', r'^use_te$', r'^use_te_',
                r'transformer_impl', r'^HAVE_', r'cuda_graph'],
    '精度': [r'^fp8', r'fp8_', r'^bf16$', r'^fp16$', r'params_dtype', r'compute_dtype',
           r'master_weights', r'main_grad', r'^dtype$', r'quantization_config', r'quant_method',
           r'quant_config'],
    '分布式优化器': [r'zero_stage', r'use_distributed_optimizer', r'^overlap_', r'_overlap$',
                r'bucket_size', r'ddp_config', r'grad_reduce', r'^fsdp', r'reduce_scatter_'],
    '调度': [r'virtual_pipeline', r'num_microbatches', r'microbatch', r'interleav',
           r'pipeline_dtype', r'first_stage', r'last_stage', r'is_pipeline'],
    'MoE': [r'^moe_', r'_moe$', r'num_experts', r'^expert_num$', r'num_local_experts',
            r'moe_router', r'token_dispatcher', r'^n_routed', r'shared_expert_intermediate',
            r'use_shared_expert', r'moe_layer_pattern', r'moe_layer_freq'],
}
_T = {g: re.compile('|'.join(p)) for g, p in TIGHT_GROUPS.items()}

CONFIG_ROOTS = {'self', 'config', 'cfg', 'args', 'parallel_config', 'moe_config',
                'model_config', 'transformer_config', 'op_config', 'quant_config',
                'ddp_config', 'optimizer_config', 'training_config', 'cls', 'spec',
                'submodules', 'params', 'kwargs'}


def _tight_hits(test):
    """返回 (groups, fields)。要求: Attribute 的 attr 匹配且根为 config-ish;
    或者 Name 匹配(裸并行度记号/配置参数名)。"""
    groups, fields = set(), set()
    for n in ast.walk(test):
        ident, ok = None, False
        if isinstance(n, ast.Attribute):
            ident = n.attr
            root = ap._fmt_target(n)
            r0 = root.split('.')[0] if root else None
            ok = (r0 in CONFIG_ROOTS) or (root is None)
        elif isinstance(n, ast.Name):
            ident = n.id
            ok = True
        elif isinstance(n, ast.Call):
            f = ap._fmt_target(n.func)
            if f:
                ident = f.rsplit('.', 1)[-1]
                ok = True
        if not (ident and ok):
            continue
        for g, rx in _T.items():
            if rx.search(ident):
                groups.add(g)
                fields.add(ident)
    return groups, fields


def tight_guards(parsed):
    out = []
    for rel, (tree, src, nl) in parsed.items():
        def walk(node, depth, poldepth, fq):
            for c in ast.iter_child_nodes(node):
                nf = fq
                if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    nf = (fq + '.' + c.name) if fq else c.name
                if isinstance(c, ast.If):
                    g, f = _tight_hits(c.test)
                    if g:
                        bc, ec = ap._calls_in(c.body), ap._calls_in(c.orelse)
                        bk, ek = ap._branch_kind(c.body), ap._branch_kind(c.orelse)
                        only_assign = all(isinstance(s, (ast.Assign, ast.AnnAssign,
                                                         ast.AugAssign, ast.Pass))
                                          for s in list(c.body) + list(c.orelse))
                        delta = set(bc) != set(ec)
                        struct = ('single_sided' if ek == 'empty'
                                  else 'one_side_trivial' if bk in ('early_exit', 'pass_only')
                                  or ek in ('early_exit', 'pass_only') else 'both_nonempty')
                        cls = ('assign_only' if only_assign and not delta
                               else 'assign_only_with_call_delta' if only_assign
                               else 'call_set_changes' if delta else 'same_calls_diff_args')
                        out.append({'file': rel, 'lineno': c.lineno, 'node': 'If', 'func': nf,
                                    'groups': sorted(g), 'fields': sorted(f)[:6],
                                    'struct': struct, 'cls': cls, 'if_depth': depth,
                                    'pol_depth': poldepth,
                                    'test_src': ap._safe_unparse(c.test)})
                    walk(c, depth + 1, poldepth + (1 if g else 0), nf)
                elif isinstance(c, ast.IfExp):
                    g, f = _tight_hits(c.test)
                    if g:
                        bc = ap._calls_in([ast.Expr(value=c.body)])
                        ec = ap._calls_in([ast.Expr(value=c.orelse)])
                        out.append({'file': rel, 'lineno': getattr(c, 'lineno', 0),
                                    'node': 'IfExp', 'func': nf, 'groups': sorted(g),
                                    'fields': sorted(f)[:6], 'struct': 'ifexp',
                                    'cls': 'call_set_changes' if set(bc) != set(ec) else 'assign_only',
                                    'if_depth': depth, 'pol_depth': poldepth,
                                    'test_src': ap._safe_unparse(c.test)})
                    walk(c, depth, poldepth, nf)
                else:
                    walk(c, depth, poldepth, nf)
        walk(tree, 0, 0, '')
    return out


# ------------------------------------------- (b) 配置规范化函数
CFG_FN_NAMES = re.compile(
    r'^(validate|__post_init__|_validate|sanity_check|.*_sanity|core_transformer_config_from_.*|'
    r'convert_.*config.*|.*_config_from_.*|set_.*default.*|_set_.*|post_init.*|'
    r'.*_check_args|validate_args|_add_.*_args|.*normalize.*)$', re.I)


def config_norm_functions(parsed):
    out = []
    for rel, (tree, src, nl) in parsed.items():
        def walk(node, fq, cls):
            for c in ast.iter_child_nodes(node):
                if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    qn = (fq + '.' + c.name) if fq else c.name
                    if CFG_FN_NAMES.match(c.name):
                        nif = sum(1 for x in ast.walk(c) if isinstance(x, ast.If))
                        nifexp = sum(1 for x in ast.walk(c) if isinstance(x, ast.IfExp))
                        nassign = sum(1 for x in ast.walk(c)
                                      if isinstance(x, (ast.Assign, ast.AugAssign, ast.AnnAssign)))
                        nraise = sum(1 for x in ast.walk(c) if isinstance(x, ast.Raise))
                        ncall = sum(1 for x in ast.walk(c) if isinstance(x, ast.Call))
                        tags, _ = ap.tag_subtree(ast.Module(body=[c], type_ignores=[]))
                        extra = sorted(t for t in tags
                                       if not t.startswith('raw.')
                                       and not t.startswith('decoratorname.')
                                       and t not in ap.CORE_TAGS)
                        out.append({'file': rel, 'qualname': qn, 'name': c.name,
                                    'lineno': c.lineno,
                                    'nlines': (c.end_lineno or c.lineno) - c.lineno + 1,
                                    'n_if': nif, 'n_ifexp': nifexp, 'n_assign': nassign,
                                    'n_raise': nraise, 'n_call': ncall,
                                    'extra_tags': extra})
                    walk(c, qn, cls)
                elif isinstance(c, ast.ClassDef):
                    walk(c, (fq + '.' + c.name) if fq else c.name, c.name)
                else:
                    walk(c, fq, cls)
        walk(tree, '', None)
    return out


# ------------------------------------------- (c) 间接层机制
def indirection(parsed):
    """注册表装饰器 / 注册表查表 / spec 间接构造 的调用点。"""
    dec = Counter()
    lookup = Counter()
    for rel, (tree, src, nl) in parsed.items():
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                for d in n.decorator_list:
                    f = ap._fmt_target(d.func if isinstance(d, ast.Call) else d)
                    if f and re.search(r'Register|register|registry', f):
                        dec[f] += 1
            if isinstance(n, ast.Call):
                f = ap._fmt_target(n.func)
                if not f:
                    continue
                if re.search(r'(get_instance|get_cls|get_instance_from_cfg|build_module|'
                             r'auto_register|registry|from_pretrained|import_module|'
                             r'build_[a-z_]+)$', f):
                    lookup[f.rsplit('.', 1)[-1] if '.' in f else f] += 1
    return dec, lookup


# ------------------------------------------- (d) 去冗余 PySub 曲线
# 这些 tag 与其"父 tag"总是共现, 计入 PySub 大小时应合并为 1 项
MERGE = {
    'dyn.getattr_conststr': 'dyn.getattr', 'dyn.getattr_dynamic': 'dyn.getattr',
    'dyn.setattr_conststr': 'dyn.setattr', 'dyn.setattr_dynamic': 'dyn.setattr',
    'dyn.hasattr_conststr': 'dyn.hasattr', 'dyn.hasattr_dynamic': 'dyn.hasattr',
    'dyn.delattr_conststr': 'dyn.delattr', 'dyn.delattr_dynamic': 'dyn.delattr',
    'ctl.except_handler': 'ctl.try', 'ctl.bare_except': 'ctl.try',
    'ctl.try_else': 'ctl.try', 'ctl.finally': 'ctl.try',
    'oop.super_zeroarg': 'oop.super_call',
    'decorator.func_parameterized': 'decorator.func',
    'comp.filter': None, 'comp.multi_for': None,   # 与推导式同轴, 单列
    'class.bases0': None, 'class.bases1': None, 'class.multi_inherit': None,
    'class.kwarg_other': None, 'arg.default': None, 'arg.kwonly': None,
    'imp.import': 'imp.stmt', 'imp.from_import': 'imp.stmt', 'imp.relative': 'imp.stmt',
    'async.await': 'def.async_func', 'ctl.async_with': 'def.async_func',
    'comp.async': 'def.async_func',
}


def merged_profiles(profs):
    out = []
    for qn, kind, tags in profs:
        t = set()
        for x in tags:
            if x in MERGE:
                m = MERGE[x]
                if m is None:
                    continue
                t.add(m)
            else:
                t.add(x)
        out.append((qn, kind, frozenset(t)))
    return out


def main():
    res = {}
    for repo, root in REPOS.items():
        files = ap.iter_py_files(root)
        parsed, failures = ap.parse_all(files)
        fw = sel(parsed, FRAMEWORK_PREFIX[repo])
        md = sel(parsed, MODEL_INCLUDE[repo], MODEL_EXCLUDE_SUBSTR[repo])
        r = {}

        tg = tight_guards(md)
        r['tight_guards_model'] = tg
        r['tight_guards_framework_count'] = len(tight_guards(fw))

        r['config_norm_fw'] = config_norm_functions(fw)
        dec, lookup = indirection(fw)
        r['indirection_decorators'] = dict(dec.most_common(20))
        r['indirection_lookups'] = dict(lookup.most_common(25))

        # 去冗余曲线
        r['curves_merged'] = {}
        for name, sub in (('framework', fw), ('model_subgraph', md)):
            profs = [p for p in ap.function_profiles(sub, include_module_scope=False)
                     if p[1] == 'func']
            mp = merged_profiles(profs)
            seed = frozenset(x for x in ap.CORE_TAGS if x not in MERGE or MERGE[x])
            seed = frozenset(MERGE.get(x, x) for x in ap.CORE_TAGS
                             if MERGE.get(x, x) is not None)
            c = ap.pysub_curve(mp, seed_tags=seed, max_steps=70)
            r['curves_merged'][name] = [{'step': s, 'tag': t, 'covered': cv, 'total': tt,
                                         'frac': round(f, 5)} for s, t, cv, tt, f in c]

        # PolicyStateBinding 精确规模: 访问器读到的可变模块级全局(去重)
        gfw = ap.analyze_globals(fw)
        acc = ap.find_accessors(fw, gfw)
        bind = set()
        for a in acc:
            for g in a['reads_mutable']:
                bind.add((a['file'], g))
        r['policystate_binding_exact'] = sorted('%s::%s' % x for x in bind)
        # 全部可变全局(不限访问器)
        allmut = sorted('%s::%s' % (f, n) for f, d in gfw.items()
                        for n, e in d.items() if e['mutable'])
        allstrong = sorted('%s::%s' % (f, n) for f, d in gfw.items()
                           for n, e in d.items() if e['mutable_strong'])
        r['all_mutable_globals'] = allmut
        r['strong_mutable_globals'] = allstrong
        res[repo] = r
        W('[%s] tight_guards_model=%d cfgfn=%d psb_exact=%d strong_mut=%d\n'
          % (repo, len(tg), len(r['config_norm_fw']), len(bind), len(allstrong)))
    with open(os.path.join(OUT, 'probe2.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
