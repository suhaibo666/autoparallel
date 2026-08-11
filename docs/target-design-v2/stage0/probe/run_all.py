"""驱动脚本: 对 mindformers / Megatron-LM 跑全部五组探针，输出 JSON 到 out/。"""
import json
import os
import sys
import time
from collections import Counter, defaultdict

sys.setrecursionlimit(20000)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import astprobe as ap  # noqa: E402

OUT = os.path.join(HERE, 'out')
os.makedirs(OUT, exist_ok=True)

REPOS = {
    'mindformers': r'E:\97-codes\torch_parallel\mindformers',
    'megatron': r'E:\97-codes\torch_parallel\Megatron-LM',
}

# ---- 口径定义 -------------------------------------------------------------
# FRAMEWORK: 框架自身实现代码(PE 有机会读到的), 排除测试/文档/工具脚本
FRAMEWORK_PREFIX = {
    'mindformers': ('mindformers/', 'research/'),
    'megatron': ('megatron/',),
}
# MODEL: 模型定义子图 —— 前向图定义与其直接支撑层
MODEL_INCLUDE = {
    'mindformers': (
        'mindformers/models/',
        'mindformers/modules/',
        'mindformers/parallel_core/',
        'mindformers/pynative/base_models/',
        'mindformers/pynative/layers/',
        'mindformers/pynative/transformers/',
        'mindformers/pynative/models/',
        'mindformers/pynative/loss/',
    ),
    'megatron': (
        'megatron/core/models/',
        'megatron/core/transformer/',
        'megatron/core/tensor_parallel/',
        'megatron/core/fusions/',
        'megatron/core/ssm/',
        'megatron/core/extensions/',
    ),
}
# 从模型子图里剔除: 分词器/处理器/权重转换/自动类等非前向图代码
MODEL_EXCLUDE_SUBSTR = {
    'mindformers': (
        'tokeniz', 'processing_utils', 'image_processing', 'convert_slow',
        'sentencepiece_model_pb2', '/auto/', 'build_tokenizer', 'build_processor',
        'base_processor',
    ),
    'megatron': (
        'tokeniz', '/huggingface/',
    ),
}

ENTRY = {
    'mindformers': ['mindformers/models/llama/llama.py',
                    'mindformers/models/deepseek3/modeling_deepseek3.py',
                    'mindformers/parallel_core/training_graph/base_models/gpt/gpt_model.py'],
    'megatron': ['megatron/core/models/gpt/gpt_model.py',
                 'megatron/core/transformer/transformer_block.py',
                 'megatron/core/models/gpt/gpt_layer_specs.py'],
}
PKG_PREFIX = {'mindformers': ('mindformers', 'research'), 'megatron': ('megatron',)}


def sel(parsed, prefixes, excl_substr=()):
    out = {}
    for rel, v in parsed.items():
        if not any(rel.startswith(p) for p in prefixes):
            continue
        if any(s in ('/' + rel) for s in excl_substr):
            continue
        out[rel] = v
    return out


def curve_json(c):
    return [{'step': s, 'tag': t, 'covered': cv, 'total': tt, 'frac': round(f, 5)}
            for s, t, cv, tt, f in c]


def run(repo, root):
    t0 = time.time()
    files = ap.iter_py_files(root)
    parsed, failures = ap.parse_all(files)
    print('[%s] files=%d parsed=%d failed=%d (%.1fs)'
          % (repo, len(files), len(parsed), len(failures), time.time() - t0))

    fw = sel(parsed, FRAMEWORK_PREFIX[repo])
    md = sel(parsed, MODEL_INCLUDE[repo], MODEL_EXCLUDE_SUBSTR[repo])
    print('   framework=%d  model_subgraph=%d' % (len(fw), len(md)))

    res = {'repo': repo, 'root': root}
    res['discovery'] = {'n_py_files_found': len(files), 'n_parsed': len(parsed),
                        'n_failed': len(failures)}
    res['failures'] = [{'file': r, 'kind': k, 'msg': m} for r, k, m in failures]

    # 规模
    res['scale'] = {
        'all_repo': ap.repo_scale(parsed),
        'framework': ap.repo_scale(fw),
        'model_subgraph': ap.repo_scale(md),
    }
    res['model_subgraph_files'] = sorted(md.keys())

    # import 闭包口径
    clos = ap.import_closure(parsed, ENTRY[repo], PKG_PREFIX[repo])
    clos_p = {r: parsed[r] for r in clos}
    res['scale']['import_closure_from_model_entries'] = ap.repo_scale(clos_p)
    res['import_closure_entries'] = ENTRY[repo]

    # ① 构件频次
    for name, sub in (('framework', fw), ('model_subgraph', md), ('all_repo', parsed)):
        st, chain = ap.repo_construct_stats(sub)
        res.setdefault('constructs', {})[name] = st
        res.setdefault('attr_chain_hist', {})[name] = {str(k): v for k, v in sorted(chain.items())}

    # PySub 曲线
    res['curves'] = {}
    res['blockers'] = {}
    for name, sub in (('framework', fw), ('model_subgraph', md)):
        for scope_mode, incl_mod in (('funcs_only', False), ('funcs_and_module', True)):
            profs = ap.function_profiles(sub, include_module_scope=incl_mod)
            if not incl_mod:
                profs = [p for p in profs if p[1] == 'func']
            c = ap.pysub_curve(profs, seed_tags=ap.CORE_TAGS, max_steps=70)
            res['curves']['%s.%s.seed_core' % (name, scope_mode)] = curve_json(c)
            if scope_mode == 'funcs_only':
                c0 = ap.pysub_curve(profs, seed_tags=frozenset(), max_steps=70)
                res['curves']['%s.%s.seed_empty' % (name, scope_mode)] = curve_json(c0)
                inv, sole, tot = ap.blocker_ranking(profs, ap.CORE_TAGS)
                res['blockers'][name] = {
                    'total_scopes': tot,
                    'involved': dict(inv.most_common()),
                    'sole_blocker': dict(sole.most_common()),
                }
        del profs

    # ③ monkey-patch
    mp_fw = ap.find_monkey_patches(fw)
    res['monkeypatch'] = [{'file': r, 'line': ln, 'target': tg, 'root': rt,
                           'root_kind': rk, 'origin': og, 'form': fm, 'guarded': gd}
                          for r, ln, tg, rt, rk, og, fm, gd in mp_fw]
    mp_all = ap.find_monkey_patches(parsed)
    res['monkeypatch_all_repo_count'] = len(mp_all)

    # ④ globals + accessors
    gfw = ap.analyze_globals(fw)
    res['globals'] = {r: d for r, d in gfw.items() if d}
    acc = ap.find_accessors(fw, gfw)
    res['accessors'] = acc
    accnames = {a['name'] for a in acc}
    cnt, perfile = ap.find_accessor_callsites(fw, accnames)
    res['accessor_callsites'] = dict(cnt.most_common())
    cnt_md, _ = ap.find_accessor_callsites(md, accnames)
    res['accessor_callsites_model_subgraph'] = dict(cnt_md.most_common())

    # ⑤ guards
    res['guards_model_subgraph'] = ap.find_policy_guards(md)
    res['guards_framework_count'] = len(ap.find_policy_guards(fw))
    nif, nifexp = ap.count_all_ifs(md)
    res['if_totals_model_subgraph'] = {'If': nif, 'IfExp': nifexp}
    nif2, nifexp2 = ap.count_all_ifs(fw)
    res['if_totals_framework'] = {'If': nif2, 'IfExp': nifexp2}

    # 类继承
    hist, ncls = ap.class_hierarchy(fw)
    res['class_depth_hist_framework'] = {str(k): v for k, v in sorted(hist.items())}
    hist2, _ = ap.class_hierarchy(md)
    res['class_depth_hist_model'] = {str(k): v for k, v in sorted(hist2.items())}

    with open(os.path.join(OUT, repo + '.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print('[%s] done in %.1fs -> %s.json' % (repo, time.time() - t0, repo))
    return res


if __name__ == '__main__':
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for repo, root in REPOS.items():
        if only and repo != only:
            continue
        run(repo, root)
