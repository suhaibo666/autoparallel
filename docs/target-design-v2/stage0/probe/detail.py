"""细节核查: monkey-patch 逐条 / guard 抽样 / 配置规范化函数。"""
import json
import os
import random
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')
D = {r: json.load(open(os.path.join(OUT, r + '.json'), encoding='utf-8'))
     for r in ('mindformers', 'megatron')}
W = sys.stdout.write

W('#### monkey-patch 逐条 ####\n')
for r in D:
    W('\n===== %s (framework, %d 条) =====\n' % (r, len(D[r]['monkeypatch'])))
    for x in sorted(D[r]['monkeypatch'], key=lambda a: (a['file'], a['line'])):
        W('%-64s:%-5d %-42s root=%-18s kind=%-14s form=%-14s guarded=%s\n'
          % (x['file'], x['line'], x['target'][:42], str(x['root'])[:18],
             x['root_kind'], x['form'], x['guarded']))

W('\n\n#### guard 抽样 (每仓 40 条随机) ####\n')
random.seed(7)
for r in D:
    g = D[r]['guards_model_subgraph']
    W('\n===== %s =====\n' % r)
    for x in random.sample(g, 40):
        W('%-58s:%-5d [%s|%s] %s\n     test= %s\n'
          % (x['file'][-58:], x['lineno'], x['struct'], x['cls'],
             ','.join(x['groups']), x['test_src']))

W('\n\n#### guard 按 "所在函数是否为配置规范化" 拆分 ####\n')
CFGFN = ('validate', '__post_init__', 'config', 'sanity', 'check_', '_setup', 'normali',
         'from_args', 'to_dict', 'from_dict', 'set_default')
for r in D:
    g = D[r]['guards_model_subgraph']
    cfg = [x for x in g if any(k in (x['func'] or '').lower() for k in CFGFN)]
    fwd = [x for x in g if x not in cfg]
    W('%s: total=%d  配置规范化路径=%d  模型构造/前向路径=%d\n' % (r, len(g), len(cfg), len(fwd)))
    W('   前向路径 cls 分布: %s\n' % dict(Counter(x['cls'] for x in fwd)))
    W('   前向路径 struct 分布: %s\n' % dict(Counter(x['struct'] for x in fwd)))
    W('   前向路径 所在函数 top12: %s\n'
      % Counter((x['func'] or '?').split('.')[-1] for x in fwd).most_common(12))
    W('   前向路径 文件 top10:\n')
    for f, n in Counter(x['file'] for x in fwd).most_common(10):
        W('      %-72s %d\n' % (f, n))
    # 按所在函数名细分 __init__ vs forward
    W('   其中 __init__=%d  forward=%d  其他=%d\n' % (
        sum(1 for x in fwd if (x['func'] or '').endswith('__init__')),
        sum(1 for x in fwd if (x['func'] or '').endswith(('forward', 'construct'))),
        sum(1 for x in fwd if not (x['func'] or '').endswith(
            ('__init__', 'forward', 'construct')))))

W('\n\n#### call_set_changes 的具体新增/删除调用 top ####\n')
for r in D:
    g = [x for x in D[r]['guards_model_subgraph'] if x['cls'] == 'call_set_changes']
    c = Counter()
    for x in g:
        for a in x['added_calls']:
            c[a.rsplit('.', 1)[-1]] += 1
    W('%s (%d 条): %s\n' % (r, len(g), c.most_common(25)))
