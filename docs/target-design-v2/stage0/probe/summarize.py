"""从 out/*.json 提取报告用表格。"""
import json
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')
D = {r: json.load(open(os.path.join(OUT, r + '.json'), encoding='utf-8'))
     for r in ('mindformers', 'megatron')}

W = sys.stdout.write


def sec(t):
    W('\n' + '=' * 78 + '\n' + t + '\n' + '=' * 78 + '\n')


sec('0. 规模与解析')
for r, d in D.items():
    W('%s: found=%d parsed=%d failed=%d\n' % (r, d['discovery']['n_py_files_found'],
      d['discovery']['n_parsed'], d['discovery']['n_failed']))
    for k, v in d['scale'].items():
        W('   %-38s files=%-5d lines=%-8d funcs=%-6d classes=%d\n'
          % (k, v['files'], v['lines'], v['functions'], v['classes']))
    if d['failures']:
        for f in d['failures']:
            W('   FAIL %s %s %s\n' % (f['file'], f['kind'], f['msg']))

sec('1. 关注构件频次 (framework / model_subgraph)  count(files)')
FOCUS = [
    ('推导式-list', 'comp.list'), ('推导式-dict', 'comp.dict'), ('推导式-set', 'comp.set'),
    ('生成器表达式', 'comp.genexp'), ('推导式带filter', 'comp.filter'), ('多重for推导', 'comp.multi_for'),
    ('函数装饰器', 'decorator.func'), ('带参装饰器', 'decorator.func_parameterized'),
    ('类装饰器', 'decorator.class'),
    ('*args', 'arg.vararg'), ('**kwargs', 'arg.kwarg'), ('默认值', 'arg.default'),
    ('kwonly', 'arg.kwonly'), ('posonly', 'arg.posonly'),
    ('getattr', 'dyn.getattr'), ('  getattr动态名', 'dyn.getattr_dynamic'),
    ('setattr', 'dyn.setattr'), ('  setattr动态名', 'dyn.setattr_dynamic'),
    ('hasattr', 'dyn.hasattr'), ('delattr', 'dyn.delattr'),
    ('exec', 'dyn.exec'), ('eval', 'dyn.eval'), ('compile', 'dyn.compile'),
    ('__import__', 'dyn.__import__'), ('importlib import', 'dyn.importlib_import'),
    ('importlib 调用', 'dyn.importlib_call'), ('type()三参造类', 'dyn.type3_classfactory'),
    ('globals()', 'dyn.globals'), ('locals()', 'dyn.locals'), ('vars()', 'dyn.vars'),
    ('functools.partial', 'dyn.functools_partial'),
    ('元类 metaclass=', 'class.metaclass'), ('动态基类(调用)', 'class.dynamic_base'),
    ('多重继承', 'class.multi_inherit'), ('单继承', 'class.bases1'), ('无基类', 'class.bases0'),
    ('super()调用', 'oop.super_call'), ('super()零参', 'oop.super_zeroarg'),
    ('def __getattr__', 'dyn.def___getattr__'), ('def __setattr__', 'dyn.def___setattr__'),
    ('嵌套def(闭包)', 'def.func'), ('nonlocal', 'scope.nonlocal'), ('global 声明', 'scope.global'),
    ('yield', 'gen.yield'), ('yield from', 'gen.yield_from'),
    ('async def', 'def.async_func'), ('await', 'async.await'),
    ('try', 'ctl.try'), ('except handler', 'ctl.except_handler'), ('裸except', 'ctl.bare_except'),
    ('finally', 'ctl.finally'), ('raise', 'ctl.raise'),
    ('with', 'ctl.with'), ('async with', 'ctl.async_with'), ('assert', 'ctl.assert'),
    ('match', 'ctl.match'), ('del', 'ctl.del'),
    ('lambda', 'expr.Lambda'), ('条件表达式 a if c else b', 'expr.IfExp'),
    ('walrus :=', 'expr.walrus'), ('链式比较', 'expr.chained_compare'),
    ('f-string', 'expr.JoinedStr'),
    ('调用*解包', 'unpack.call_star'), ('调用**解包', 'unpack.call_dstar'),
    ('字面量**解包', 'unpack.dict_star'), ('赋值序列解包', 'unpack.assign_seq'),
    ('赋值星号解包', 'unpack.assign_star'), ('链式赋值', 'assign.chained'),
    ('import *', 'imp.star'), ('相对 import', 'imp.relative'),
]
W('%-26s | %-22s | %-22s\n' % ('构件', 'mindformers fw / model', 'megatron fw / model'))
W('-' * 78 + '\n')
for label, tag in FOCUS:
    cells = []
    for r in ('mindformers', 'megatron'):
        c = D[r]['constructs']
        fw = c['framework'].get(tag, {'count': 0, 'files': 0})
        md = c['model_subgraph'].get(tag, {'count': 0, 'files': 0})
        cells.append('%d(%df) / %d(%df)' % (fw['count'], fw['files'], md['count'], md['files']))
    W('%-26s | %-22s | %-22s\n' % (label, cells[0], cells[1]))

sec('2. 属性链长度分布 (framework)')
for r in D:
    W('%s: %s\n' % (r, D[r]['attr_chain_hist']['framework']))
    W('   model: %s\n' % D[r]['attr_chain_hist']['model_subgraph'])

sec('3. 类继承深度直方图 (仓内近似)')
for r in D:
    W('%s fw=%s\n        model=%s\n' % (r, D[r]['class_depth_hist_framework'],
                                        D[r]['class_depth_hist_model']))

sec('4. PySub 覆盖率曲线')
for key in ('framework.funcs_only.seed_core', 'model_subgraph.funcs_only.seed_core',
            'model_subgraph.funcs_and_module.seed_core', 'framework.funcs_only.seed_empty'):
    W('\n--- %s ---\n' % key)
    for r in D:
        c = D[r]['curves'][key]
        W('%s (total=%d):\n' % (r, c[0]['total']))
        for row in c[:40]:
            W('   N=%-3d +%-32s %6d  %6.2f%%\n'
              % (row['step'], row['tag'], row['covered'], row['frac'] * 100))
        # 达到里程碑所需 N
        for target in (0.5, 0.8, 0.9, 0.95, 0.99, 1.0):
            hit = next((x['step'] for x in c if x['frac'] >= target - 1e-9), None)
            W('     >= %.0f%% at N=%s\n' % (target * 100, hit))

sec('5. blocker 排行 (model_subgraph, seed=CORE)')
for r in D:
    b = D[r]['blockers']['model_subgraph']
    W('%s total_scopes=%d\n' % (r, b['total_scopes']))
    W('  involved(用到该构件的函数数) top25:\n')
    for k, v in list(b['involved'].items())[:25]:
        W('     %-34s %5d  (%.1f%%)\n' % (k, v, 100 * v / b['total_scopes']))
    W('  sole_blocker(该构件是唯一超纲项) top15:\n')
    for k, v in list(b['sole_blocker'].items())[:15]:
        W('     %-34s %5d\n' % (k, v))

sec('6. monkey-patch 点')
for r in D:
    mp = D[r]['monkeypatch']
    W('%s: framework=%d  all_repo=%d\n' % (r, len(mp), D[r]['monkeypatch_all_repo_count']))
    kinds = Counter(x['root_kind'] for x in mp)
    forms = Counter(x['form'] for x in mp)
    W('   by root_kind: %s\n' % dict(kinds))
    W('   by form: %s\n' % dict(forms))
    imported = [x for x in mp if x['root_kind'] in ('module', 'name')]
    W('   *** 对已 import 模块/名字的赋值 = %d\n' % len(imported))
    byfile = Counter(x['file'] for x in imported)
    W('   涉及文件 %d 个, top:\n' % len(byfile))
    for f, n in byfile.most_common(15):
        W('      %-70s %d\n' % (f, n))

sec('7. 模块级全局')
for r in D:
    g = D[r]['globals']
    allnames = [(f, n, e) for f, d in g.items() for n, e in d.items()]
    mut = [x for x in allnames if x[2]['mutable']]
    strong = [x for x in allnames if x[2]['mutable_strong']]
    caps = [x for x in allnames if x[2]['allcaps']]
    W('%s: 顶层绑定名字总数=%d  可变(任一证据)=%d  强可变(global 重绑定)=%d  全大写命名=%d\n'
      % (r, len(allnames), len(mut), len(strong), len(caps)))
    capsmut = [x for x in caps if x[2]['mutable']]
    W('   全大写里实际可变的=%d  (说明"全大写=常量"启发式的误判率 %.1f%%)\n'
      % (len(capsmut), 100 * len(capsmut) / max(1, len(caps))))
    byfile = Counter(x[0] for x in strong)
    W('   强可变全局按文件 top10:\n')
    for f, n in byfile.most_common(10):
        W('      %-70s %d\n' % (f, n))

sec('8. 访问器')
for r in D:
    acc = D[r]['accessors']
    W('%s: 访问器函数定义=%d\n' % (r, len(acc)))
    reading = [a for a in acc if a['reads_mutable']]
    W('   其中读到本文件可变模块级全局的 = %d\n' % len(reading))
    byfile = Counter(a['file'] for a in acc)
    for f, n in byfile.most_common(12):
        W('      %-70s %d\n' % (f, n))
    cs = D[r]['accessor_callsites']
    csm = D[r]['accessor_callsites_model_subgraph']
    W('   访问器调用点: framework=%d 次 / model_subgraph=%d 次\n'
      % (sum(cs.values()), sum(csm.values())))
    W('   model_subgraph top20 调用:\n')
    for k, v in list(csm.items())[:20]:
        W('      %-52s %d\n' % (k, v))

sec('9. policy-tainted guards (model_subgraph)')
for r in D:
    g = D[r]['guards_model_subgraph']
    tot = D[r]['if_totals_model_subgraph']
    W('%s: guards=%d  (子图内 If=%d IfExp=%d, 占比 %.1f%%)  framework guards=%d\n'
      % (r, len(g), tot['If'], tot['IfExp'],
         100 * len(g) / max(1, tot['If'] + tot['IfExp']), D[r]['guards_framework_count']))
    gc = Counter()
    for x in g:
        for grp in x['groups']:
            gc[grp] += 1
    W('   按字段组(可重复计):\n')
    for k, v in gc.most_common():
        W('      %-14s %5d\n' % (k, v))
    W('   按结构 struct: %s\n' % dict(Counter(x['struct'] for x in g)))
    W('   按改写类别 cls: %s\n' % dict(Counter(x['cls'] for x in g)))
    W('   node 类型: %s\n' % dict(Counter(x['node'] for x in g)))
    W('   if 嵌套深度分布: %s\n' % dict(sorted(Counter(x['if_depth'] for x in g).items())))
    W('   policy-guard 嵌套深度分布: %s\n' % dict(sorted(Counter(x['pol_depth'] for x in g).items())))
    fc = Counter()
    for x in g:
        for f in x['fields']:
            fc[f] += 1
    W('   最高频字段 top25:\n')
    for k, v in fc.most_common(25):
        W('      %-44s %d\n' % (k, v))
    byfile = Counter(x['file'] for x in g)
    W('   guard 最多的文件 top12:\n')
    for f, n in byfile.most_common(12):
        W('      %-70s %d\n' % (f, n))
