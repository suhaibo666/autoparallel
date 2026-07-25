# -*- coding: utf-8 -*-
"""Enumerate, from the AUTHORITATIVE snapshot, every primitive the dsv4 chain uses:
  (A) `self.X = <ns>.<fn>` bare aliases  (per class, per file)
  (B) call sites `self.X(...)` for those aliases
  (C) free calls `<ns>.<fn>(...)` appearing directly in construct/helpers
  (D) module-level free functions called by name
Pure AST. No imports of mindformers.
"""
import ast, collections, os, sys

MF = r"E:\97-codes\torch_parallel\mf-src-167\mindformers"
REL = "pynative/transformers/experimental_attention_variant"
FILES = [
    f"{REL}/deepseek_v4_hybrid_attention.py",
    f"{REL}/csa.py",
    f"{REL}/indexer.py",
    f"{REL}/compressor.py",
    f"{REL}/utils.py",
    "pynative/transformers/multi_latent_attention.py",
]
NS = ("mint", "ops", "F", "mindspore", "P", "nn", "np")


def dotted(node):
    parts = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if not isinstance(cur, ast.Name):
        return None
    parts.append(cur.id)
    return ".".join(reversed(parts))


aliases = {}           # (file, cls, attr) -> alias dotted
alias_calls = collections.Counter()   # (file, cls, attr) -> n
free_calls = collections.Counter()    # dotted -> n
free_call_src = collections.defaultdict(list)
modfunc_calls = collections.Counter() # bare name -> n

for rel in FILES:
    path = os.path.join(MF, *rel.split("/"))
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    fn = os.path.basename(rel)
    modfuncs = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    for cls in [n for n in tree.body if isinstance(n, ast.ClassDef)]:
        init = next((n for n in cls.body if isinstance(n, ast.FunctionDef)
                     and n.name == "__init__"), None)
        if init is not None:
            for st in ast.walk(init):
                if not (isinstance(st, ast.Assign) and len(st.targets) == 1):
                    continue
                t = st.targets[0]
                if not (isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
                        and t.value.id == "self"):
                    continue
                v = st.value
                if isinstance(v, ast.Attribute):
                    d = dotted(v)
                    if d and d.split(".")[0] in NS:
                        aliases[(fn, cls.name, t.attr)] = (d, st.lineno)
        # call sites of self.X(...) anywhere in the class
        for n in ast.walk(cls):
            if isinstance(n, ast.Call):
                f = n.func
                if (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name)
                        and f.value.id == "self"):
                    alias_calls[(fn, cls.name, f.attr)] += 1
    # free calls anywhere in the file (module funcs + class bodies)
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Attribute):
                d = dotted(f)
                if d and d.split(".")[0] in NS:
                    free_calls[d] += 1
                    free_call_src[d].append(f"{fn}:{n.lineno}")
            elif isinstance(f, ast.Name) and f.id in modfuncs:
                modfunc_calls[f.id] += 1

print("=" * 78)
print("(A) BARE ALIASES  self.X = <ns>.<fn>   [attr -> alias]  (called? n)")
print("=" * 78)
by_alias = collections.Counter()
for (fn, cls, attr), (d, ln) in sorted(aliases.items()):
    n = alias_calls[(fn, cls, attr)]
    by_alias[d] += n
    print(f"  {fn:38s} {cls:28s} self.{attr:22s} = {d:28s} :{ln:<4d} calls={n}")
print(f"\n  distinct classes with aliases: "
      f"{len({(f, c) for (f, c, _) in aliases})}, alias assignments: {len(aliases)}")
print(f"  distinct alias targets: {len(by_alias)}   total call sites: {sum(by_alias.values())}")
print("\n  --- distinct alias target -> total call sites ---")
for d, n in sorted(by_alias.items(), key=lambda kv: (-kv[1], kv[0])):
    print(f"    {d:34s} {n}")

print()
print("=" * 78)
print("(C) FREE CALLS  <ns>.<fn>(...)  directly in source")
print("=" * 78)
for d, n in sorted(free_calls.items(), key=lambda kv: (-kv[1], kv[0])):
    print(f"  {d:38s} {n:3d}   {' '.join(free_call_src[d][:4])}")
print(f"\n  distinct free-call paths: {len(free_calls)}  sites: {sum(free_calls.values())}")

print()
print("=" * 78)
print("(D) MODULE-LEVEL FREE FUNCTIONS called by bare name")
print("=" * 78)
for d, n in sorted(modfunc_calls.items(), key=lambda kv: (-kv[1], kv[0])):
    print(f"  {d:44s} {n}")

# union of primitive leaf names needing a table entry
leafs = set()
for d in list(by_alias) + list(free_calls):
    leafs.add(d)
print()
print(f"UNION of dotted primitives needing a table entry: {len(leafs)}")
for d in sorted(leafs):
    print("   ", d)
