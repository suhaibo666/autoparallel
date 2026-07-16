"""Z3（2026-07-16）：`default_mf_root()` 两级探测的健壮性单测（不依赖真 mindformers 路径）。

已知陷阱：抽取器要求 `MINDFORMERS_ROOT` 指向**包目录**（含 `parallel_core` 的那层，即
`…/mindformers/mindformers`），而非仓库根（其下才是 `mindformers/parallel_core`）。误指仓库根会让
抽取器找不到 `parallel_core` 而失败。`default_mf_root()` 现做两级探测：候选自身不含 `parallel_core`
但 `<候选>/mindformers/parallel_core` 存在 → 下降一层；候选已含 `parallel_core` → 原样返回。
"""
import os

from cost_eval.opdag.crosscheck import default_mf_root


def test_descends_one_level_when_candidate_is_repo_root(tmp_path, monkeypatch):
    """候选是仓库根（其下 `mindformers/parallel_core` 存在）→ 探测下降一层到包目录。"""
    root = tmp_path / "repo"
    pkg = root / "mindformers"
    (pkg / "parallel_core").mkdir(parents=True)
    monkeypatch.setenv("MINDFORMERS_ROOT", str(root))
    assert default_mf_root() == str(pkg)
    assert os.path.isdir(os.path.join(default_mf_root(), "parallel_core"))


def test_returns_unchanged_when_candidate_already_package_dir(tmp_path, monkeypatch):
    """候选已是包目录（自身含 `parallel_core`）→ 原样返回，不下降（向后兼容/逐字节不变）。"""
    pkg = tmp_path / "mindformers" / "mindformers"
    (pkg / "parallel_core").mkdir(parents=True)
    monkeypatch.setenv("MINDFORMERS_ROOT", str(pkg))
    assert default_mf_root() == str(pkg)


def test_returns_candidate_unchanged_when_neither_level_exists(tmp_path, monkeypatch):
    """两级都无 `parallel_core`（如 CI 缺源）→ 原样返回候选，缺源判定行为不变（不因探测改变）。"""
    missing = tmp_path / "nowhere"
    monkeypatch.setenv("MINDFORMERS_ROOT", str(missing))
    assert default_mf_root() == str(missing)
    assert not os.path.isdir(os.path.join(default_mf_root(), "parallel_core"))
