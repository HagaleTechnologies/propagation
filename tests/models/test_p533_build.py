from propagation.models.p533_build import repo_root, binary_path


def test_repo_root_finds_baselines(tmp_path, monkeypatch):
    (tmp_path / "baselines" / "p533").mkdir(parents=True)
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    assert repo_root() == tmp_path


def test_binary_path_is_under_bin(tmp_path, monkeypatch):
    (tmp_path / "baselines" / "p533").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    assert binary_path() == tmp_path / "baselines" / "p533" / "bin" / "iturhfprop"
