import pathlib, re


def test_readme_example_runs():
    text = (pathlib.Path(__file__).parent.parent / "README.md").read_text()
    code = re.search(r"```python\n(.*?)```", text, re.S).group(1)
    ns = {}
    exec(code, ns)
    assert ns["obs"].ndim == 1 and "reward" in ns
