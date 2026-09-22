import ast
import importlib
import re
from pathlib import Path

ROOT = Path(__file__).parents[1]
DOCS = (ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md")))
PYTHON_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)
MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)]+\.md(?:#[^)]+)?)\)")


def _python_blocks(path):
    return PYTHON_BLOCK.findall(path.read_text())


def test_documented_python_blocks_parse():
    for path in DOCS:
        for index, block in enumerate(_python_blocks(path), start=1):
            ast.parse(block, filename=f"{path}:python-block-{index}")


def test_documented_imports_resolve():
    for path in DOCS:
        for block in _python_blocks(path):
            tree = ast.parse(block)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    module = importlib.import_module(node.module)
                    for alias in node.names:
                        if alias.name != "*":
                            assert hasattr(module, alias.name), (
                                f"{path}: {node.module}.{alias.name} does not exist"
                            )
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        importlib.import_module(alias.name)


def test_relative_markdown_links_exist():
    for path in DOCS:
        text = path.read_text()
        for target in MARKDOWN_LINK.findall(text):
            target = target.split("#", 1)[0]
            if "://" in target:
                continue
            assert (path.parent / target).resolve().exists(), (
                f"{path}: missing {target}"
            )


def test_marked_documentation_examples_execute():
    namespace = {"__name__": "__probstats_docs__"}
    for path in DOCS:
        for index, block in enumerate(_python_blocks(path), start=1):
            if "# probstats: execute" not in block:
                continue
            code = compile(block, f"{path}:python-block-{index}", "exec")
            exec(code, dict(namespace))


def test_generated_api_reference_is_current(tmp_path):
    import subprocess
    import sys

    api_path = ROOT / "docs" / "api-reference.md"
    capability_path = ROOT / "docs" / "distribution-capabilities.md"
    before = (api_path.read_text(), capability_path.read_text())
    subprocess.run(
        [sys.executable, str(ROOT / "tools" / "generate_api_reference.py")],
        cwd=ROOT,
        check=True,
    )
    after = (api_path.read_text(), capability_path.read_text())
    assert after == before
