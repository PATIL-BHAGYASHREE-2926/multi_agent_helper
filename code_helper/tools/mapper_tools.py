import ast
from ..settings import PROJECT_PATH


def _imports_of(path):
    """Helper: return the module names a file imports."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    mods = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods.append(node.module or "")
    return mods


def list_project_files() -> str:
    """List every Python file in the project. Use to see what files exist."""
    files = sorted(p.name for p in PROJECT_PATH.glob("*.py"))
    return "Files: " + ", ".join(files)


def list_functions(file_name: str) -> str:
    """List the functions, classes and imports inside ONE file, e.g. 'indexer.py'."""
    path = PROJECT_PATH / file_name
    if not path.exists():
        return f"I could not find {file_name}."
    tree = ast.parse(path.read_text(encoding="utf-8"))
    funcs = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    return (f"{file_name}\n functions: {funcs}\n classes: {classes}\n"
            f" imports: {_imports_of(path)}")


def find_importers(file_name: str) -> str:
    """Find which files import a given file, e.g. 'config.py' -> who imports config."""
    target = file_name.replace(".py", "")
    found = [p.name for p in sorted(PROJECT_PATH.glob("*.py"))
             if target in _imports_of(p)]
    if found:
        return f"Files importing {file_name}: {found}"
    return f"No file imports {file_name}."


def count_lines(file_name: str) -> str:
    """Count lines of code in ONE file, e.g. 'main.py'. Use for size questions."""
    path = PROJECT_PATH / file_name
    if not path.exists():
        return f"I could not find {file_name}."
    return f"{file_name} has {len(path.read_text(encoding='utf-8').splitlines())} lines."


def generate_mermaid_diagram() -> str:
    """Build a Mermaid diagram of which project file imports which.
    Use when the user wants a diagram or picture of the architecture."""
    names = {p.stem for p in PROJECT_PATH.glob("*.py")}
    edges = set()
    for path in sorted(PROJECT_PATH.glob("*.py")):
        for m in _imports_of(path):
            if m in names and m != path.stem:
                edges.add(f"    {path.stem} --> {m}")
    return "graph TD\n" + "\n".join(sorted(edges))


def project_overview() -> str:
    """Give a friendly newcomer guide: what each file does, where to start,
    the reading order and how to run the project. Use for 'I am new' questions."""
    files = {p.name: p for p in PROJECT_PATH.glob("*.py")}

    def first_line(path):
        doc = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))) or "no description"
        line = doc.splitlines()[0]
        return line.split("\u2014", 1)[-1].strip() if "\u2014" in line else line

    wanted = ["config.py", "vector_store.py", "indexer.py", "query_pipeline.py", "main.py"]
    order = [n for n in wanted if n in files]
    order += sorted(n for n in files if n not in order)

    lines = ["Welcome! Here is your starting guide.", "", "What each file does:"]
    for n in order:
        lines.append(f"- {n}: {first_line(files[n])}")
    lines += [
        "",
        "Run first: main.py is the entry point.",
        "Read first: config.py (all settings live there).",
        "Reading order: " + " -> ".join(order),
        "",
        "How to run it:",
        "- python main.py                  (full demo)",
        "- python main.py --index          (index only)",
        '- python main.py --query "your question"',
    ]
    return "\n".join(lines)
