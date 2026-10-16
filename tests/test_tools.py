import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from code_helper.tools.mapper_tools import (
    list_project_files, list_functions, find_importers,
    count_lines, generate_mermaid_diagram, project_overview,
)

assert "config.py" in list_project_files()
assert "indexer.py" in find_importers("config.py")
assert "lines" in count_lines("main.py")
assert generate_mermaid_diagram().startswith("graph TD")
assert "main.py" in project_overview()
assert "could not find" in list_functions("missing.py")
print("All no-AI tool tests passed.")
