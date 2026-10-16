"""Multi-Agent Code Helper: 1 coordinator + 4 workers, built with Google ADK."""
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from google.genai import types
from mcp import StdioServerParameters

from .settings import PROJECT_PATH
from .tools.rag import rag_search
from .tools.mapper_tools import (
    list_project_files, list_functions, find_importers,
    count_lines, generate_mermaid_diagram, project_overview,
)

# ONE model object, shared by every agent
model = LiteLlm(model="nvidia_nim/meta/llama-3.2-11b-vision-instruct")


# ---------- helpers: read tool results and build replies ----------
def _text_of(response):
    """Pull plain text out of a tool response (handles the MCP wrapper)."""
    if isinstance(response, dict):
        content = response.get("content")
        if isinstance(content, list):
            texts = [c.get("text", "") for c in content if isinstance(c, dict)]
            if any(texts):
                return "\n".join(texts)
        if "result" in response:
            return str(response["result"])
    return str(response)


def _tool_results(llm_request):
    """Tool results from the current question, newest first."""
    results = []
    for content in reversed(llm_request.contents):
        parts = content.parts or []
        if content.role == "user" and any(p.text for p in parts):
            break
        for p in parts:
            fr = p.function_response
            if fr and fr.name != "transfer_to_agent":
                results.append((fr.name, _text_of(fr.response)))
    return results


def _answer(text):
    return LlmResponse(content=types.Content(role="model", parts=[types.Part(text=text)]))


# ---------- guards: stop small-model loops and tidy the output ----------
def direct_answer_guard(callback_context, llm_request):
    """After ONE tool result, return it as the answer (Mapper and Guide)."""
    results = _tool_results(llm_request)
    if not results:
        return None
    text = results[0][1]
    if text.startswith("graph"):
        text = "Import diagram (paste it into mermaid.live to see it):\n\n```mermaid\n" + text + "\n```"
    return _answer(text)


def search_guard(callback_context, llm_request):
    """After ONE rag_search, show the best files and top 3 code snippets."""
    results = _tool_results(llm_request)
    if not results:
        return None
    raw = results[0][1]
    pieces = [p.strip() for p in raw.split("\n\n---\n\n") if p.strip()]
    if not pieces or raw.startswith("I could not find"):
        return _answer("I could not find that.")
    files = list(dict.fromkeys(p.split("|")[0].lstrip("[").strip() for p in pieces))
    lines = ["Most relevant file(s): " + ", ".join(f"`{f}`" for f in files), "", "Top matching code:"]
    for p in pieces[:3]:
        header, _, body = p.partition("\n")
        name = header.strip("[]").split("|")
        label = name[0].strip() + " (" + (name[1].strip() if len(name) > 1 else "") + ")"
        lines += ["", "**" + label + "**", "```python", body.strip()[:600], "```"]
    return _answer("\n".join(lines))


def lock_after_tool(callback_context, llm_request):
    """After the first tool result, remove the tools so the model must write the answer."""
    results = _tool_results(llm_request)
    if not results:
        return None
    if len(results) >= 2:
        return _answer(results[0][1])
    llm_request.config.tools = None
    if hasattr(llm_request, "tools_dict"):
        llm_request.tools_dict.clear()
    return None


# ---------- the 4 workers ----------
searcher = LlmAgent(
    name="code_searcher",
    model=model,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    description="Finds WHERE something is in the code. Use for 'where is X done?' questions.",
    instruction="Call rag_search exactly once with the user's question.",
    before_model_callback=search_guard,
    tools=[rag_search],
)

file_tools = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", str(PROJECT_PATH)],
        ),
        timeout=60,
    ),
    tool_filter=["read_text_file"],
)

explainer = LlmAgent(
    name="code_explainer",
    model=model,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    description="Reads ONE whole file and explains what it does. Use for 'explain <file>' questions.",
    instruction=(
        f"The project folder is {PROJECT_PATH}.\n"
        "Call read_text_file ONE time with the full path of the file the user named.\n"
        "Then explain the file in 5 plain sentences: its purpose, its main functions, "
        "and what it imports. Describe only what is really in the file."
    ),
    before_model_callback=lock_after_tool,
    tools=[file_tools],
)

mapper = LlmAgent(
    name="architecture_mapper",
    model=model,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    description=(
        "Lists files, functions, imports, line counts and how files connect. "
        "Use for 'which files import X?', 'what functions are in X?', "
        "'how many lines?' and 'draw a diagram'."
    ),
    instruction=(
        "Call exactly ONE tool that matches the question:\n"
        "- which files exist -> list_project_files\n"
        "- functions or imports inside a file -> list_functions\n"
        "- which files import a file -> find_importers\n"
        "- how many lines -> count_lines\n"
        "- diagram or how files connect -> generate_mermaid_diagram"
    ),
    before_model_callback=direct_answer_guard,
    tools=[list_project_files, list_functions, find_importers,
           count_lines, generate_mermaid_diagram],
)

guide = LlmAgent(
    name="onboarding_guide",
    model=model,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    description=(
        "Helps newcomers: what the project does, how to run it, "
        "where to start reading. Use for 'I am new' questions."
    ),
    instruction="Call project_overview exactly once.",
    before_model_callback=direct_answer_guard,
    tools=[project_overview],
)

# ---------- the boss: routes only, never answers ----------
root_agent = LlmAgent(
    name="coordinator",
    model=model,
    description="Sends each question to the right worker.",
    instruction=(
        "You are a router. You never answer questions yourself.\n"
        "Transfer every question to exactly one worker, using these rules in order:\n"
        "1. Question starts with 'Explain' or asks what a named file does -> code_explainer\n"
        "2. Question contains 'import', 'imports', 'how many lines', 'functions in', "
        "'diagram' or 'list files' -> architecture_mapper\n"
        "3. Question contains 'I am new', 'where do I start' or 'how to run' -> onboarding_guide\n"
        "4. Any other question, including 'where is', 'which function', 'where does' "
        "and 'how does the code' -> code_searcher\n"
        "Transfer immediately without asking the user anything."
    ),
    sub_agents=[searcher, explainer, mapper, guide],
)
