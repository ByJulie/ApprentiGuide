"""ApprentiGuide MCP server (Agentic AI, S3): our read tools of app/tools.py, exposed with FastMCP.

The parameters reuse the pydantic fields of app/tools.py, so the schema a client lists is the one of the agent
sheet, and the result is the pydantic result model (structured output). send_email, the write tool, is not
exposed: its approval belongs to the interface (Session 7).

    python server.py                    stdio (Claude Desktop, VS Code, OpenCode, the S3 notebook)
    MCP_PORT=8765 python server.py      Streamable HTTP on 127.0.0.1:8765/mcp (n8n)

deadline needs only pydantic and fastmcp. search_texts needs the PostgreSQL database and the local models of
the RAG; without them it returns a clear error."""
import os
from datetime import date
from typing import Annotated, Literal

from fastmcp import FastMCP

from app import tools

mcp = FastMCP("apprentiguide")
D = tools.DeadlineArgs.model_fields
S = tools.SearchTextsArgs.model_fields


@mcp.tool(name="deadline", description=tools.TOOLS["deadline"]["description"])
def deadline(event_date: Annotated[date, D["event_date"]],
             event_time: Annotated[str | None, D["event_time"]],
             amount: Annotated[int, D["amount"]],
             unit: Annotated[Literal["hours", "calendar_days", "working_days_mon_sat", "working_days_mon_fri", "months"],
                             D["unit"]]) -> tools.DeadlineResult:
    args = tools.DeadlineArgs(event_date=event_date, event_time=event_time, amount=amount, unit=unit)
    return tools.deadline(args, today=date.today())


@mcp.tool(name="search_texts", description=tools.TOOLS["search_texts"]["description"])
def search_texts(query: Annotated[str, S["query"]]) -> tools.SearchTextsResult:
    try:
        return tools.search_texts(tools.SearchTextsArgs(query=query))
    except Exception as e:                     # no database or no models on this machine
        raise RuntimeError(f"search_texts needs the ApprentiGuide database and models ({type(e).__name__}). "
                           "Start PostgreSQL (README, section 3) or use deadline only.") from e


if __name__ == "__main__":
    port = os.environ.get("MCP_PORT")
    if port:
        mcp.run(transport="http", host="127.0.0.1", port=int(port), show_banner=False)
    else:
        mcp.run(show_banner=False)
