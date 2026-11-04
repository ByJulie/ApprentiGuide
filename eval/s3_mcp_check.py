"""S3 check of server.py, the steps of the Session 3 notebook on our server: list what it offers, call the tools
by hand (one valid call, one invalid), then the notebook's agent loop with the tools coming from MCP.
python -m eval.s3_mcp_check   →   eval/runs/<date>_s3_mcp_check.json"""
import asyncio, json, os, sys
from datetime import date, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fastmcp import Client

from app import pipeline as P, tools

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Same JSON format as Claude Desktop, VS Code and OpenCode.
CONFIG = {"mcpServers": {"apprentiguide": {"command": sys.executable, "args": [os.path.join(ROOT, "server.py")],
                                           "env": {"FASTMCP_LOG_LEVEL": "WARNING"}}}}
TODAY = date.today()
CALENDAR = "; ".join(f"{tools.WEEKDAYS_FR[(TODAY - timedelta(days=i)).weekday()]} {TODAY - timedelta(days=i)}" for i in range(7, -1, -1))
SYSTEM = ("You are ApprentiGuide, an assistant for apprentices and international students. Use the tools; never guess "
          "rules, dates or amounts. Call search_texts before stating a rule and cite only ids it returned. Call deadline "
          f"for every delay or date limit. Today is {tools.WEEKDAYS_FR[TODAY.weekday()]} {TODAY}. Last days: {CALENDAR}.")
GOALS = ["J'ai le droit à combien de jours de congé pour préparer mes examens en apprentissage ?",
         f"Mon contrat d'apprentissage a commencé le {TODAY - timedelta(days=20)}. Jusqu'à quand mon employeur et moi "
         "pouvons-nous le rompre librement pendant la période d'essai ?"]


def text_of(result):
    return "\n".join(getattr(c, "text", str(c)) for c in result.content)


async def agent(mc, goal, max_steps=6):
    """The agent loop of the Session 3 notebook: list_tools → schemas for the model → call_tool on the server."""
    listed = await mc.list_tools()
    schemas = [{"type": "function", "function": {"name": t.name, "description": t.description or "", "parameters": t.inputSchema}}
               for t in listed]
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": goal}]
    calls = []
    for _ in range(max_steps):
        m = P.client.chat.completions.create(messages=messages, tools=schemas, **{**P._params(), "temperature": 0}).choices[0].message
        messages.append({"role": "assistant", "content": m.content or "", **({"tool_calls": [
            {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}}
            for c in m.tool_calls]} if m.tool_calls else {})})
        if not m.tool_calls:
            return {"goal": goal, "mcp_calls": calls, "answer": P.THINK.sub("", m.content or "").strip()}
        for c in m.tool_calls:
            try:
                out = text_of(await mc.call_tool(c.function.name, json.loads(c.function.arguments or "{}")))
            except Exception as e:
                out = f"Error: {str(e)[:300]}"
            calls.append({"tool": c.function.name, "arguments": json.loads(c.function.arguments or "{}"),
                          # the school regulations are private: passage ids only
                          "result": out if c.function.name != "search_texts" else
                          "passage ids: " + ", ".join(p["id"] for p in json.loads(out).get("passages", [])) if out.startswith("{") else out})
            messages.append({"role": "tool", "tool_call_id": c.id, "content": out})
    return {"goal": goal, "mcp_calls": calls, "answer": "(max steps)"}


async def main():
    out = {"date": str(TODAY), "model": P.CFG["model"]["name"], "config": CONFIG}
    async with Client(CONFIG) as mc:
        listed = await mc.list_tools()
        out["tools"] = [{"name": t.name, "input_schema": t.inputSchema, "has_output_schema": bool(t.outputSchema)} for t in listed]
        out["resources"] = [str(r.uri) for r in await mc.list_resources()]
        out["prompts"] = [p.name for p in await mc.list_prompts()]
        valid = await mc.call_tool("deadline", {"event_date": str(TODAY - timedelta(days=1)), "event_time": None,
                                                "amount": 48, "unit": "hours"})
        out["call_valid"] = valid.structured_content
        try:
            await mc.call_tool("deadline", {"event_date": "mardi", "event_time": None, "amount": 48, "unit": "jours"})
            out["call_invalid"] = "accepted (unexpected)"
        except Exception as e:
            out["call_invalid"] = f"refused by the server before our code ran: {str(e).splitlines()[0]}"
        out["agent"] = [await agent(mc, g) for g in GOALS]
    path = os.path.join(ROOT, "eval", "runs", f"{TODAY}_s3_mcp_check.json")
    json.dump(out, open(path, "w"), ensure_ascii=False, indent=1, default=str)
    print(json.dumps({k: out[k] for k in ("tools", "call_valid", "call_invalid")}, ensure_ascii=False, indent=1, default=str)[:2500])
    for a in out["agent"]:
        print("\n🎯", a["goal"], "\n🔧", [(c["tool"], c["arguments"]) for c in a["mcp_calls"]], "\n✅", a["answer"][:600])
    print("\n→", path)


if __name__ == "__main__":
    asyncio.run(main())
