import asyncio

from fastmcp import Client

from server import mcp


def run(coro):
    return asyncio.run(coro)


async def listed():
    async with Client(mcp) as c:          # in memory, the same protocol as stdio
        return {t.name: t for t in await c.list_tools()}


def test_server_lists_the_two_read_tools_with_the_sheet_schemas():
    from app import tools
    t = run(listed())
    assert set(t) == {"deadline", "search_texts"}          # send_email, the write tool, is not exposed
    assert t["deadline"].inputSchema["properties"]["unit"]["enum"] == \
        tools.schema("deadline")["function"]["parameters"]["properties"]["unit"]["enum"]
    assert t["deadline"].inputSchema["additionalProperties"] is False and t["deadline"].outputSchema


def test_deadline_over_mcp_returns_the_result_model():
    async def call():
        async with Client(mcp) as c:
            return await c.call_tool("deadline", {"event_date": "2026-11-03", "event_time": "10:00", "amount": 48, "unit": "hours"})
    r = run(call()).structured_content
    assert (r["deadline_date"], r["deadline_weekday"], r["deadline_time"]) == ("2026-11-05", "jeudi", "10:00")


def test_invalid_call_is_refused_by_the_server():
    async def call():
        async with Client(mcp) as c:
            return await c.call_tool("deadline", {"event_date": "mardi", "event_time": None, "amount": 48, "unit": "jours"},
                                     raise_on_error=False)
    r = run(call())
    assert r.is_error and "event_date" in r.content[0].text and "unit" in r.content[0].text
