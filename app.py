# app.py — Expert Card (Assistants · Option A · Single Thread, 3 Assistants, In-Band State)
import os, time, json, re, requests, uuid
from typing import Dict, Any, List, Optional, Tuple
import streamlit as st
from openai import OpenAI

# ----------------------------
# Config & Env
# ----------------------------
APP_TITLE = "🟡 Expert Card — GPT-5 (3 Assistants · Single Thread · In-Band State)"
MODEL_MAIN = os.getenv("OPENAI_GPT5_SNAPSHOT", "gpt-5-2025-08-07")
MODEL_AGENT2_VALIDATOR = os.getenv("OPENAI_AGENT2_MODEL", "gpt-5-mini")

GOOGLE_CSE_KEY  = os.getenv("GOOGLE_CSE_KEY", "").strip()
GOOGLE_CSE_CX   = os.getenv("GOOGLE_CSE_CX", "").strip()
GOOGLE_CSE_SAFE = os.getenv("GOOGLE_CSE_SAFE", "off").strip().lower()  # "off" | "active"

# Hard limits
MAX_POLL_SECONDS = 90
AGENT2_MAX_CANDIDATES = 3

# ----------------------------
# UI Setup
# ----------------------------
st.set_page_config(page_title=APP_TITLE, page_icon="🟡", layout="wide")
st.title(APP_TITLE)
st.caption("Assistants API · 1 Thread als einziger Speicher · Agent-2 ruft Custom-Tools (Google CSE + Vision-Check) auf und schreibt Ergebnisse als Plaintext in den Thread.")

# ----------------------------
# OpenAI Client
# ----------------------------
def get_client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        st.stop()
    return OpenAI(api_key=key)

# ----------------------------
# Assistant blueprints (Prompts: Platzhalter — du kannst sie später verfeinern)
# ----------------------------
SYSTEM_AGENT1 = (
    "You are Agent 1 — an interviewer. Drive a short, warm conversation to collect exactly 4 strong anchors "
    "(public items: book, podcast, person, tool, film) grounded in the user's words and practices. "
    "State awareness is IN-BAND: read prior AGENT2_RESULT blocks in this same thread to avoid duplicates and to see clusters.\n"
    "Per user turn:\n"
    "1) Write a concise reply to the user (1–3 short sentences; one move: clarify, deepen, pivot, or close).\n"
    "2) Plan at most one search for Agent 2 ONLY IF a new, clearly searchable public item was mentioned. "
    "   Output a JSON object (strict) via Structured Output with:\n"
    "   - assistant_message: string (what to show to user)\n"
    "   - search_calls: array of 0..1 items {entity_type, entity_name, artifact, search_query, cluster_id}\n"
    "   - handoff: boolean (true only when the user asked to finalize)\n"
    "Artifact mapping: person→portrait, book→book_cover, podcast→podcast_cover, tool→tool_logo, film→film_poster.\n"
    "Dedupe rule: if AGENT2_RESULT exists with same key (type|normalized_name) and artifact, do not plan another search.\n"
    "Never mention agents or tools."
)

SYSTEM_AGENT2 = (
    "You are Agent 2 — Search & Validation.\n"
    "You operate ONLY when you see a new 'AGENT2_TASK' message in the thread. That JSON contains: "
    "{entity_type, entity_name, artifact, search_query, cluster_id}.\n"
    "Use tools:\n"
    " - image_search(query, max_results)\n"
    " - vision_validate(url, item_type, item_name, artifact)\n"
    "Policy:\n"
    "- Before searching, scan the thread for prior AGENT2_RESULT JSON blocks; if an identical key (type|normalized name) with same artifact already exists, reply with AGENT2_RESULT {\"status\":\"duplicate_skip\", ...} and STOP.\n"
    "- Otherwise call image_search exactly once. Take up to 3 unique candidates; run vision_validate on each until one passes (ok=true).\n"
    "- If none pass, return AGENT2_RESULT {\"status\":\"no_valid_image\", ...}.\n"
    "Return your final decision as a single plaintext line starting with:\n"
    "AGENT2_RESULT { ...json... }\n"
    "The JSON MUST include: status, key, label, artifact, cluster_id, best_image_url, source_url.\n"
    "Label examples: 'Role Model — <name>' for person; 'Must-Read — <title>' for book; 'Podcast — <title>'; 'Go-to Tool — <name>'; 'Influence — <title>'.\n"
    "Do not speak to the user. No extra prose beyond the single AGENT2_RESULT line."
)

SYSTEM_AGENT3 = (
    "You are Agent 3 — Finalizer. Read the thread (assistant questions + user answers) and the AGENT2_RESULT items. "
    "Produce exactly 4 bullet lines, grounded in THIS user's language. Each line: '- Label: ...' with 1–3 concise sentences. "
    "Prefer public anchors; if fewer than 4 exist, fill with user's practices. No tool-talk."
)

# ----------------------------
# Assistants: create/reuse (IDs im Session State cachen)
# ----------------------------
def ensure_assistants(client: OpenAI):
    if "assistants" not in st.session_state:
        st.session_state.assistants = {}

    a = st.session_state.assistants
    if not a.get("A1"):
        a["A1"] = client.assistants.create(
            name="Interview Agent",
            model=MODEL_MAIN,
            instructions=SYSTEM_AGENT1,
        ).id
    if not a.get("A2"):
        a["A2"] = client.assistants.create(
            name="Search & Validator Agent",
            model=MODEL_MAIN,
            instructions=SYSTEM_AGENT2,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "image_search",
                        "description": "Google CSE image search for public items.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"},
                                "max_results": {"type": "integer", "minimum": 1, "maximum": 10, "default": 6}
                            },
                            "required": ["query"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "vision_validate",
                        "description": "Validate that the image at url plausibly matches the requested item + artifact.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "url": {"type": "string"},
                                "item_type": {"type": "string", "enum": ["person","book","podcast","tool","film"]},
                                "item_name": {"type": "string"},
                                "artifact": {"type": "string", "enum": ["portrait","book_cover","podcast_cover","tool_logo","film_poster"]}
                            },
                            "required": ["url","item_type","item_name","artifact"]
                        }
                    }
                }
            ],
        ).id
    if not a.get("A3"):
        a["A3"] = client.assistants.create(
            name="Finalizer Agent",
            model=MODEL_MAIN,
            instructions=SYSTEM_AGENT3,
        ).id
    st.session_state.assistants = a
    return a["A1"], a["A2"], a["A3"]

# ----------------------------
# Thread helpers
# ----------------------------
def ensure_thread(client: OpenAI) -> str:
    if "thread_id" not in st.session_state or not st.session_state.thread_id:
        tid = client.threads.create().id
        st.session_state.thread_id = tid
    return st.session_state.thread_id

def add_user_message(client: OpenAI, thread_id: str, content: str):
    client.messages.create(thread_id=thread_id, role="user", content=content)

def list_messages(client: OpenAI, thread_id: str, limit: int = 100) -> List[Dict[str, Any]]:
    out = client.messages.list(thread_id=thread_id, limit=limit)
    # SDK returns .data list; preserve order oldest->newest for readability
    return list(reversed(out.data))

# ----------------------------
# Run + polling
# ----------------------------
def poll_run(client: OpenAI, thread_id: str, run_id: str) -> Dict[str, Any]:
    t0 = time.time()
    while True:
        r = client.runs.retrieve(thread_id=thread_id, run_id=run_id)
        if r.status in ("completed", "failed", "cancelled", "expired"):
            return r.to_dict()
        if r.status == "requires_action":
            return r.to_dict()
        if time.time() - t0 > MAX_POLL_SECONDS:
            return r.to_dict()
        time.sleep(0.7)

# ----------------------------
# JSON helpers
# ----------------------------
def find_json_block(text: str) -> Optional[dict]:
    if not text: return None
    m = re.search(r"\{[\s\S]*\}\s*$", text.strip())
    if not m: 
        # try first {...}
        m = re.search(r"\{[\s\S]*?\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def normalize_key(entity_type: str, entity_name: str) -> str:
    et = (entity_type or "").strip().lower()
    nm = (entity_name or "").strip().lower()
    nm = re.sub(r"[^\w\s\-:]", " ", nm)
    nm = re.sub(r"\s+", " ", nm).strip()
    return f"{et}|{nm}"

# ----------------------------
# Google CSE Tool
# ----------------------------
def google_image_search(query: str, max_results: int = 6) -> List[Dict[str, str]]:
    if not GOOGLE_CSE_KEY or not GOOGLE_CSE_CX:
        return []
    params = {
        "q": query,
        "searchType": "image",
        "num": max(1, min(max_results, 10)),
        "safe": GOOGLE_CSE_SAFE,
        "key": GOOGLE_CSE_KEY,
        "cx": GOOGLE_CSE_CX,
    }
    try:
        r = requests.get("https://www.googleapis.com/customsearch/v1", params=params, timeout=12)
        r.raise_for_status()
    except Exception:
        return []
    data = r.json()
    items = data.get("items") or []
    out = []
    seen = set()
    for it in items:
        url = (it.get("link") or "").strip()
        ctx = ""
        try:
            ctx = (it.get("image", {}).get("contextLink") or "").strip()
        except Exception:
            ctx = ""
        if url and url not in seen:
            seen.add(url)
            out.append({"url": url, "page_url": ctx})
        if len(out) >= max_results:
            break
    return out

# ----------------------------
# Vision Validator Tool (Responses API call)
# ----------------------------
def tool_vision_validate(client: OpenAI, url: str, item_type: str, item_name: str, artifact: str) -> Dict[str, Any]:
    if item_type == "person" and artifact == "portrait":
        sys = (
            'You are an image verifier. Return STRICT JSON {"ok":true/false,"reason":"..."}.\n'
            "Do NOT identify the person. Only check portrait criteria:\n"
            "- Single-human portrait or headshot; not a logo/meme/cartoon/product.\n"
            "- Not a group photo; 1 clearly visible person.\n"
            "Return ok=true if criteria hold; else false."
        )
        user_text = f"Item type: person\nItem name (context only): {item_name}\nOnly check portrait criteria."
    else:
        sys = (
            'You are an image verifier. Return STRICT JSON {"ok":true/false,"reason":"..."}.\n'
            "Check plausibility that the image matches the item + artifact given context. "
            "Be strict against mismatches; allow reasonable cover/poster variants."
        )
        user_text = f"Item type: {item_type}\nItem name: {item_name}\nArtifact: {artifact}"
    try:
        resp = client.responses.create(
            model=MODEL_AGENT2_VALIDATOR,
            input=[
                {"role":"system","content":sys},
                {"role":"user","content":[
                    {"type":"input_text","text":user_text},
                    {"type":"input_image","image_url":url}
                ]}
            ],
        )
        txt = (resp.output_text or "").strip()
        data = find_json_block(txt) or {}
        ok = bool(data.get("ok"))
        reason = (data.get("reason") or "")[:200]
        return {"ok": ok, "reason": reason}
    except Exception:
        return {"ok": False, "reason": "validator_error"}

# ----------------------------
# Agent 1 — Run with Structured Outputs
# ----------------------------
AGENT1_SCHEMA = {
    "name": "Agent1Turn",
    "schema": {
        "type": "object",
        "properties": {
            "assistant_message": {"type": "string"},
            "search_calls": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "entity_type": {"type":"string", "enum":["person","book","podcast","tool","film"]},
                        "entity_name": {"type":"string"},
                        "artifact": {"type":"string", "enum":["portrait","book_cover","podcast_cover","tool_logo","film_poster"]},
                        "search_query": {"type":"string"},
                        "cluster_id": {"type":"string"}
                    },
                    "required": ["entity_type","entity_name","artifact","search_query","cluster_id"],
                    "additionalProperties": False
                }
            },
            "handoff": {"type":"boolean", "default": False}
        },
        "required": ["assistant_message"],
        "additionalProperties": False
    },
    "strict": True
}

def run_agent1_plan(client: OpenAI, thread_id: str, a1_id: str) -> Dict[str, Any]:
    run = client.runs.create(
        thread_id=thread_id,
        assistant_id=a1_id,
        response_format={"type":"json_schema","json_schema": AGENT1_SCHEMA},
        tool_choice="none"
    )
    res = poll_run(client, thread_id, run.id)
    # Fetch latest message from A1; content will be the structured JSON (as text)
    msgs = list_messages(client, thread_id, limit=10)
    plan = {"assistant_message":"", "search_calls": [], "handoff": False}
    for m in msgs:
        if m.role == "assistant" and getattr(m, "assistant_id", None) == a1_id:
            text_parts = [p.text.value for p in m.content if getattr(p, "type", None) == "text"]
            if not text_parts: 
                continue
            data = find_json_block("\n".join(text_parts))
            if isinstance(data, dict) and "assistant_message" in data:
                plan = data
                break
    return plan

# ----------------------------
# Agent 2 — Run with tools + submit_tool_outputs
# ----------------------------
def run_agent2_task(
    client: OpenAI,
    thread_id: str,
    a2_id: str,
    task: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    # Post a task message to the thread (in-band)
    add_user_message(client, thread_id, "AGENT2_TASK " + json.dumps(task, ensure_ascii=False))

    run = client.runs.create(thread_id=thread_id, assistant_id=a2_id, tool_choice="auto")
    # Tool loop
    t0 = time.time()
    while True:
        r = client.runs.retrieve(thread_id=thread_id, run_id=run.id)
        status = r.status
        if status == "requires_action":
            calls = r.required_action.submit_tool_outputs.tool_calls
            tool_outputs = []
            for c in calls:
                name = c.function.name
                args = json.loads(c.function.arguments or "{}")
                if name == "image_search":
                    query = args.get("query","")
                    k = int(args.get("max_results", 6))
                    items = google_image_search(query, max_results=k)
                    output = json.dumps(items, ensure_ascii=False)
                    tool_outputs.append({"tool_call_id": c.id, "output": output})
                elif name == "vision_validate":
                    url = args.get("url","")
                    it = args.get("item_type","")
                    nm = args.get("item_name","")
                    art = args.get("artifact","")
                    verdict = tool_vision_validate(client, url, it, nm, art)
                    output = json.dumps(verdict, ensure_ascii=False)
                    tool_outputs.append({"tool_call_id": c.id, "output": output})
                else:
                    tool_outputs.append({"tool_call_id": c.id, "output": json.dumps({"error":"unknown_tool"}, ensure_ascii=False)})
            client.runs.submit_tool_outputs(thread_id=thread_id, run_id=run.id, tool_outputs=tool_outputs)
        elif status in ("queued", "in_progress"):
            if time.time() - t0 > MAX_POLL_SECONDS:
                break
            time.sleep(0.7)
        else:
            break

    # Parse Agent2 result (latest assistant message from A2 starting with AGENT2_RESULT)
    msgs = list_messages(client, thread_id, limit=20)
    for m in msgs:
        if m.role == "assistant" and getattr(m, "assistant_id", None) == a2_id:
            parts = [p.text.value for p in m.content if getattr(p, "type", None) == "text"]
            joined = "\n".join(parts)
            for line in joined.splitlines():
                line = line.strip()
                if line.startswith("AGENT2_RESULT"):
                    data = find_json_block(line)
                    if isinstance(data, dict) and data.get("status"):
                        return data
    return None

# ----------------------------
# Agent 3 — Finalize (returns 4 bullet lines as text)
# ----------------------------
def run_agent3_finalize(client: OpenAI, thread_id: str, a3_id: str) -> str:
    run = client.runs.create(thread_id=thread_id, assistant_id=a3_id, tool_choice="none")
    _ = poll_run(client, thread_id, run.id)
    # Read last A3 message
    msgs = list_messages(client, thread_id, limit=10)
    for m in msgs:
        if m.role == "assistant" and getattr(m, "assistant_id", None) == a3_id:
            parts = [p.text.value for p in m.content if getattr(p, "type", None) == "text"]
            return "\n".join(parts).strip()
    return ""

# ----------------------------
# UI helpers (Slots)
# ----------------------------
def get_seen_items_from_thread(client: OpenAI, thread_id: str) -> Dict[str, Dict[str, Any]]:
    """Scan all AGENT2_RESULT lines in thread to reconstruct items dict by key+artifact."""
    seen: Dict[str, Dict[str, Any]] = {}
    msgs = list_messages(client, thread_id, limit=100)
    for m in msgs:
        parts = []
        for c in m.content:
            if getattr(c, "type", None) == "text":
                parts.append(c.text.value)
        joined = "\n".join(parts)
        for line in joined.splitlines():
            if not line.startswith("AGENT2_RESULT"):
                continue
            data = find_json_block(line)
            if not isinstance(data, dict): 
                continue
            key = data.get("key","")
            artifact = data.get("artifact","")
            if data.get("status") == "ok" and key and artifact:
                seen_key = f"{key}||{artifact}"
                seen[seen_key] = data
    return seen

def render_progress(slots_count: int):
    st.progress(min(1.0, slots_count / 4), text=f"Progress: {slots_count}/4")

def render_slots(seen: Dict[str, Dict[str, Any]]):
    # Stable order by first appearance key
    keys = list(seen.keys())[:4]
    cols = st.columns(4)
    for idx, k in enumerate(keys):
        with cols[idx]:
            item = seen[k]
            st.markdown(f"**S{idx+1} · {item.get('label','')}**")
            img = item.get("best_image_url","")
            if img:
                st.image(img, caption=item.get("source_url",""), use_container_width=True)
            else:
                st.caption("—")

def extract_four_lines(text: str) -> List[str]:
    lines = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s: continue
        if s.startswith("- "):
            lines.append(s[2:].strip())
        elif s.lower().startswith("- label:"):
            lines.append(s.split(":",1)[-1].strip())
    return lines[:4]

def build_export_html(lines: List[str], seen: Dict[str, Dict[str, Any]]) -> str:
    html_items = []
    # Pair first four seen items with lines
    keys = list(seen.keys())[:4]
    for idx in range(min(4, len(lines))):
        title = seen.get(keys[idx], {}).get("label","Item")
        img = seen.get(keys[idx], {}).get("best_image_url","")
        body = lines[idx]
        html_items.append((title, body, img))

    sections = []
    for title, body, img in html_items:
        img_tag = f'<img src="{img}" alt="{title}" style="max-width:100%;height:auto;border-radius:12px;border:1px solid rgba(0,0,0,0.06);object-fit:contain;" />' if img else '<div style="color:#999;font-size:13px;">(no image)</div>'
        sections.append(f"""
        <section style="display:flex;gap:20px;align-items:flex-start;justify-content:space-between;margin:0 0 28px 0;flex-wrap:wrap;">
            <div style="flex: 1 1 56%;min-width:260px;">
                <h2 style="margin:0 0 8px 0;font-size:20px;">{title}</h2>
                <p style="margin:0;font-size:16px;">{body}</p>
            </div>
            <div style="flex: 1 1 38%;min-width:220px;display:flex;justify-content:center;">{img_tag}</div>
        </section>
        """)
    return f"""<!doctype html>
<html><head><meta charset="utf-8"/><title>Expert Card</title></head>
<body style="margin:0;padding:24px;background:#fff;color:#111;font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif;line-height:1.5;">
  <div style="max-width:900px;margin:0 auto;">
    <h1 style="margin:0 0 16px 0;font-size:28px;">Expert Card</h1>
    <div style="font-size:14px;color:#555;margin:0 0 24px 0;">Generated from interview notes.</div>
    {''.join(sections)}
  </div>
</body></html>"""

# ----------------------------
# App Main
# ----------------------------
def main():
    # Start-Gate
    st.info("Klicke **Start**, um das Interview zu beginnen. Bis dahin werden **keine** Modell-Aufrufe ausgeführt.")
    started = st.session_state.get("started", False)
    c1, c2 = st.columns([1,5])
    with c1:
        if st.button("▶️ Start"):
            st.session_state.started = True
            started = True
    if not started:
        st.stop()

    # Mode (nur UI-Hinweis für A1; Anpassung kannst du später in den Prompts verankern)
    mode = st.radio("Interview focus", ["Professional", "General / Lifescope"], horizontal=True)
    st.session_state["mode"] = mode

    client = get_client()
    a1_id, a2_id, a3_id = ensure_assistants(client)
    thread_id = ensure_thread(client)

    # On first start → seed with an empty user cue to trigger opener
    if not st.session_state.get("seeded"):
        add_user_message(client, thread_id, "[[START]]")
        st.session_state.seeded = True
        # Immediately ask A1 for the first turn
        plan = run_agent1_plan(client, thread_id, a1_id)
        st.session_state.last_plan = plan

        # If A1 already planned a search, run A2 now
        calls = plan.get("search_calls") or []
        if calls:
            task = calls[0]
            _ = run_agent2_task(client, thread_id, a2_id, task)

    # Read state from thread and render
    seen = get_seen_items_from_thread(client, thread_id)
    render_progress(len(seen))
    render_slots(seen)

    # Render visible chat: show last 12 messages from A1 and user
    msgs = list_messages(client, thread_id, limit=50)
    for m in msgs:
        parts = [p.text.value for p in m.content if getattr(p, "type", None) == "text"]
        if not parts: continue
        text = "\n".join(parts).strip()
        if m.role == "user":
            # Skip internal signals to keep UI clean
            if text.startswith("AGENT2_TASK") or text.startswith("[[START]]"):
                continue
            with st.chat_message("user"):
                st.markdown(text)
        elif m.role == "assistant" and getattr(m, "assistant_id", None) == a1_id:
            # A1 messages are structured JSON; show assistant_message
            data = find_json_block(text)
            visible = ""
            if isinstance(data, dict) and "assistant_message" in data:
                visible = data["assistant_message"]
            else:
                visible = text
            with st.chat_message("assistant"):
                st.markdown(visible)

    # Final card (if already produced)
    if st.session_state.get("final_text"):
        st.subheader("Your Expert Card")
        lines = extract_four_lines(st.session_state["final_text"])
        # Simple render: pair first 4 seen items with lines
        cols = st.columns(2)
        for idx in range(min(4, len(lines))):
            with cols[idx % 2]:
                st.markdown(f"**{list(seen.values())[idx].get('label','Item')}**")
                st.write(lines[idx])
                img = list(seen.values())[idx].get("best_image_url","")
                if img:
                    st.image(img, use_container_width=True)

        export_html = build_export_html(lines, seen)
        st.download_button("⬇️ Export HTML", data=export_html.encode("utf-8"),
                           file_name="expert-card.html", mime="text/html")

    # Chat input
    user_text = st.chat_input("Your turn…")
    if user_text:
        add_user_message(client, thread_id, user_text)

        # Agent 1 turn → plan (assistant message + optional search)
        plan = run_agent1_plan(client, thread_id, a1_id)
        st.session_state.last_plan = plan

        # Optional search (max 1 per turn)
        calls = plan.get("search_calls") or []
        if calls:
            task = calls[0]
            # Dedupe in-band: if AGENT2_RESULT already has same key+artifact, skip
            k = normalize_key(task["entity_type"], task["entity_name"])
            art = task["artifact"]
            seen_now = get_seen_items_from_thread(client, thread_id)
            if f"{k}||{art}" not in seen_now:
                _ = run_agent2_task(client, thread_id, a2_id, task)

        # Finalize?
        if plan.get("handoff"):
            final_text = run_agent3_finalize(client, thread_id, a3_id)
            st.session_state["final_text"] = final_text

        st.rerun()

    # Manual actions
    cA, cB = st.columns(2)
    with cA:
        if st.button("✨ Finalize (manual)"):
            final_text = run_agent3_finalize(client, thread_id, a3_id)
            st.session_state["final_text"] = final_text
            st.rerun()
    with cB:
        if st.button("🔄 Restart"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

if __name__ == "__main__":
    main()
