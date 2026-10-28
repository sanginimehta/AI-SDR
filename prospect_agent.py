"""
Prospect Research Agent — Step 3: Lead Scoring + Memory
Uses the Claude Agent SDK to score prospects on ICP fit, timing signals,
and reachability before generating outreach emails for Risk Cloud by LogicGate.

Flow:
  score_lead()         → research prospect, return 1-10 score + rationale
  research_and_email() → only runs if score >= SCORE_THRESHOLD (default 6)
  save_memory()        → persists score + every email to prospect_memory/
"""

import re
import json
import anyio
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from core.output import humanize_prose
from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    AssistantMessage,
    ResultMessage,
    SystemMessage,
    TextBlock,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MEMORY_DIR = Path(__file__).parent / "prospect_memory"
SCORE_THRESHOLD = 6          # prospects scoring below this are skipped

SYSTEM_PROMPT = """You are an expert B2B sales development representative for LogicGate,
a GRC (Governance, Risk, and Compliance) software company. Your product is Risk Cloud —
a flexible, no-code GRC platform that helps organizations manage risk, compliance, and
audit programs in a single connected system.

Risk Cloud value propositions most relevant to CISOs in financial services:
- Replaces fragmented spreadsheets and point solutions with one unified GRC platform
- Pre-built control frameworks for SOC 2, ISO 27001, NIST CSF, PCI-DSS, FFIEC, and more
- Real-time risk visibility with automated workflows and evidence collection
- Scales without custom code — business teams can own their own risk programs
- Purpose-built for regulated industries; trusted by 1,000+ companies

Email guidelines:
- Subject line + 3 short paragraphs, under 150 words total
- Paragraph 1: personalized opener referencing something specific and current
- Paragraph 2: connect their pain points to what Risk Cloud solves
- Paragraph 3: low-pressure ask, a 20-minute call or a relevant resource
- Tone: peer-to-peer, direct, no fluff
- Write like a person. Use contractions. No em dashes, en dashes, or double hyphens (--).
- No bullets, headers, or bold. Skip "I hope this finds you well", "leverage", and "excited to".
"""

# ---------------------------------------------------------------------------
# Lead score dataclass
# ---------------------------------------------------------------------------

@dataclass
class LeadScore:
    total: float        # 1–10 composite score
    icp_fit: float      # Is this the right company type, size, vertical, and title?
    timing: float       # Are there urgency signals — M&A, breach, regulation, new hire?
    reachability: float # Is the prospect a real decision-maker we can reach?
    signals: list       # Bullet points of what the agent actually found
    rationale: str      # One-paragraph explanation of the score
    skip: bool          # True if total < SCORE_THRESHOLD


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

def _memory_path(prospect: dict) -> Path:
    company = prospect["company"].lower().replace(" ", "_")
    name = prospect["name"].lower().replace(" ", "_")
    return MEMORY_DIR / f"{company}__{name}.json"


def load_memory(prospect: dict) -> dict:
    MEMORY_DIR.mkdir(exist_ok=True)
    path = _memory_path(prospect)
    if path.exists():
        return json.loads(path.read_text())
    return {"prospect": prospect, "score": None, "interactions": []}


def save_memory(prospect: dict, score: LeadScore, session_id: str = "", email: str = "") -> None:
    """Save score and (optionally) a new email interaction to disk."""
    MEMORY_DIR.mkdir(exist_ok=True)
    memory = load_memory(prospect)

    # Always persist/update the score
    memory["score"] = {
        "total": score.total,
        "icp_fit": score.icp_fit,
        "timing": score.timing,
        "reachability": score.reachability,
        "signals": score.signals,
        "rationale": score.rationale,
    }

    # Only append an email entry if one was generated
    if email:
        memory["interactions"].append({
            "date": datetime.now().isoformat(timespec="seconds"),
            "session_id": session_id,
            "email": email,
        })

    _memory_path(prospect).write_text(json.dumps(memory, indent=2))


def _format_past_interactions(interactions: list) -> str:
    lines = []
    for i, entry in enumerate(interactions, start=1):
        lines.append(f"--- Past email #{i}  ({entry['date']}) ---")
        lines.append(entry["email"].strip())
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Console helpers
# ---------------------------------------------------------------------------

def _print_step(label: str, detail: str = "") -> None:
    if detail:
        print(f"  [{label}] {detail}")
    else:
        print(f"  [{label}]")


def _handle_assistant_message(message: AssistantMessage, turn: int) -> None:
    for block in message.content:
        if isinstance(block, TextBlock) and block.text.strip():
            preview = block.text.strip()
            if len(preview) > 200:
                preview = preview[:200] + "…"
            _print_step(f"turn {turn} · agent text", preview)

        elif hasattr(block, "type") and block.type == "tool_use":
            tool_name = getattr(block, "name", "unknown_tool")
            tool_input = getattr(block, "input", {}) or {}
            if tool_name == "WebSearch":
                _print_step(f"turn {turn} · WebSearch", f'"{tool_input.get("query", "")}"')
            elif tool_name == "WebFetch":
                _print_step(f"turn {turn} · WebFetch", tool_input.get("url", ""))
            else:
                _print_step(f"turn {turn} · tool call", f"{tool_name}({tool_input})")


def _parse_score(text: str) -> dict:
    """
    Pull structured fields out of the agent's scoring response.
    The agent is asked to output labelled lines; we extract them with regex
    and fall back to 5.0 for any field we can't parse.
    """
    def _num(pattern):
        m = re.search(pattern, text, re.IGNORECASE)
        try:
            return float(m.group(1)) if m else 5.0
        except (ValueError, AttributeError):
            return 5.0

    # Signals: lines starting with a dash after "SIGNALS:"
    signals = []
    in_signals = False
    for line in text.splitlines():
        if re.match(r"SIGNALS\s*:", line, re.IGNORECASE):
            in_signals = True
            continue
        if in_signals:
            stripped = line.strip()
            if stripped.startswith("-"):
                signals.append(stripped.lstrip("- ").strip())
            elif stripped and not stripped.startswith("-"):
                in_signals = False

    # Rationale: everything after "RATIONALE:"
    rationale = ""
    m = re.search(r"RATIONALE\s*:(.*)", text, re.IGNORECASE | re.DOTALL)
    if m:
        rationale = m.group(1).strip()

    return {
        "total":         _num(r"TOTAL\s*:\s*([\d.]+)"),
        "icp_fit":       _num(r"ICP_FIT\s*:\s*([\d.]+)"),
        "timing":        _num(r"TIMING\s*:\s*([\d.]+)"),
        "reachability":  _num(r"REACHABILITY\s*:\s*([\d.]+)"),
        "signals":       signals or ["No specific signals found"],
        "rationale":     rationale or text.strip(),
    }


# ---------------------------------------------------------------------------
# Step 1: Score the lead
# ---------------------------------------------------------------------------

async def score_lead(prospect: dict) -> LeadScore:
    """
    Research the prospect and score them 1–10 on three dimensions.
    Returns a LeadScore dataclass.  If a score already exists in memory
    (from a previous run), it is returned immediately without a new agent call.

    Scoring dimensions:
      ICP Fit      — right industry (regulated), right size, right title
      Timing       — urgency signals: M&A, breach, new hire, regulatory action
      Reachability — title is a genuine decision-maker, company is findable
    """
    # Return cached score if we already researched this prospect
    memory = load_memory(prospect)
    if memory.get("score"):
        cached = memory["score"]
        _print_step("score (cached)", f"{cached['total']}/10")
        ls = LeadScore(skip=cached["total"] < SCORE_THRESHOLD, **cached)
        return ls

    website_line = f"- Website: {prospect['website']}" if prospect.get("website") else ""

    prompt = f"""You are evaluating whether this prospect is worth outreach for Risk Cloud by LogicGate
(a GRC platform for regulated industries — financial services, healthcare, manufacturing).

Prospect:
- Name: {prospect['name']}
- Title: {prospect['title']}
- Company: {prospect['company']}
- Industry: {prospect['industry']}
{website_line}

Steps:
1. Search for what {prospect['company']} does — size, industry, regulatory environment.
2. Search for recent signals: M&A activity, regulatory actions, compliance initiatives,
   leadership changes, breaches, or technology investments.
3. Score this prospect on three dimensions (1–10 each):
   - ICP_FIT: Is this the right company type, size, vertical? Is the title a decision-maker?
   - TIMING: Are there recent urgency signals suggesting a need RIGHT NOW?
   - REACHABILITY: Is this a real decision-maker we can meaningfully reach?

Respond in EXACTLY this format (no extra text before the scores):

TOTAL: <weighted average — ICP_FIT 40%, TIMING 40%, REACHABILITY 20%>
ICP_FIT: <score>
TIMING: <score>
REACHABILITY: <score>
SIGNALS:
- <signal 1>
- <signal 2>
- <signal 3>
RATIONALE: <one paragraph explaining the score>"""

    raw_text = ""
    turn = 0

    async for message in query(
        prompt=prompt,
        options=ClaudeAgentOptions(
            allowed_tools=["WebSearch", "WebFetch"],
            max_turns=8,
        ),
    ):
        if isinstance(message, SystemMessage) and message.subtype == "init":
            _print_step("scoring session", message.data.get("session_id", ""))

        elif isinstance(message, AssistantMessage):
            turn += 1
            _handle_assistant_message(message, turn)

        elif isinstance(message, ResultMessage):
            raw_text = message.result

    parsed = _parse_score(raw_text)
    ls = LeadScore(skip=parsed["total"] < SCORE_THRESHOLD, **parsed)
    return ls


# ---------------------------------------------------------------------------
# Step 2: Research and write the email
# ---------------------------------------------------------------------------

async def research_and_email(prospect: dict) -> str:
    """
    Research a prospect and generate a personalized outreach email.
    Should only be called after score_lead() confirms the lead passes threshold.
    """
    memory = load_memory(prospect)
    interactions = memory["interactions"]
    is_followup = len(interactions) > 0

    if is_followup:
        _print_step("memory", f"found {len(interactions)} past email(s) — writing follow-up")
    else:
        _print_step("memory", "first contact — writing cold email")

    website_line = f"- Website: {prospect['website']}" if prospect.get("website") else ""

    if not is_followup:
        prompt = f"""Research this prospect and write a personalized outreach email for Risk Cloud by LogicGate.

Prospect:
- Name: {prospect['name']}
- Title: {prospect['title']}
- Company: {prospect['company']}
- Industry: {prospect['industry']}
{website_line}

Steps:
1. Search for recent news — security incidents, compliance, M&A, regulation, tech investments.
2. Look up what {prospect['company']} does and who they serve.
3. Write a cold outreach email from a LogicGate sales rep to {prospect['name']}.

Output ONLY the final email (subject + body). No preamble."""

    else:
        past = _format_past_interactions(interactions)
        prompt = f"""You are following up with a prospect you have already contacted.
Do NOT repeat any hook, subject line, or angle from the emails below.

Prospect:
- Name: {prospect['name']}
- Title: {prospect['title']}
- Company: {prospect['company']}
- Industry: {prospect['industry']}
{website_line}

Emails already sent:
{past}

Steps:
1. Search for news DIFFERENT from the topics already used above.
2. If no new news, shift the angle — customer story, industry stat, different product angle.
3. Write a follow-up that feels fresh, not a nudge.

Output ONLY the final email (subject + body). No preamble."""

    result_text = ""
    session_id = "unknown"
    turn = 0

    async for message in query(
        prompt=prompt,
        options=ClaudeAgentOptions(
            system_prompt=SYSTEM_PROMPT,
            allowed_tools=["WebSearch", "WebFetch"],
            max_turns=10,
        ),
    ):
        if isinstance(message, SystemMessage) and message.subtype == "init":
            session_id = message.data.get("session_id", "unknown")
            _print_step("email session", session_id)

        elif isinstance(message, AssistantMessage):
            turn += 1
            _print_step(f"turn {turn}", "agent is working…")
            _handle_assistant_message(message, turn)
            if message.usage:
                inp = message.usage.get("input_tokens", "?")
                out = message.usage.get("output_tokens", "?")
                _print_step(f"turn {turn} · tokens", f"in={inp}  out={out}")

        elif isinstance(message, ResultMessage):
            _print_step("done", f"stop_reason={message.stop_reason}  turns={turn}")
            result_text = message.result

    return humanize_prose(result_text), session_id


# ---------------------------------------------------------------------------
# Main: score all prospects, then email the ones that pass
# ---------------------------------------------------------------------------

async def main():
    # Three prospects with meaningfully different ICP profiles so the
    # scoring filter has real work to do.
    prospects = [
        {
            # Strong ICP: CISO at regulated regional bank, active M&A
            "name": "Sarah Chen",
            "title": "Chief Information Security Officer",
            "company": "Pinnacle Financial Partners",
            "industry": "Financial Services / Regional Banking",
            "website": "https://www.pnfp.com",
        },
        {
            # Weak ICP: IT Director at a restaurant chain — no regulatory pressure,
            # wrong industry for GRC software
            "name": "Marcus Webb",
            "title": "IT Director",
            "company": "Denny's Corporation",
            "industry": "Food & Beverage / Restaurants",
            "website": "https://www.dennys.com",
        },
        {
            # Strong ICP: Chief Risk Officer at large regional bank
            "name": "Jennifer Park",
            "title": "Chief Risk Officer",
            "company": "Regions Financial Corporation",
            "industry": "Financial Services / Banking",
            "website": "https://www.regions.com",
        },
    ]

    print()
    print("=" * 64)
    print("  LEAD SCORING PASS")
    print(f"  Evaluating {len(prospects)} prospects  |  threshold: {SCORE_THRESHOLD}/10")
    print("=" * 64)

    scored = []  # list of (prospect, LeadScore)

    for p in prospects:
        print(f"\n  Scoring: {p['name']} — {p['title']} @ {p['company']}")
        print("  " + "-" * 40)
        score = await score_lead(p)
        scored.append((p, score))

        status = "PASS ✓" if not score.skip else f"SKIP — below {SCORE_THRESHOLD}"
        print(f"\n  SCORE  {score.total}/10  [{status}]")
        print(f"  ICP fit={score.icp_fit}  timing={score.timing}  reachability={score.reachability}")
        for sig in score.signals:
            print(f"    • {sig}")

    # Summary table
    print()
    print("=" * 64)
    print("  SCORING SUMMARY")
    print("=" * 64)
    print(f"  {'Prospect':<30} {'Score':>6}  {'Decision'}")
    print("  " + "-" * 50)
    for p, s in scored:
        decision = "EMAIL  →" if not s.skip else "SKIP"
        print(f"  {p['name'] + ' @ ' + p['company']:<30} {s.total:>5.1f}/10  {decision}")

    # Email generation — only for passing leads
    passing = [(p, s) for p, s in scored if not s.skip]
    print()
    print(f"  {len(passing)} of {len(prospects)} prospects pass. Generating emails…")

    for p, score in passing:
        memory = load_memory(p)
        n_past = len(memory["interactions"])
        contact_label = "FIRST CONTACT" if n_past == 0 else f"FOLLOW-UP #{n_past}"

        print()
        print("=" * 64)
        print(f"  GENERATING EMAIL  —  {contact_label}")
        print(f"  {p['name']}  |  {p['title']}  |  {p['company']}")
        print("=" * 64)
        print()

        email, session_id = await research_and_email(p)

        # Persist score + new email to memory
        save_memory(p, score, session_id=session_id, email=email)
        _print_step("memory saved", str(_memory_path(p)))

        print()
        print(email)
        print()

    print("=" * 64)
    print(f"  Done. Emailed {len(passing)} prospect(s), skipped {len(prospects) - len(passing)}.")
    print("=" * 64)


if __name__ == "__main__":
    anyio.run(main)
