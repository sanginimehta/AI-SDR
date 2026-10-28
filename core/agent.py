"""
AI Agent pipeline — deep prospect research via Claude Agent SDK.
CompanyContext drives both the scoring ICP and the email value props.
"""

import re
from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    AssistantMessage,
    ResultMessage,
    SystemMessage,
    TextBlock,
)

from .models import CompanyContext, Prospect, LeadScore
from .output import EMAIL_VOICE, parse_email

SCORE_THRESHOLD = 6


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_prospects(text: str) -> list[dict]:
    """
    Parse the discovery agent's output into a list of prospect dicts.
    Expects repeated blocks of labelled lines:
      NAME: ...
      TITLE: ...
      COMPANY: ...
      INDUSTRY: ...
      WEBSITE: ...  (optional)
    Blocks are separated by blank lines or dashes.
    """
    prospects = []
    current: dict = {}

    for line in text.splitlines():
        line = line.strip().lstrip("-").strip()
        if not line:
            if current.get("name") and current.get("company"):
                prospects.append(current)
                current = {}
            continue
        for field in ("name", "title", "company", "industry", "website"):
            m = re.match(rf"{field}\s*:\s*(.+)", line, re.IGNORECASE)
            if m:
                current[field] = m.group(1).strip()
                break

    # Catch last block if file doesn't end with blank line
    if current.get("name") and current.get("company"):
        prospects.append(current)

    return prospects

def _print_step(label: str, detail: str = "") -> None:
    if detail:
        print(f"  [{label}] {detail}")
    else:
        print(f"  [{label}]")


def _log_message(message: AssistantMessage, turn: int) -> None:
    for block in message.content:
        if isinstance(block, TextBlock) and block.text.strip():
            preview = block.text.strip()
            if len(preview) > 200:
                preview = preview[:200] + "…"
            _print_step(f"turn {turn} · text", preview)
        elif hasattr(block, "type") and block.type == "tool_use":
            name = getattr(block, "name", "tool")
            inp = getattr(block, "input", {}) or {}
            if name == "WebSearch":
                _print_step(f"turn {turn} · WebSearch", f'"{inp.get("query", "")}"')
            elif name == "WebFetch":
                _print_step(f"turn {turn} · WebFetch", inp.get("url", ""))
            else:
                _print_step(f"turn {turn} · {name}", str(inp)[:120])


def _parse_score(text: str) -> dict:
    def _num(pattern, default=5.0):
        m = re.search(pattern, text, re.IGNORECASE)
        try:
            return float(m.group(1)) if m else default
        except (ValueError, AttributeError):
            return default

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
            elif stripped:
                in_signals = False

    rationale = ""
    m = re.search(r"RATIONALE\s*:(.*)", text, re.IGNORECASE | re.DOTALL)
    if m:
        rationale = m.group(1).strip()

    return {
        "total":        _num(r"TOTAL\s*:\s*([\d.]+)"),
        "icp_fit":      _num(r"ICP_FIT\s*:\s*([\d.]+)"),
        "timing":       _num(r"TIMING\s*:\s*([\d.]+)"),
        "reachability": _num(r"REACHABILITY\s*:\s*([\d.]+)"),
        "signals":      signals or ["No specific signals found"],
        "rationale":    rationale or text.strip(),
    }


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

async def score_lead(prospect: Prospect, ctx: CompanyContext) -> LeadScore:
    """
    Research the prospect and score them 1-10 on ICP fit, timing, and reachability.
    Scoring prompt is built from ctx.icp_description so it works for any company.
    """
    website_line = f"- Website: {prospect.website}" if prospect.website else ""

    prompt = f"""Evaluate whether this prospect is worth outreach for {ctx.name} ({ctx.product}).

{ctx.name}'s ideal customer:
{ctx.icp_description}

Prospect:
- Name: {prospect.name}
- Title: {prospect.title}
- Company: {prospect.company}
- Industry: {prospect.industry}
{website_line}

Steps:
1. Search for what {prospect.company} does — size, industry, regulatory or operational context.
2. Search for recent signals: M&A, regulatory actions, compliance initiatives, leadership changes,
   breaches, technology investments, or product launches.
3. Score on three dimensions (1-10 each):
   - ICP_FIT: Does this company and title match the ideal customer profile above?
   - TIMING: Are there recent urgency signals suggesting a need RIGHT NOW?
   - REACHABILITY: Is this title a real decision-maker we can meaningfully reach?

Respond in EXACTLY this format:

TOTAL: <ICP_FIT×0.4 + TIMING×0.4 + REACHABILITY×0.2>
ICP_FIT: <score>
TIMING: <score>
REACHABILITY: <score>
SIGNALS:
- <signal 1>
- <signal 2>
- <signal 3>
RATIONALE: <one paragraph>"""

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
            _print_step("scoring session", message.data.get("session_id", "")[:8])
        elif isinstance(message, AssistantMessage):
            turn += 1
            _log_message(message, turn)
        elif isinstance(message, ResultMessage):
            raw_text = message.result

    parsed = _parse_score(raw_text)
    return LeadScore(skip=parsed["total"] < SCORE_THRESHOLD, **parsed)


async def research_and_email(
    prospect: Prospect,
    ctx: CompanyContext,
    past_emails: list[str] | None = None,
) -> tuple[str, str]:
    """
    Research the prospect and write a personalized email.
    past_emails: list of previously sent email strings (for follow-ups).
    Returns (subject, body).
    """
    website_line = f"- Website: {prospect.website}" if prospect.website else ""
    is_followup = bool(past_emails)

    system_prompt = f"""You are an expert B2B sales development representative for {ctx.name}.
Your product is {ctx.product}.

{ctx.value_props}

Email guidelines:
- Subject line + 3 short paragraphs, under 150 words total
- Paragraph 1: personalized opener referencing something specific and current
- Paragraph 2: connect their pain points to what {ctx.product} solves
- Paragraph 3: low-pressure ask, a 20-minute call or a relevant resource
- Tone: peer-to-peer, direct, no fluff

{EMAIL_VOICE}
"""

    if not is_followup:
        prompt = f"""Research this prospect and write a personalized cold outreach email for {ctx.product}.

Prospect:
- Name: {prospect.name}
- Title: {prospect.title}
- Company: {prospect.company}
- Industry: {prospect.industry}
{website_line}

Steps:
1. Search for recent news about {prospect.company} — M&A, compliance, regulation, leadership, breaches.
2. Look up what {prospect.company} does and who they serve.
3. Write the email.

Format:
Subject: <subject line>

<email body>

Output ONLY the email. No preamble."""

    else:
        past_block = "\n\n".join(
            f"--- Email #{i+1} ---\n{e}" for i, e in enumerate(past_emails)
        )
        prompt = f"""Follow up with this prospect. Do NOT repeat any hook or subject line from past emails below.

Prospect:
- Name: {prospect.name}
- Title: {prospect.title}
- Company: {prospect.company}
- Industry: {prospect.industry}
{website_line}

Past emails sent:
{past_block}

Steps:
1. Search for news DIFFERENT from topics already covered above.
2. If no new news, shift angle — customer story, industry stat, different product capability.
3. Write a fresh follow-up.

Format:
Subject: <subject line>

<email body>

Output ONLY the email. No preamble."""

    raw_text = ""
    turn = 0

    async for message in query(
        prompt=prompt,
        options=ClaudeAgentOptions(
            system_prompt=system_prompt,
            allowed_tools=["WebSearch", "WebFetch"],
            max_turns=10,
        ),
    ):
        if isinstance(message, SystemMessage) and message.subtype == "init":
            _print_step("email session", message.data.get("session_id", "")[:8])
        elif isinstance(message, AssistantMessage):
            turn += 1
            _print_step(f"turn {turn}", "working…")
            _log_message(message, turn)
            if message.usage:
                _print_step(
                    f"turn {turn} · tokens",
                    f"in={message.usage.get('input_tokens','?')}  out={message.usage.get('output_tokens','?')}",
                )
        elif isinstance(message, ResultMessage):
            _print_step("done", f"stop={message.stop_reason}  turns={turn}")
            raw_text = message.result

    return parse_email(raw_text)


async def discover_prospects(ctx: CompanyContext, count: int = 5) -> list[Prospect]:
    """
    Autonomously find `count` real prospects that match ctx.icp_description.

    The agent searches the web for companies that fit the ICP, identifies the
    right decision-maker at each one, and returns them as Prospect objects.
    The caller then passes these straight into score_lead() + research_and_email().

    Args:
        ctx:   CompanyContext describing the seller and their ICP
        count: how many prospects to return (default 5)

    Returns:
        List of Prospect objects (may be fewer than count if not enough found)
    """
    prompt = f"""You are a B2B sales researcher for {ctx.name}, which sells {ctx.product}.

Your job: find {count} real, named decision-makers at companies that match this ICP:

{ctx.icp_description}

Steps:
1. Search for companies that match this ICP — focus on finding real company names,
   not generic descriptions. Be specific: search for actual named companies.
2. For each company you find, search for the right decision-maker title
   (e.g. CISO, Chief Risk Officer, VP Compliance, Head of Security) and find
   a real named person in that role if possible. Use LinkedIn search queries,
   press releases, company leadership pages, or news articles.
3. Return exactly {count} prospects in the format below. Only include real people
   and companies you actually found evidence for — do not fabricate names.

Output format — one block per prospect, separated by a blank line:

NAME: <full name>
TITLE: <job title>
COMPANY: <company name>
INDUSTRY: <industry>
WEBSITE: <company website URL>

Output ONLY the {count} prospect blocks. No introduction, no explanation."""

    raw_text = ""
    turn = 0

    print(f"\n[discovery] Searching for {count} prospects matching ICP…")

    async for message in query(
        prompt=prompt,
        options=ClaudeAgentOptions(
            allowed_tools=["WebSearch", "WebFetch"],
            max_turns=12,
        ),
    ):
        if isinstance(message, SystemMessage) and message.subtype == "init":
            _print_step("discovery session", message.data.get("session_id", "")[:8])
        elif isinstance(message, AssistantMessage):
            turn += 1
            _log_message(message, turn)
        elif isinstance(message, ResultMessage):
            _print_step("discovery done", f"turns={turn}")
            raw_text = message.result

    parsed = _parse_prospects(raw_text)
    print(f"  Found {len(parsed)} prospect(s)")

    return [
        Prospect(
            name=p.get("name", "Unknown"),
            title=p.get("title", ""),
            company=p.get("company", ""),
            industry=p.get("industry", ""),
            website=p.get("website", ""),
        )
        for p in parsed
        if p.get("name") and p.get("company")
    ]
