"""
AI Agent pipeline — deep prospect research via Claude Agent SDK.
CompanyContext drives both the scoring ICP and the email value props.
"""

import asyncio
import logging
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
from .output import parse_email, parse_score
from .reliability import (
    DISCOVERY_TIMEOUT_SECONDS,
    EMAIL_TIMEOUT_SECONDS,
    SCORE_TIMEOUT_SECONDS,
    job_id_var,
    with_retries,
)

SCORE_THRESHOLD = 6
logger = logging.getLogger(__name__)


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

def _log(label: str, detail: str = "") -> None:
    if detail:
        logger.info("job=%s [%s] %s", job_id_var.get(), label, detail)
    else:
        logger.info("job=%s [%s]", job_id_var.get(), label)


def _log_message(message: AssistantMessage, turn: int) -> None:
    for block in message.content:
        if isinstance(block, TextBlock) and block.text.strip():
            preview = block.text.strip()
            if len(preview) > 200:
                preview = preview[:200] + "…"
            _log(f"turn {turn} · text", preview)
        elif hasattr(block, "type") and block.type == "tool_use":
            name = getattr(block, "name", "tool")
            inp = getattr(block, "input", {}) or {}
            if name == "WebSearch":
                _log(f"turn {turn} · WebSearch", f'"{inp.get("query", "")}"')
            elif name == "WebFetch":
                _log(f"turn {turn} · WebFetch", inp.get("url", ""))
            else:
                _log(f"turn {turn} · {name}", str(inp)[:120])


async def _query_text(prompt: str, options: ClaudeAgentOptions, label: str) -> str:
    """Collect the final ResultMessage text from one agent session."""
    raw_text = ""
    turn = 0
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, SystemMessage) and message.subtype == "init":
            session = (message.data or {}).get("session_id", "")
            _log(label, session[:8])
        elif isinstance(message, AssistantMessage):
            turn += 1
            _log(f"turn {turn}", "working…")
            _log_message(message, turn)
            if message.usage:
                _log(
                    f"turn {turn} · tokens",
                    f"in={message.usage.get('input_tokens', '?')}  out={message.usage.get('output_tokens', '?')}",
                )
        elif isinstance(message, ResultMessage):
            _log(f"{label} done", f"stop={message.stop_reason}  turns={turn}")
            raw_text = message.result or ""
    if not raw_text.strip():
        raise ValueError(f"{label} returned an empty response.")
    return raw_text


async def _query_within(prompt: str, options: ClaudeAgentOptions, label: str, timeout: int) -> str:
    try:
        return await asyncio.wait_for(_query_text(prompt, options, label), timeout=timeout)
    except TimeoutError:
        minutes = max(1, round(timeout / 60))
        raise TimeoutError(f"{label} timed out after {minutes} minutes.") from None


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

    options = ClaudeAgentOptions(
        allowed_tools=["WebSearch", "WebFetch"],
        max_turns=8,
    )

    async def attempt() -> LeadScore:
        raw_text = await _query_within(prompt, options, "scoring session", SCORE_TIMEOUT_SECONDS)
        parsed = parse_score(raw_text)
        return LeadScore(skip=parsed["total"] < SCORE_THRESHOLD, **parsed)

    return await with_retries(attempt)


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
- Paragraph 3: low-pressure CTA — 20-minute call or relevant resource
- Tone: peer-to-peer, direct, no fluff
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

    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        allowed_tools=["WebSearch", "WebFetch"],
        max_turns=10,
    )

    async def attempt() -> tuple[str, str]:
        raw_text = await _query_within(prompt, options, "email session", EMAIL_TIMEOUT_SECONDS)
        return parse_email(raw_text)

    return await with_retries(attempt)


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

    options = ClaudeAgentOptions(
        allowed_tools=["WebSearch", "WebFetch"],
        max_turns=12,
    )

    logger.info("job=%s [discovery] Searching for %s prospects matching ICP", job_id_var.get(), count)

    async def attempt() -> str:
        return await _query_within(prompt, options, "discovery session", DISCOVERY_TIMEOUT_SECONDS)

    raw_text = await with_retries(attempt)
    parsed = _parse_prospects(raw_text)
    logger.info("job=%s [discovery] Found %s prospect(s)", job_id_var.get(), len(parsed))

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
