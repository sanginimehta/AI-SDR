"""
Bulk pipeline — fast email generation via Groq + optional Clearbit enrichment.
Prompt is built dynamically from CompanyContext so this works for any company.
"""

import os
import re
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from groq import Groq

from .models import CompanyContext, Prospect
from .output import EMAIL_VOICE, parse_email

MODEL = "openai/gpt-oss-20b"


def scrape_homepage(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)[:3000]


def _enrich_company(domain: str) -> dict:
    key = os.getenv("CLEARBIT_API_KEY")
    if not key:
        return {}
    try:
        resp = requests.get(
            "https://company.clearbit.com/v2/companies/find",
            params={"domain": domain},
            auth=(key, ""),
            timeout=8,
        )
        if resp.status_code != 200:
            return {}
        data = resp.json()
        return {
            "description": (data.get("description") or "").strip(),
            "industry":    (data.get("category") or {}).get("industry", ""),
            "employees":   data.get("metrics", {}).get("employees"),
            "location":    ", ".join(filter(None, [
                (data.get("geo") or {}).get("city"),
                (data.get("geo") or {}).get("country"),
            ])),
            "tech": ", ".join(
                t.get("name", "") for t in (data.get("tech") or [])[:5]
            ),
        }
    except Exception:
        return {}


def _domain(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc or parsed.path
    return host.removeprefix("www.")


def generate_email(
    prospect: Prospect,
    ctx: CompanyContext,
    groq_client: Groq,
) -> tuple[str, str]:
    """
    Scrape the prospect's homepage, enrich with Clearbit, and generate
    a personalized email using Groq.  Returns (subject, body).
    """
    # Scrape
    homepage_text = ""
    if prospect.website:
        try:
            homepage_text = scrape_homepage(prospect.website)
        except Exception:
            pass

    # Enrich
    enrichment = _enrich_company(_domain(prospect.website)) if prospect.website else {}
    enrichment_lines = []
    if enrichment.get("description"):
        enrichment_lines.append(f"Description: {enrichment['description']}")
    if enrichment.get("industry"):
        enrichment_lines.append(f"Industry: {enrichment['industry']}")
    if enrichment.get("employees"):
        enrichment_lines.append(f"Employees: {enrichment['employees']}")
    if enrichment.get("location"):
        enrichment_lines.append(f"Location: {enrichment['location']}")
    if enrichment.get("tech"):
        enrichment_lines.append(f"Tech stack: {enrichment['tech']}")
    enrichment_block = "\n".join(enrichment_lines)

    prompt = f"""You are a B2B sales rep for {ctx.name}, writing a personalized cold outreach email.

About {ctx.name}:
{ctx.value_props}

Ideal customer profile:
{ctx.icp_description}

Prospect:
- Name: {prospect.name}
- Title: {prospect.title}
- Company: {prospect.company}
- Industry: {prospect.industry}

{f"Live company data:{chr(10)}{enrichment_block}" if enrichment_block else ""}
{"Homepage content:" + chr(10) + homepage_text if homepage_text else ""}

Write a subject line and 3-paragraph email:
1. Personalized opener referencing something specific about {prospect.company}
2. Connect their likely challenges to what {ctx.product} solves
3. Low-pressure ask (20-minute call or a relevant resource)

{EMAIL_VOICE}

Format:
Subject: <subject line>

<email body>

Keep it under 150 words. Peer-to-peer tone, no buzzwords. Do not use placeholders."""

    resp = groq_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1200,
        temperature=0.7,
    )
    raw = resp.choices[0].message.content.strip()
    return parse_email(raw)
