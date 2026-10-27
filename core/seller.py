"""
Learn the seller's product from their company website.

Used by Autopilot so CompanyContext.value_props (and optional product name)
are grounded in the live site, not only the form textarea.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from .models import CompanyContext
from .reliability import job_id_var

logger = logging.getLogger(__name__)

# Below this, value props are treated as too thin and get website enrichment.
SHORT_VALUE_PROPS = 80
SCRAPE_CHAR_LIMIT = 4000
MODEL = "llama-3.3-70b-versatile"


@dataclass
class SellerSummary:
    product_summary: str = ""
    value_props: str = ""
    suggested_icp: str = ""
    product_name: str = ""
    source_url: str = ""
    scrape_error: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.product_summary or self.value_props) and not (
            self.scrape_error and not self.product_summary and not self.value_props
        )


def normalize_website(url: str) -> str:
    cleaned = (url or "").strip()
    if not cleaned:
        return ""
    if not cleaned.startswith(("http://", "https://")):
        cleaned = "https://" + cleaned
    return cleaned


def scrape_seller_homepage(url: str) -> str:
    """Fetch a seller homepage and return readable text."""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; AI-SDR/1.0)"}
    resp = requests.get(url, headers=headers, timeout=12, allow_redirects=True)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    description = ""
    meta = soup.find("meta", attrs={"name": re.compile(r"^description$", re.I)})
    if meta and meta.get("content"):
        description = meta["content"].strip()
    else:
        og = soup.find("meta", attrs={"property": "og:description"})
        if og and og.get("content"):
            description = og["content"].strip()

    for tag in soup(["script", "style", "nav", "footer", "noscript", "svg"]):
        tag.decompose()

    body_text = soup.get_text(separator=" ", strip=True)
    body_text = re.sub(r"\s+", " ", body_text).strip()[:SCRAPE_CHAR_LIMIT]

    parts = []
    if title:
        parts.append(f"Title: {title}")
    if description:
        parts.append(f"Description: {description}")
    if body_text:
        parts.append(f"Page text: {body_text}")
    return "\n".join(parts)


def _parse_labelled_summary(text: str) -> SellerSummary:
    fields = {
        "PRODUCT_NAME": "",
        "PRODUCT_SUMMARY": "",
        "VALUE_PROPS": "",
        "SUGGESTED_ICP": "",
    }
    current = None
    buffers: dict[str, list[str]] = {key: [] for key in fields}

    for line in (text or "").splitlines():
        match = re.match(
            r"\*{0,2}(PRODUCT_NAME|PRODUCT_SUMMARY|VALUE_PROPS|SUGGESTED_ICP)\s*:\s*\*{0,2}\s*(.*)$",
            line.strip(),
            re.IGNORECASE,
        )
        if match:
            current = match.group(1).upper()
            rest = match.group(2).strip().strip("*").strip()
            if rest:
                buffers[current].append(rest)
            continue
        if current and line.strip():
            buffers[current].append(line.strip())

    return SellerSummary(
        product_name=" ".join(buffers["PRODUCT_NAME"]).strip(),
        product_summary=" ".join(buffers["PRODUCT_SUMMARY"]).strip(),
        value_props="\n".join(buffers["VALUE_PROPS"]).strip()
        or " ".join(buffers["PRODUCT_SUMMARY"]).strip(),
        suggested_icp=" ".join(buffers["SUGGESTED_ICP"]).strip(),
    )


def _fallback_summary(homepage_text: str, url: str) -> SellerSummary:
    """Build a usable summary from scrape text when no LLM key is available."""
    title = ""
    description = ""
    page_text = homepage_text
    for line in homepage_text.splitlines():
        if line.lower().startswith("title:"):
            title = line.split(":", 1)[1].strip()
        elif line.lower().startswith("description:"):
            description = line.split(":", 1)[1].strip()
        elif line.lower().startswith("page text:"):
            page_text = line.split(":", 1)[1].strip()

    snippet = page_text[:900].strip()
    product_summary = description or title or snippet[:240]
    value_props = "\n".join(
        part for part in (
            f"Website: {url}",
            f"Product signal: {product_summary}" if product_summary else "",
            f"Homepage excerpt: {snippet}" if snippet else "",
        ) if part
    )
    return SellerSummary(
        product_summary=product_summary,
        value_props=value_props,
        product_name=title.split("|")[0].split("-")[0].strip() if title else "",
        source_url=url,
    )


def _summarize_with_groq(homepage_text: str, company_name: str, product_hint: str) -> SellerSummary:
    from groq import Groq

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    prompt = f"""You are helping an AI SDR understand a B2B seller from their website copy.

Company name (from the user): {company_name or "Unknown"}
Product hint (from the user, may be empty): {product_hint or "Unknown"}

Website content:
{homepage_text}

Extract what they sell. Respond in EXACTLY this format:

PRODUCT_NAME: <short product or platform name>
PRODUCT_SUMMARY: <1-2 sentences on what the product does>
VALUE_PROPS: <3 short bullets or sentences: who it helps, main outcomes, why it is different>
SUGGESTED_ICP: <one paragraph: ideal customer titles, industries, company size — do not invent specifics not supported by the page>

If the page is thin, still fill every field with your best grounded guess from the text. Do not invent fake customers or metrics."""

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=450,
        temperature=0.2,
    )
    raw = (resp.choices[0].message.content or "").strip()
    return _parse_labelled_summary(raw)


def summarize_seller_site(
    url: str,
    *,
    company_name: str = "",
    product_hint: str = "",
) -> SellerSummary:
    """
    Scrape `url` and return a seller product summary.

    Uses Groq when GROQ_API_KEY is set; otherwise falls back to a structured
    excerpt of the scraped page so Autopilot still gets website context.
    """
    cleaned = normalize_website(url)
    if not cleaned:
        return SellerSummary(scrape_error="No company website provided.")

    parsed = urlparse(cleaned)
    if not parsed.netloc:
        return SellerSummary(scrape_error="Company website URL is invalid.", source_url=cleaned)

    try:
        homepage_text = scrape_seller_homepage(cleaned)
    except Exception as exc:
        logger.warning(
            "job=%s seller scrape failed url=%s error=%s",
            job_id_var.get(),
            cleaned,
            exc,
        )
        return SellerSummary(scrape_error=str(exc), source_url=cleaned)

    if not homepage_text.strip():
        return SellerSummary(scrape_error="Company website returned no readable text.", source_url=cleaned)

    summary: SellerSummary
    if os.getenv("GROQ_API_KEY"):
        try:
            summary = _summarize_with_groq(homepage_text, company_name, product_hint)
            summary.source_url = cleaned
        except Exception as exc:
            logger.warning(
                "job=%s seller summarize via Groq failed url=%s error=%s",
                job_id_var.get(),
                cleaned,
                exc,
            )
            summary = _fallback_summary(homepage_text, cleaned)
            summary.scrape_error = f"LLM summarize failed; used page excerpt. ({exc})"
    else:
        summary = _fallback_summary(homepage_text, cleaned)

    if not summary.value_props and summary.product_summary:
        summary.value_props = summary.product_summary
    summary.source_url = cleaned
    return summary


def enrich_company_context(ctx: CompanyContext, summary: SellerSummary) -> CompanyContext:
    """
    Merge a seller website summary into CompanyContext.

    User-provided ICP is never overwritten. Short or empty value props and
    product name are filled from the summary; longer user text is kept and
    the website summary is appended as enrichment.
    """
    if not summary.ok and not summary.product_summary and not summary.value_props:
        return ctx

    product = ctx.product
    if (not product or product.lower() in {"unknown", "n/a"}) and summary.product_name:
        product = summary.product_name

    existing_values = (ctx.value_props or "").strip()
    site_block = summary.value_props or summary.product_summary
    website_note = f"Learned from {summary.source_url}:\n{site_block}".strip()

    if not existing_values or len(existing_values) < SHORT_VALUE_PROPS:
        value_props = website_note if site_block else existing_values
    elif site_block and site_block not in existing_values:
        value_props = f"{existing_values}\n\n{website_note}"
    else:
        value_props = existing_values

    return CompanyContext(
        name=ctx.name,
        product=product,
        website=ctx.website or summary.source_url,
        value_props=value_props,
        icp_description=ctx.icp_description,
    )


def seller_prompt_block(ctx: CompanyContext) -> str:
    """Compact seller block for agent prompts."""
    lines = [
        f"Seller company: {ctx.name}",
        f"Product: {ctx.product}",
    ]
    if ctx.website:
        lines.append(f"Website: {ctx.website}")
    lines.append("What they sell / value props:")
    lines.append(ctx.value_props or "(not provided)")
    return "\n".join(lines)
