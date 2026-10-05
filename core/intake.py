"""Validate a run request before any model call."""

import csv
import io
import os

from .models import CompanyContext, Prospect
from .reliability import MAX_AGENT, MAX_AUTOPILOT, MAX_BULK

MODES = ("autopilot", "agent", "bulk")


class IntakeError(Exception):
    """A user-facing validation error. Shown on the form, not a 500."""


def normalize_mode(mode: str) -> str:
    cleaned = (mode or "").strip().lower()
    if cleaned not in MODES:
        raise IntakeError("Unknown mode. Choose autopilot, agent, or bulk.")
    return cleaned


def company_context(
    name: str,
    product: str,
    website: str,
    value_props: str,
    icp_description: str,
) -> CompanyContext:
    cleaned_name = (name or "").strip()
    cleaned_product = (product or "").strip()
    cleaned_values = (value_props or "").strip()
    cleaned_icp = (icp_description or "").strip()
    missing = []
    if not cleaned_name:
        missing.append("company name")
    if not cleaned_product:
        missing.append("product name")
    if not cleaned_values:
        missing.append("value props")
    if not cleaned_icp:
        missing.append("ICP description")
    if missing:
        raise IntakeError("Please fill in: " + ", ".join(missing) + ".")
    return CompanyContext(
        name=cleaned_name,
        product=cleaned_product,
        website=(website or "").strip(),
        value_props=cleaned_values,
        icp_description=cleaned_icp,
    )


def require_api_key(mode: str) -> None:
    if mode == "bulk":
        if not os.getenv("GROQ_API_KEY"):
            raise IntakeError(
                "GROQ_API_KEY is not set. Add it to your .env file and restart the server."
            )
        return
    if mode in ("agent", "autopilot"):
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise IntakeError(
                "ANTHROPIC_API_KEY is not set. Add it to your .env file and restart the server."
            )
        return
    raise IntakeError("Unknown mode. Choose autopilot, agent, or bulk.")


def parse_prospect_count(raw: str) -> int:
    text = (raw or "").strip()
    try:
        count = int(text)
    except ValueError:
        raise IntakeError(f"Prospect count must be a whole number from 1 to {MAX_AUTOPILOT}.")
    if count < 1 or count > MAX_AUTOPILOT:
        raise IntakeError(f"Prospect count must be between 1 and {MAX_AUTOPILOT}.")
    return count


def prospects_from_csv(content: bytes, mode: str) -> list[Prospect]:
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise IntakeError("CSV must be UTF-8 encoded.")
    if not text.strip():
        raise IntakeError("The CSV file is empty.")

    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise IntakeError("CSV must include a header row with name and company columns.")
    fields = {(name or "").strip().lower() for name in reader.fieldnames}
    if "name" not in fields or "company" not in fields:
        raise IntakeError("CSV must include name and company columns.")

    prospects: list[Prospect] = []
    for line_number, row in enumerate(reader, start=2):
        normalized = {
            (key or "").strip().lower(): (value or "").strip()
            for key, value in row.items()
            if key is not None
        }
        name = normalized.get("name", "")
        company = normalized.get("company", "")
        if not any(normalized.values()):
            continue
        if not name or not company:
            raise IntakeError(f"Row {line_number} must include both name and company.")
        prospects.append(Prospect(
            name=name,
            title=normalized.get("title", ""),
            company=company,
            industry=normalized.get("industry", ""),
            website=normalized.get("website", ""),
        ))

    if not prospects:
        raise IntakeError("No prospects found. Please upload a CSV or add at least one prospect manually.")
    return _cap(prospects, mode)


def prospects_from_manual(
    names: list[str],
    titles: list[str],
    companies: list[str],
    industries: list[str],
    websites: list[str],
    mode: str,
) -> list[Prospect]:
    prospects: list[Prospect] = []
    for index in range(len(names)):
        name = _at(names, index)
        company = _at(companies, index)
        title = _at(titles, index)
        industry = _at(industries, index)
        website = _at(websites, index)
        if not any((name, company, title, industry, website)):
            continue
        if not name or not company:
            raise IntakeError(f"Prospect {index + 1} must include both name and company.")
        prospects.append(Prospect(
            name=name,
            title=title,
            company=company,
            industry=industry,
            website=website,
        ))

    if not prospects:
        raise IntakeError("No prospects found. Please upload a CSV or add at least one prospect manually.")
    return _cap(prospects, mode)


def _at(values: list[str], index: int) -> str:
    if index >= len(values):
        return ""
    return (values[index] or "").strip()


def _cap(prospects: list[Prospect], mode: str) -> list[Prospect]:
    limit = MAX_BULK if mode == "bulk" else MAX_AGENT
    if len(prospects) > limit:
        label = "Bulk" if mode == "bulk" else "Agent"
        raise IntakeError(f"{label} mode accepts at most {limit} prospects (got {len(prospects)}).")
    return prospects
