"""
CSV and CRM-ready output formatting.
Columns are chosen to map cleanly into HubSpot, Salesforce, or Pipedrive.
"""

import csv
import io
import re
from .models import LeadScore, OutreachResult, Prospect


class OutputParseError(ValueError):
    """Model text could not be turned into a score or an email."""


def parse_email(raw: str) -> tuple[str, str]:
    """
    Split raw agent/pipeline output into (subject, body).
    Handles 'Subject: ...' lines and markdown bold subject lines.

    Raises OutputParseError when the reply is empty or has no subject or body.
    A missing subject is a failure, not a guess from the first line.
    """
    if not raw or not raw.strip():
        raise OutputParseError("Model returned an empty email.")

    lines = raw.strip().splitlines()
    subject = ""
    body_lines = []
    found_subject = False

    for line in lines:
        clean = line.strip()
        # Match "Subject: ..." or "**Subject: ...**"
        m = re.match(r"\*{0,2}Subject\s*:\s*\*{0,2}(.+)\*{0,2}", clean, re.IGNORECASE)
        if m and not found_subject:
            subject = m.group(1).strip().strip("*").strip()
            found_subject = True
            continue
        if found_subject:
            body_lines.append(line)

    if not found_subject or not subject:
        raise OutputParseError("Model response did not include a subject line.")

    body = "\n".join(body_lines).strip()
    if not body:
        raise OutputParseError("Model response did not include an email body.")
    return subject, body


def parse_score(text: str) -> dict:
    """
    Parse a lead-score reply into numeric fields.

    Raises OutputParseError when the reply is empty or any required score
    label is missing. Missing numbers are not filled in with a default.
    """
    if not text or not text.strip():
        raise OutputParseError("Model returned an empty score response.")

    def _num(pattern: str):
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            return None
        try:
            return float(match.group(1))
        except ValueError:
            return None

    scores = {
        "TOTAL": _num(r"TOTAL\s*:\s*([\d.]+)"),
        "ICP_FIT": _num(r"ICP_FIT\s*:\s*([\d.]+)"),
        "TIMING": _num(r"TIMING\s*:\s*([\d.]+)"),
        "REACHABILITY": _num(r"REACHABILITY\s*:\s*([\d.]+)"),
    }
    missing = [name for name, value in scores.items() if value is None]
    if missing:
        raise OutputParseError("Score response was missing: " + ", ".join(missing) + ".")

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
    rationale_match = re.search(r"RATIONALE\s*:(.*)", text, re.IGNORECASE | re.DOTALL)
    if rationale_match:
        rationale = rationale_match.group(1).strip()

    return {
        "total": scores["TOTAL"],
        "icp_fit": scores["ICP_FIT"],
        "timing": scores["TIMING"],
        "reachability": scores["REACHABILITY"],
        "signals": signals or ["No specific signals found"],
        "rationale": rationale,
    }


def results_to_csv(results: list[OutreachResult]) -> str:
    """
    Serialise a list of OutreachResult objects to a CSV string.
    Column names match common CRM import field names.
    """
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=[
        "first_name",
        "last_name",
        "title",
        "company",
        "industry",
        "website",
        "mode",
        "skipped",
        "skip_reason",
        "score_total",
        "score_icp_fit",
        "score_timing",
        "score_reachability",
        "timing_signals",
        "score_rationale",
        "email_subject",
        "email_body",
        "date_generated",
        "error",
    ])
    writer.writeheader()

    for r in results:
        p = r.prospect
        s = r.score

        writer.writerow({
            "first_name":          p.first_name,
            "last_name":           p.last_name,
            "title":               p.title,
            "company":             p.company,
            "industry":            p.industry,
            "website":             p.website,
            "mode":                r.mode,
            "skipped":             "yes" if r.skipped else "no",
            "skip_reason":         r.skip_reason,
            "score_total":         s.total if s else "",
            "score_icp_fit":       s.icp_fit if s else "",
            "score_timing":        s.timing if s else "",
            "score_reachability":  s.reachability if s else "",
            "timing_signals":      " | ".join(s.signals) if s else "",
            "score_rationale":     s.rationale if s else "",
            "email_subject":       r.email_subject,
            "email_body":          r.email_body,
            "date_generated":      r.date_generated,
            "error":               r.error,
        })

    return output.getvalue()


def dict_to_result(data: dict) -> OutreachResult:
    """Rebuild an OutreachResult from a stored result dict."""
    score = None
    if data.get("score_total") is not None:
        score = LeadScore(
            total=data.get("score_total") or 0,
            icp_fit=data.get("score_icp_fit") or 0,
            timing=data.get("score_timing") or 0,
            reachability=data.get("score_reachability") or 0,
            signals=data.get("signals") or [],
            rationale=data.get("score_rationale") or "",
            skip=bool(data.get("skipped")),
        )
    name = data.get("name") or f"{data.get('first_name', '')} {data.get('last_name', '')}".strip()
    return OutreachResult(
        prospect=Prospect(
            name=name or "",
            title=data.get("title") or "",
            company=data.get("company") or "",
            industry=data.get("industry") or "",
            website=data.get("website") or "",
        ),
        mode=data.get("mode") or "",
        skipped=bool(data.get("skipped")),
        skip_reason=data.get("skip_reason") or "",
        score=score,
        email_subject=data.get("email_subject") or "",
        email_body=data.get("email_body") or "",
        date_generated=data.get("date_generated") or "",
        error=data.get("error") or "",
    )


def result_to_dict(r: OutreachResult) -> dict:
    """Serialise a single OutreachResult to a plain dict (for Jinja2 templates)."""
    p = r.prospect
    s = r.score
    return {
        "name":              p.name,
        "first_name":        p.first_name,
        "title":             p.title,
        "company":           p.company,
        "industry":          p.industry,
        "website":           p.website,
        "mode":              r.mode,
        "skipped":           r.skipped,
        "skip_reason":       r.skip_reason,
        "score_total":       s.total if s else None,
        "score_icp_fit":     s.icp_fit if s else None,
        "score_timing":      s.timing if s else None,
        "score_reachability":s.reachability if s else None,
        "signals":           s.signals if s else [],
        "score_rationale":   s.rationale if s else "",
        "email_subject":     r.email_subject,
        "email_body":        r.email_body,
        "date_generated":    r.date_generated,
        "error":             r.error,
    }
