from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CompanyContext:
    """Everything about the company doing the outreach — drives all prompts."""
    name: str            # "LogicGate"
    product: str         # "Risk Cloud"
    website: str         # required — scraped when value_props is left blank
    value_props: str     # optional — filled from the company website when empty
    icp_description: str # free-form: ideal customer profile


@dataclass
class Prospect:
    name: str
    title: str
    company: str
    industry: str
    website: str = ""

    @property
    def first_name(self) -> str:
        return self.name.split()[0] if self.name else ""

    @property
    def last_name(self) -> str:
        parts = self.name.split()
        return " ".join(parts[1:]) if len(parts) > 1 else ""


@dataclass
class LeadScore:
    total: float
    icp_fit: float
    timing: float
    reachability: float
    signals: list
    rationale: str
    skip: bool


@dataclass
class OutreachResult:
    prospect: Prospect
    mode: str                        # "bulk" or "agent"
    skipped: bool = False
    skip_reason: str = ""
    score: Optional[LeadScore] = None
    email_subject: str = ""
    email_body: str = ""
    date_generated: str = ""
    error: str = ""
