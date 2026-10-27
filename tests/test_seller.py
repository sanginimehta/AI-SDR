"""Seller website learning for Autopilot product understanding."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

_db_dir = tempfile.mkdtemp(prefix="outreach-seller-")
os.environ["JOBS_DB"] = str(Path(_db_dir) / "seller-jobs.db")
os.environ.pop("GROQ_API_KEY", None)
os.environ.pop("ANTHROPIC_API_KEY", None)

from fastapi.testclient import TestClient

from core import jobs
from core.intake import IntakeError, company_context
from core.models import CompanyContext, Prospect
from core.seller import (
    SellerSummary,
    enrich_company_context,
    seller_prompt_block,
    summarize_seller_site,
)
from webapp.main import app, _run_autopilot


class IntakeSellerTests(unittest.TestCase):
    def test_autopilot_allows_empty_value_props_with_website(self):
        ctx = company_context(
            "LogicGate",
            "Risk Cloud",
            "https://logicgate.com",
            "",
            "CISOs at regional banks",
            mode="autopilot",
        )
        self.assertEqual(ctx.website, "https://logicgate.com")
        self.assertEqual(ctx.value_props, "")

    def test_autopilot_requires_website_or_value_props(self):
        with self.assertRaises(IntakeError) as caught:
            company_context(
                "LogicGate",
                "Risk Cloud",
                "",
                "",
                "CISOs at regional banks",
                mode="autopilot",
            )
        self.assertIn("value props or company website", str(caught.exception))

    def test_bulk_still_requires_value_props(self):
        with self.assertRaises(IntakeError):
            company_context(
                "LogicGate",
                "Risk Cloud",
                "https://logicgate.com",
                "",
                "CISOs at regional banks",
                mode="bulk",
            )


class SellerSummaryTests(unittest.TestCase):
    def test_fallback_summary_from_homepage_text(self):
        html = """
        <html><head>
          <title>Risk Cloud | LogicGate</title>
          <meta name="description" content="No-code GRC for regulated industries.">
        </head><body>
          <h1>Unify risk and compliance</h1>
          <p>Risk Cloud helps banks automate evidence collection.</p>
        </body></html>
        """
        response = MagicMock()
        response.text = html
        response.raise_for_status = MagicMock()

        with patch("core.seller.requests.get", return_value=response):
            summary = summarize_seller_site(
                "logicgate.com",
                company_name="LogicGate",
                product_hint="Risk Cloud",
            )

        self.assertTrue(summary.ok)
        self.assertIn("GRC", summary.product_summary)
        self.assertIn("logicgate.com", summary.value_props)
        self.assertTrue(summary.source_url.startswith("https://"))

    def test_enrich_fills_short_value_props_without_overwriting_icp(self):
        ctx = CompanyContext(
            name="LogicGate",
            product="Risk Cloud",
            website="https://logicgate.com",
            value_props="short",
            icp_description="CISOs at banks",
        )
        summary = SellerSummary(
            product_summary="No-code GRC platform",
            value_props="- Automates evidence\n- Maps to FFIEC",
            suggested_icp="Should not replace user ICP",
            product_name="Risk Cloud",
            source_url="https://logicgate.com",
        )
        enriched = enrich_company_context(ctx, summary)
        self.assertEqual(enriched.icp_description, "CISOs at banks")
        self.assertIn("Learned from https://logicgate.com", enriched.value_props)
        self.assertIn("Automates evidence", enriched.value_props)

    def test_enrich_appends_when_user_wrote_long_value_props(self):
        long_props = "A" * 120
        ctx = CompanyContext(
            name="LogicGate",
            product="Risk Cloud",
            website="https://logicgate.com",
            value_props=long_props,
            icp_description="CISOs at banks",
        )
        summary = SellerSummary(
            value_props="Website-learned bullet",
            source_url="https://logicgate.com",
        )
        enriched = enrich_company_context(ctx, summary)
        self.assertTrue(enriched.value_props.startswith(long_props))
        self.assertIn("Website-learned bullet", enriched.value_props)

    def test_seller_prompt_block_includes_product_context(self):
        block = seller_prompt_block(CompanyContext(
            name="LogicGate",
            product="Risk Cloud",
            website="https://logicgate.com",
            value_props="No-code GRC",
            icp_description="CISOs",
        ))
        self.assertIn("Risk Cloud", block)
        self.assertIn("No-code GRC", block)
        self.assertIn("https://logicgate.com", block)


class AutopilotSellerFlowTests(unittest.TestCase):
    def test_run_enriches_context_before_discovery(self):
        jobs.init_db()
        job_id = jobs.create_job("autopilot", 1)
        seen = {}

        async def fake_discover(ctx, count=5):
            seen["value_props"] = ctx.value_props
            seen["product"] = ctx.product
            return [
                Prospect(
                    name="Sarah Chen",
                    title="CISO",
                    company="Pinnacle",
                    industry="Financial Services",
                    website="https://pnfp.com",
                )
            ]

        async def fake_score_email(job_id, prospects, ctx, mode):
            seen["score_value_props"] = ctx.value_props
            jobs.append_result(job_id, {
                "name": prospects[0].name,
                "title": prospects[0].title,
                "company": prospects[0].company,
                "industry": prospects[0].industry,
                "website": prospects[0].website,
                "mode": mode,
                "skipped": False,
                "skip_reason": "",
                "score_total": 8,
                "signals": [],
                "score_rationale": "",
                "email_subject": "Hello",
                "email_body": "Body",
                "date_generated": "2026-01-01T00:00:00+00:00",
                "error": "",
            }, 0)

        summary = SellerSummary(
            product_summary="No-code GRC",
            value_props="Automates exams",
            product_name="Risk Cloud",
            source_url="https://logicgate.com",
        )
        ctx = CompanyContext(
            name="LogicGate",
            product="Risk Cloud",
            website="https://logicgate.com",
            value_props="",
            icp_description="CISOs at banks",
        )

        with patch("core.seller.summarize_seller_site", return_value=summary), \
             patch("core.agent.discover_prospects", fake_discover), \
             patch("webapp.main._score_and_email_all", fake_score_email):
            import asyncio
            asyncio.run(_run_autopilot(job_id, ctx, 1))

        self.assertIn("Automates exams", seen["value_props"])
        self.assertIn("Automates exams", seen["score_value_props"])
        stored = jobs.get_job(job_id)
        self.assertEqual(stored["completed"], 1)

    def test_failed_scrape_with_empty_value_props_stops_cleanly(self):
        jobs.init_db()
        job_id = jobs.create_job("autopilot", 1)
        summary = SellerSummary(scrape_error="Connection timed out", source_url="https://bad.example")
        ctx = CompanyContext(
            name="Acme",
            product="Widget",
            website="https://bad.example",
            value_props="",
            icp_description="CISOs",
        )
        with patch("core.seller.summarize_seller_site", return_value=summary):
            import asyncio
            asyncio.run(_run_autopilot(job_id, ctx, 2))
        stored = jobs.get_job(job_id)
        self.assertEqual(stored["completed"], 1)
        self.assertIn("Could not learn your product", stored["results"][0]["error"])

    def test_form_accepts_autopilot_without_value_props_when_website_set(self):
        os.environ["ANTHROPIC_API_KEY"] = "present"
        client = TestClient(app)

        async def fake_execute(job_id, mode, ctx, prospects, count):
            jobs.mark_running(job_id)
            jobs.mark_done(job_id)

        with patch("webapp.main._execute_job", fake_execute):
            response = client.post(
                "/run",
                data={
                    "company_name": "LogicGate",
                    "product_name": "Risk Cloud",
                    "company_website": "https://logicgate.com",
                    "value_props": "",
                    "icp_description": "CISOs at regional banks",
                    "mode": "autopilot",
                    "prospect_count": "3",
                },
                follow_redirects=False,
            )
        self.assertEqual(response.status_code, 303)
        self.assertIn("/results/", response.headers["location"])
        os.environ.pop("ANTHROPIC_API_KEY", None)


if __name__ == "__main__":
    unittest.main()
