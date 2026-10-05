"""Reliability coverage for validation, parsing, jobs, and bounded runners."""

import asyncio
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

_db_dir = tempfile.mkdtemp(prefix="outreach-jobs-")
os.environ["JOBS_DB"] = str(Path(_db_dir) / "jobs.db")
os.environ.pop("GROQ_API_KEY", None)
os.environ.pop("ANTHROPIC_API_KEY", None)

from fastapi.testclient import TestClient

from core import jobs
from core.intake import IntakeError, prospects_from_csv, prospects_from_manual
from core.models import CompanyContext, LeadScore, Prospect
from core.output import OutputParseError, parse_email, parse_score, results_to_csv, dict_to_result
from core.reliability import (
    BULK_CONCURRENCY,
    is_transient,
    with_retries,
)
from webapp.main import app, _run_agent, _run_bulk


def _prospect(name: str, company: str) -> Prospect:
    return Prospect(name=name, title="CISO", company=company, industry="Finance", website="https://example.com")


def _ctx() -> CompanyContext:
    return CompanyContext(
        name="Acme",
        product="Widget",
        website="https://acme.example",
        value_props="Saves time",
        icp_description="CISOs at banks",
    )


def _form(**overrides):
    data = {
        "company_name": "Acme",
        "product_name": "Widget",
        "company_website": "https://acme.example",
        "value_props": "Saves time",
        "icp_description": "CISOs at banks",
        "mode": "bulk",
        "prospect_count": "5",
        "prospect_name": "Jane Smith",
        "prospect_title": "CISO",
        "prospect_company": "Nike",
        "prospect_industry": "Retail",
        "prospect_website": "https://nike.com",
    }
    data.update(overrides)
    return data


class ParseTests(unittest.TestCase):
    def test_parse_email_requires_subject_and_body(self):
        subject, body = parse_email("Subject: Hello there\n\nFirst paragraph.")
        self.assertEqual(subject, "Hello there")
        self.assertIn("First paragraph", body)

    def test_parse_email_rejects_empty_and_missing_subject(self):
        with self.assertRaises(OutputParseError):
            parse_email("   ")
        with self.assertRaises(OutputParseError):
            parse_email("Just a paragraph with no subject label.")
        with self.assertRaises(OutputParseError):
            parse_email("Subject: Only a subject")

    def test_parse_score_requires_every_label(self):
        parsed = parse_score(
            "TOTAL: 8.5\nICP_FIT: 9\nTIMING: 8\nREACHABILITY: 7\n"
            "SIGNALS:\n- Merger\nRATIONALE: Strong fit."
        )
        self.assertEqual(parsed["total"], 8.5)
        self.assertEqual(parsed["signals"], ["Merger"])

    def test_parse_score_does_not_invent_a_five(self):
        with self.assertRaises(OutputParseError):
            parse_score("")
        with self.assertRaises(OutputParseError):
            parse_score("The company looks interesting but here is no score.")


class IntakeTests(unittest.TestCase):
    def test_csv_requires_utf8_and_columns(self):
        with self.assertRaises(IntakeError):
            prospects_from_csv(b"\xff\xfe not utf8", "bulk")
        with self.assertRaises(IntakeError):
            prospects_from_csv(b"title,industry\nCISO,Bank\n", "bulk")

    def test_csv_cap(self):
        rows = "name,company\n" + "".join(f"Person {i},Co {i}\n" for i in range(101))
        with self.assertRaises(IntakeError) as caught:
            prospects_from_csv(rows.encode(), "bulk")
        self.assertIn("100", str(caught.exception))

    def test_manual_requires_name_and_company(self):
        with self.assertRaises(IntakeError):
            prospects_from_manual(["Jane"], ["CISO"], [""], ["Bank"], [""], "agent")
        found = prospects_from_manual(
            ["Jane Smith"], ["CISO"], ["Nike"], ["Retail"], ["https://nike.com"], "agent"
        )
        self.assertEqual(found[0].company, "Nike")


class RetryTests(unittest.TestCase):
    def test_transient_classification(self):
        self.assertTrue(is_transient(ConnectionResetError("connection reset by peer")))
        self.assertFalse(is_transient(TimeoutError("timed out")))
        self.assertFalse(is_transient(OutputParseError("missing subject")))

        class RateLimit(Exception):
            status_code = 429

        self.assertTrue(is_transient(RateLimit("slow down")))
        self.assertFalse(is_transient(type("Missing", (Exception,), {"status_code": 400})()))

    def test_retries_stop_on_parse_errors_and_timeouts(self):
        attempts = {"n": 0}

        async def fail_parse():
            attempts["n"] += 1
            raise OutputParseError("Model response did not include a subject line.")

        with self.assertRaises(OutputParseError):
            asyncio.run(with_retries(fail_parse))
        self.assertEqual(attempts["n"], 1)

        attempts["n"] = 0

        async def fail_timeout():
            attempts["n"] += 1
            raise TimeoutError("Email generation timed out after 4 minutes.")

        with self.assertRaises(TimeoutError):
            asyncio.run(with_retries(fail_timeout))
        self.assertEqual(attempts["n"], 1)

    def test_retries_transient_then_succeeds(self):
        attempts = {"n": 0}

        async def flaky():
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise ConnectionResetError("connection reset by peer")
            return "ok"

        with patch("core.reliability.RETRY_BASE_DELAY_SECONDS", 0):
            result = asyncio.run(with_retries(flaky))
        self.assertEqual(result, "ok")
        self.assertEqual(attempts["n"], 3)


class JobStoreTests(unittest.TestCase):
    def test_results_survive_and_interrupted_jobs_fail(self):
        jobs.init_db()
        job_id = jobs.create_job("bulk", 2)
        jobs.mark_running(job_id)
        jobs.append_result(job_id, {"name": "Jane", "company": "Nike", "error": ""}, 1)
        jobs.append_result(job_id, {"name": "Bob", "company": "Stripe", "error": ""}, 0)
        jobs.mark_done(job_id)
        stored = jobs.get_job(job_id)
        self.assertEqual([item["name"] for item in stored["results"]], ["Bob", "Jane"])
        self.assertEqual(stored["completed"], 2)

        running_id = jobs.create_job("agent", 1)
        jobs.mark_running(running_id)
        failed = jobs.fail_interrupted_jobs()
        self.assertGreaterEqual(failed, 1)
        interrupted = jobs.get_job(running_id)
        self.assertEqual(interrupted["status"], "failed")
        self.assertIn("Server restarted", interrupted["error"])
        # A finished job is left alone.
        self.assertEqual(jobs.get_job(job_id)["status"], "done")

    def test_first_call_does_not_deadlock_before_init(self):
        import subprocess
        import sys
        db = Path(_db_dir) / "fresh.db"
        code = (
            "import os\n"
            f"os.environ['JOBS_DB'] = {str(db)!r}\n"
            "from core import jobs\n"
            "job_id = jobs.create_job('bulk', 1)\n"
            "print(jobs.get_job(job_id)['status'])\n"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd="/workspace",
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("queued", completed.stdout)


class RunnerTests(unittest.TestCase):
    def test_bulk_runs_are_bounded_and_off_the_event_loop(self):
        jobs.init_db()
        prospects = [_prospect(f"Person {i}", f"Co {i}") for i in range(8)]
        state = {"in_flight": 0, "peak": 0}
        lock = threading.Lock()

        def slow_generate(prospect, ctx, client):
            with lock:
                state["in_flight"] += 1
                state["peak"] = max(state["peak"], state["in_flight"])
            time.sleep(0.2)
            with lock:
                state["in_flight"] -= 1
            return "Hello", f"Body for {prospect.name}"

        job_id = jobs.create_job("bulk", len(prospects))
        started = time.perf_counter()
        with patch("core.pipeline.generate_email", slow_generate):
            asyncio.run(_run_bulk(job_id, prospects, _ctx()))
        elapsed = time.perf_counter() - started

        stored = jobs.get_job(job_id)
        self.assertEqual(stored["completed"], 8)
        self.assertLessEqual(state["peak"], BULK_CONCURRENCY)
        self.assertGreaterEqual(state["peak"], 2)
        self.assertLess(elapsed, 1.2)
        self.assertEqual(stored["results"][0]["email_subject"], "Hello")

    def test_bulk_does_not_retry_parse_failures(self):
        jobs.init_db()
        calls = {"n": 0}

        def bad_generate(prospect, ctx, client):
            calls["n"] += 1
            raise OutputParseError("Model response did not include a subject line.")

        job_id = jobs.create_job("bulk", 1)
        with patch("core.pipeline.generate_email", bad_generate):
            asyncio.run(_run_bulk(job_id, [_prospect("Jane", "Nike")], _ctx()))
        stored = jobs.get_job(job_id)
        self.assertEqual(calls["n"], 1)
        self.assertIn("subject", stored["results"][0]["error"])

    def test_agent_concurrency_is_bounded(self):
        jobs.init_db()
        prospects = [_prospect(f"Person {i}", f"Co {i}") for i in range(6)]
        state = {"in_flight": 0, "peak": 0}
        lock = asyncio.Lock()

        async def fake_score(prospect, ctx):
            async with lock:
                state["in_flight"] += 1
                state["peak"] = max(state["peak"], state["in_flight"])
            await asyncio.sleep(0.2)
            async with lock:
                state["in_flight"] -= 1
            return LeadScore(
                total=8, icp_fit=8, timing=8, reachability=8,
                signals=["Merger"], rationale="Fit", skip=False,
            )

        async def fake_email(prospect, ctx, past_emails=None):
            return "Hello", "Body"

        job_id = jobs.create_job("agent", len(prospects))
        started = time.perf_counter()
        with patch("core.agent.score_lead", fake_score), patch("core.agent.research_and_email", fake_email):
            asyncio.run(_run_agent(job_id, prospects, _ctx()))
        elapsed = time.perf_counter() - started
        stored = jobs.get_job(job_id)
        self.assertEqual(stored["completed"], 6)
        self.assertLessEqual(state["peak"], 2)
        self.assertGreaterEqual(state["peak"], 2)
        self.assertLess(elapsed, 1.2)


class WebTests(unittest.TestCase):
    def setUp(self):
        os.environ.pop("GROQ_API_KEY", None)
        os.environ.pop("ANTHROPIC_API_KEY", None)
        self.client_cm = TestClient(app)
        self.client = self.client_cm.__enter__()

    def tearDown(self):
        self.client_cm.__exit__(None, None, None)
        os.environ.pop("GROQ_API_KEY", None)
        os.environ.pop("ANTHROPIC_API_KEY", None)

    def test_health_reports_key_presence_only(self):
        os.environ["GROQ_API_KEY"] = "secret-groq"
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["ok"])
        self.assertTrue(body["groq_api_key"])
        self.assertFalse(body["anthropic_api_key"])
        self.assertNotIn("secret-groq", response.text)

    def test_missing_key_and_bad_csv_stay_on_the_form(self):
        missing = self.client.post("/run", data=_form())
        self.assertEqual(missing.status_code, 200)
        self.assertIn("GROQ_API_KEY is not set", missing.text)

        os.environ["GROQ_API_KEY"] = "present"
        bad_csv = self.client.post(
            "/run",
            data=_form(prospect_name=""),
            files={"csv_file": ("prospects.csv", b"\xff\xfe", "text/csv")},
        )
        self.assertEqual(bad_csv.status_code, 200)
        self.assertIn("UTF-8", bad_csv.text)

        unknown = self.client.post("/run", data=_form(mode="nope"))
        self.assertEqual(unknown.status_code, 200)
        self.assertIn("Unknown mode", unknown.text)

    def test_job_is_stored_and_readable_after_the_request(self):
        os.environ["GROQ_API_KEY"] = "present"

        async def fake_bulk(job_id, prospects, ctx):
            jobs.append_result(job_id, {
                "name": prospects[0].name,
                "title": prospects[0].title,
                "company": prospects[0].company,
                "industry": prospects[0].industry,
                "website": prospects[0].website,
                "mode": "bulk",
                "skipped": False,
                "skip_reason": "",
                "score_total": None,
                "signals": [],
                "score_rationale": "",
                "email_subject": "Hello Nike",
                "email_body": "Body",
                "date_generated": "2026-01-01T00:00:00+00:00",
                "error": "",
            }, 0)

        with patch("webapp.main._run_bulk", fake_bulk):
            created = self.client.post("/run", data=_form(), follow_redirects=False)

        self.assertEqual(created.status_code, 303)
        location = created.headers["location"]
        job_id = location.rsplit("/", 1)[-1]
        status = self.client.get(f"/jobs/{job_id}")
        self.assertEqual(status.status_code, 200)
        payload = status.json()
        self.assertEqual(payload["status"], "done")
        self.assertEqual(payload["results"][0]["email_subject"], "Hello Nike")

        page = self.client.get(location)
        self.assertEqual(page.status_code, 200)
        self.assertIn("Hello Nike", page.text)
        self.assertNotIn("setInterval", page.text)

        download = self.client.get(f"/download/{job_id}")
        self.assertEqual(download.status_code, 200)
        self.assertIn("Hello Nike", download.text)
        rebuilt = dict_to_result(payload["results"][0])
        self.assertIn("Hello Nike", results_to_csv([rebuilt]))

    def test_running_page_polls_until_the_job_finishes(self):
        job_id = jobs.create_job("bulk", 2)
        jobs.mark_running(job_id)
        jobs.set_current(job_id, "Jane Smith @ Nike")
        page = self.client.get(f"/results/{job_id}")
        self.assertEqual(page.status_code, 200)
        self.assertIn("Run in progress", page.text)
        self.assertIn(job_id, page.text)
        self.assertIn("'/jobs/' + jobId", page.text)
        self.assertIn("setInterval", page.text)

        jobs.mark_failed(job_id, "Server restarted before this run finished.")
        failed = self.client.get(f"/results/{job_id}")
        self.assertIn("Server restarted before this run finished.", failed.text)
        self.assertNotIn("setInterval", failed.text)

    def test_missing_job_is_not_found(self):
        self.assertEqual(self.client.get("/jobs/missing").status_code, 404)
        self.assertEqual(self.client.get("/results/missing").status_code, 404)
        self.assertEqual(self.client.get("/download/missing").status_code, 404)


if __name__ == "__main__":
    unittest.main()
