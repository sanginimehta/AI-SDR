"""
FastAPI web app — form-driven interface for the AI outreach pipeline.
Supports bulk (Groq) and agent (Claude) modes, CSV upload or manual entry.

Runs are validated up front, executed in the background, and stored in SQLite
so a slow model call or a process restart does not lose the job.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from core import jobs
from core.intake import (
    IntakeError,
    company_context,
    normalize_mode,
    parse_prospect_count,
    prospects_from_csv,
    prospects_from_manual,
    require_api_key,
)
from core.models import CompanyContext, Prospect, OutreachResult
from core.output import dict_to_result, results_to_csv, result_to_dict
from core.reliability import (
    AGENT_CONCURRENCY,
    BULK_CONCURRENCY,
    BULK_EMAIL_TIMEOUT_SECONDS,
    configure_logging,
    job_id_var,
    with_retries,
)

load_dotenv()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    jobs.init_db()
    interrupted = jobs.fail_interrupted_jobs()
    if interrupted:
        logger.info("marked %s interrupted job(s) as failed", interrupted)
    yield


app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="webapp/templates")


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _form_error(request: Request, message: str):
    return templates.TemplateResponse(request, "index.html", {"error": message})


def _counts(results: list[dict]) -> dict:
    emailed = [item for item in results if not item.get("skipped") and not item.get("error")]
    skipped = [item for item in results if item.get("skipped")]
    errored = [item for item in results if item.get("error")]
    return {
        "results": results,
        "total": len(results),
        "emailed_count": len(emailed),
        "skipped_count": len(skipped),
        "errored_count": len(errored),
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "ok": True,
        "groq_api_key": bool(os.getenv("GROQ_API_KEY")),
        "anthropic_api_key": bool(os.getenv("ANTHROPIC_API_KEY")),
    }


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/run")
async def run(
    request: Request,
    background_tasks: BackgroundTasks,
    company_name: str = Form(...),
    product_name: str = Form(...),
    company_website: str = Form(""),
    value_props: str = Form(""),
    icp_description: str = Form(...),
    mode: str = Form(...),
    prospect_count: str = Form(default="5"),
    csv_file: UploadFile = File(None),
    prospect_name: list[str] = Form(default=[]),
    prospect_title: list[str] = Form(default=[]),
    prospect_company: list[str] = Form(default=[]),
    prospect_industry: list[str] = Form(default=[]),
    prospect_website: list[str] = Form(default=[]),
):
    try:
        selected = normalize_mode(mode)
        ctx = company_context(
            company_name,
            product_name,
            company_website,
            value_props,
            icp_description,
            mode=selected,
        )
        require_api_key(selected)
        prospects: list[Prospect] = []
        count = 0
        if selected == "autopilot":
            count = parse_prospect_count(prospect_count)
        elif csv_file and csv_file.filename:
            prospects = prospects_from_csv(await csv_file.read(), selected)
        else:
            prospects = prospects_from_manual(
                prospect_name,
                prospect_title,
                prospect_company,
                prospect_industry,
                prospect_website,
                selected,
            )
    except IntakeError as exc:
        return _form_error(request, str(exc))

    total = count if selected == "autopilot" else len(prospects)
    job_id = jobs.create_job(selected, total)
    background_tasks.add_task(_execute_job, job_id, selected, ctx, prospects, count)
    logger.info("job=%s queued mode=%s total=%s", job_id, selected, total)
    return RedirectResponse(f"/results/{job_id}", status_code=303)


@app.get("/jobs/{job_id}")
async def job_status(job_id: str):
    job = jobs.get_job(job_id)
    if job is None:
        return JSONResponse({"detail": "Results not found. They may have expired."}, status_code=404)
    return JSONResponse(
        {
            "id": job["id"],
            "status": job["status"],
            "mode": job["mode"],
            "total": job["total"],
            "completed": job["completed"],
            "current_prospect": job["current_prospect"],
            "error": job["error"],
            "results": job["results"],
        },
        headers={"Cache-Control": "no-store"},
    )


@app.get("/results/{job_id}", response_class=HTMLResponse)
async def show_results(request: Request, job_id: str):
    job = jobs.get_job(job_id)
    if job is None:
        return HTMLResponse("Results not found. They may have expired.", status_code=404)

    page = templates.TemplateResponse(request, "results.html", {
        "job_id": job_id,
        "job_status": job["status"],
        "job_error": job["error"],
        "completed": job["completed"],
        "planned_total": job["total"],
        "current_prospect": job["current_prospect"],
        **_counts(job["results"]),
    })
    page.headers["Cache-Control"] = "no-store"
    return page


@app.get("/download/{job_id}")
async def download_csv(job_id: str):
    job = jobs.get_job(job_id)
    if job is None:
        return Response("Results not found.", status_code=404)

    csv_content = results_to_csv([dict_to_result(item) for item in job["results"]])
    filename = f"outreach_{job_id[:8]}.csv"
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.get("/sample-csv")
async def sample_csv():
    content = "name,title,company,industry,website\n"
    content += "Sarah Chen,Chief Information Security Officer,Pinnacle Financial Partners,Financial Services,https://www.pnfp.com\n"
    content += "Marcus Webb,IT Director,Denny's Corporation,Food & Beverage,https://www.dennys.com\n"
    return Response(
        content=content,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=sample_prospects.csv"},
    )


# ---------------------------------------------------------------------------
# Pipeline runners
# ---------------------------------------------------------------------------

async def _execute_job(
    job_id: str,
    mode: str,
    ctx: CompanyContext,
    prospects: list[Prospect],
    count: int,
) -> None:
    token = job_id_var.set(job_id)
    try:
        jobs.mark_running(job_id)
        logger.info("job=%s started mode=%s", job_id, mode)
        if mode == "autopilot":
            await _run_autopilot(job_id, ctx, count)
        elif mode == "bulk":
            await _run_bulk(job_id, prospects, ctx)
        else:
            await _run_agent(job_id, prospects, ctx)
        jobs.mark_done(job_id)
        logger.info("job=%s finished", job_id)
    except Exception as exc:
        logger.exception("job=%s failed", job_id)
        jobs.mark_failed(job_id, str(exc) or "Run failed unexpectedly.")
    finally:
        job_id_var.reset(token)


async def _run_bulk(job_id: str, prospects: list[Prospect], ctx: CompanyContext) -> None:
    from core.pipeline import generate_email

    groq_key = os.getenv("GROQ_API_KEY") or ""
    semaphore = asyncio.Semaphore(BULK_CONCURRENCY)

    async def work(position: int, prospect: Prospect) -> None:
        async with semaphore:
            label = f"{prospect.name} @ {prospect.company}"
            jobs.set_current(job_id, label)
            logger.info("job=%s step=bulk prospect=%s", job_id, label)
            try:
                subject, body = await with_retries(
                    lambda prospect=prospect: _generate_bulk_email(prospect, ctx, groq_key, generate_email)
                )
                result = OutreachResult(
                    prospect=prospect,
                    mode="bulk",
                    email_subject=subject,
                    email_body=body,
                    date_generated=_timestamp(),
                )
            except Exception as exc:
                logger.warning("job=%s step=bulk prospect=%s error=%s", job_id, label, exc)
                result = OutreachResult(prospect=prospect, mode="bulk", error=str(exc))
            jobs.append_result(job_id, result_to_dict(result), position)

    await asyncio.gather(*(work(index, prospect) for index, prospect in enumerate(prospects)))


async def _generate_bulk_email(prospect: Prospect, ctx: CompanyContext, groq_key: str, generate_email):
    from groq import Groq

    def call():
        client = Groq(api_key=groq_key)
        return generate_email(prospect, ctx, client)

    try:
        return await asyncio.wait_for(asyncio.to_thread(call), timeout=BULK_EMAIL_TIMEOUT_SECONDS)
    except TimeoutError:
        raise TimeoutError("Email generation timed out after 4 minutes.") from None


async def _run_autopilot(job_id: str, ctx: CompanyContext, count: int) -> None:
    from core.agent import discover_prospects
    from core.seller import enrich_company_context, summarize_seller_site

    enriched = ctx
    if ctx.website:
        jobs.set_current(job_id, f"Learning product from {ctx.website}")
        logger.info("job=%s step=seller-site url=%s", job_id, ctx.website)
        summary = await asyncio.to_thread(
            summarize_seller_site,
            ctx.website,
            company_name=ctx.name,
            product_hint=ctx.product,
        )
        if summary.scrape_error and not summary.ok:
            logger.warning(
                "job=%s step=seller-site error=%s",
                job_id,
                summary.scrape_error,
            )
            # Keep going with whatever the user typed; do not fail the whole run
            # unless they also left value props empty.
            if not (ctx.value_props or "").strip():
                _store_discovery_error(
                    job_id,
                    "Could not learn your product from the company website "
                    f"({summary.scrape_error}). Add value props manually or fix the URL.",
                )
                return
        else:
            enriched = enrich_company_context(ctx, summary)
            logger.info(
                "job=%s step=seller-site product=%s value_props_len=%s",
                job_id,
                enriched.product,
                len(enriched.value_props or ""),
            )
    elif not (ctx.value_props or "").strip():
        _store_discovery_error(
            job_id,
            "Add a company website or value props so Autopilot knows what you sell.",
        )
        return

    logger.info("job=%s step=discover count=%s", job_id, count)
    try:
        prospects = await discover_prospects(enriched, count=count)
    except Exception as exc:
        logger.warning("job=%s step=discover error=%s", job_id, exc)
        _store_discovery_error(job_id, str(exc))
        return

    if not prospects:
        _store_discovery_error(
            job_id,
            "Agent could not find prospects matching your ICP. Try broadening your ICP description.",
        )
        return

    jobs.set_total(job_id, len(prospects))
    await _score_and_email_all(job_id, prospects, enriched, mode="autopilot")


def _store_discovery_error(job_id: str, message: str) -> None:
    dummy = Prospect(name="Discovery failed", title="", company="", industry="")
    jobs.set_total(job_id, 1)
    jobs.append_result(
        job_id,
        result_to_dict(OutreachResult(prospect=dummy, mode="autopilot", error=message)),
        position=0,
    )


async def _run_agent(job_id: str, prospects: list[Prospect], ctx: CompanyContext) -> None:
    await _score_and_email_all(job_id, prospects, ctx, mode="agent")


async def _score_and_email_all(
    job_id: str,
    prospects: list[Prospect],
    ctx: CompanyContext,
    mode: str,
) -> None:
    from core.agent import SCORE_THRESHOLD, research_and_email, score_lead

    semaphore = asyncio.Semaphore(AGENT_CONCURRENCY)

    async def work(position: int, prospect: Prospect) -> None:
        async with semaphore:
            label = f"{prospect.name} @ {prospect.company}"
            jobs.set_current(job_id, label)
            logger.info("job=%s step=score mode=%s prospect=%s", job_id, mode, label)
            try:
                score = await score_lead(prospect, ctx)
                logger.info(
                    "job=%s step=score prospect=%s total=%s skip=%s",
                    job_id,
                    label,
                    score.total,
                    score.skip,
                )
                if score.skip:
                    result = OutreachResult(
                        prospect=prospect,
                        mode=mode,
                        skipped=True,
                        skip_reason=f"Score {score.total}/10 — below threshold ({SCORE_THRESHOLD})",
                        score=score,
                        date_generated=_timestamp(),
                    )
                else:
                    logger.info("job=%s step=email prospect=%s", job_id, label)
                    subject, body = await research_and_email(prospect, ctx)
                    result = OutreachResult(
                        prospect=prospect,
                        mode=mode,
                        score=score,
                        email_subject=subject,
                        email_body=body,
                        date_generated=_timestamp(),
                    )
            except Exception as exc:
                logger.warning("job=%s step=score prospect=%s error=%s", job_id, label, exc)
                result = OutreachResult(prospect=prospect, mode=mode, error=str(exc))
            jobs.append_result(job_id, result_to_dict(result), position)

    await asyncio.gather(*(work(index, prospect) for index, prospect in enumerate(prospects)))
