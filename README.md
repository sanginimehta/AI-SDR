# AI SDR

An AI-powered sales development agent that learns what you sell from your company website, finds prospects that match your ICP, scores them, and writes personalized cold emails grounded in live research.

Built for GTM / growth workflows. Stack: Claude Agent SDK + FastAPI + Tailwind CSS.

**[Try the web app](#how-to-run-it)**

---

## The problem it solves

A typical SDR spends 30–45 minutes researching a single high-value prospect before writing a cold email: reading news, checking LinkedIn, scanning filings, looking for a hook that doesn't feel generic. At scale, that's most of the working day — before a single word of outreach gets written.

This tool automates that research and writes the email. The result is a message grounded in something real — a merger, a regulatory filing, a compliance initiative — not a mail-merge template with the company name swapped in.

---

## Autopilot demo path (recommended)

This is the resume / portfolio happy path: **website + ICP → prospects → scored emails**.

1. Install and configure keys:

```bash
git clone https://github.com/sanginimehta/AI-SDR.git
cd AI-SDR
pip install -r requirements.txt
cp .env.example .env
# Required for Autopilot: ANTHROPIC_API_KEY
# Optional but better product summaries: GROQ_API_KEY
```

2. Start the app:

```bash
uvicorn webapp.main:app --reload
```

Open `http://localhost:8000`.

3. On **Autopilot** (default), fill in:

| Field | Example |
|---|---|
| Company name | LogicGate |
| Product name | Risk Cloud |
| Company website | `https://www.logicgate.com` |
| What your product does | Optional if the website is set — Autopilot scrapes the site and learns value props |
| ICP | CISOs / CROs at mid-market banks with active compliance programs |
| Prospect count | Start with **3** |

4. Submit. The results page opens immediately while the job runs in the background:

- Learns your product from the company website
- Discovers matching decision-makers
- Scores each lead (ICP fit / timing / reachability)
- Writes a personalized email for anyone above the threshold
- Lets you download a CRM-ready CSV

If the website cannot be scraped and you left value props empty, the job fails with a clear error instead of inventing a product story.

---

## What it does

**Step 0 — Learn the product (Autopilot)**

When you provide a company website, the app scrapes the homepage and summarizes what you sell. Short or empty value props are filled from that summary. Your ICP text is never overwritten.

**Step 1 — Lead scoring**

Before writing anything, the agent researches each prospect and scores them 1–10 across three dimensions:

- **ICP fit** — Right industry, company profile, and title for what you sell?
- **Timing signals** — Recent urgency triggers (M&A, fine, new hire, breach, IPO)?
- **Reachability** — Real decision-maker or gatekeeper?

Prospects scoring below 6 are skipped. Only qualified leads get emails written.

**Step 2 — Personalized email generation**

For each lead that passes scoring, the agent reads recent news and company context, then writes a cold email that:

- Opens with a specific hook tied to something happening at that company right now
- Connects their challenges to what your product solves (from the website-learned context)
- Closes with a low-pressure ask

**Step 3 — Memory (CLI)**

The CLI agent (`prospect_agent.py`) saves every email per prospect under `prospect_memory/`. On the next cycle it finds a different angle. The web app does not wire memory yet.

---

## Modes

| Mode | How it works | Best for |
|---|---|---|
| **Autopilot** | Website + ICP → learn product → find prospects → score → email | Starting from scratch, resume demos |
| **Bulk** | Upload a CSV or enter rows — scrape prospect sites, generate emails via LLaMA 3.3 (Groq). No scoring. | High-volume top-of-funnel |
| **AI Agent** (API / CLI) | Your prospect list + deep Claude research and scoring | ABM, high-value accounts |

The web UI currently shows Autopilot and Bulk. Agent mode remains available via the CLI and backend.

---

## How to run it

### Web app

```bash
pip install -r requirements.txt
cp .env.example .env   # ANTHROPIC_API_KEY; optional GROQ_API_KEY / CLEARBIT_API_KEY
uvicorn webapp.main:app --reload
```

Open `http://localhost:8000`. Jobs are stored in SQLite (`jobs.db`) so refreshing the results page does not lose a run.

### Command line

```bash
python prospect_agent.py   # agent mode with memory
python generate_emails.py  # bulk pipeline
```

### Health check

```bash
curl http://localhost:8000/health
```

Returns whether each API key is present (booleans only).

---

## Files

| File / Folder | What it does |
|---|---|
| `webapp/` | FastAPI app — form UI, background jobs, results poll, CSV download |
| `core/agent.py` | Claude Agent SDK: discovery, scoring, email research |
| `core/seller.py` | Scrape + summarize the seller company website |
| `core/pipeline.py` | Bulk pipeline: prospect homepage scraping + Groq email generation |
| `core/jobs.py` | SQLite job store |
| `core/intake.py` | Request validation (mode, keys, CSV, caps) |
| `core/reliability.py` | Timeouts, retries, concurrency limits |
| `tests/` | Reliability and seller-site unit tests |
| `agent_output.csv` | Sample agent output |
| `output.csv` | Sample bulk output — Nike and Stripe |

---

## Why two approaches?

| | Bulk pipeline | Autopilot / Agent |
|---|---|---|
| Speed | Seconds per prospect | Minutes per prospect |
| Cost | Low (open-source LLM) | Higher (Claude) |
| Personalization | Good — homepage + enrichment | High — live web research |
| Best for | Top-of-funnel, high volume | High-value accounts, net-new prospecting |

In a real GTM workflow: Autopilot to open a new market, bulk to process the long tail, agent for your top accounts.
