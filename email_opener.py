timport argparse
import os
import sys
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from groq import Groq

load_dotenv()


def scrape_homepage(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    return text[:3000]


def enrich_company(domain: str) -> dict:
    """Fetch live company data from Clearbit. Returns {} on failure or missing key."""
    clearbit_key = os.getenv("CLEARBIT_API_KEY")
    if not clearbit_key:
        return {}
    try:
        resp = requests.get(
            "https://company.clearbit.com/v2/companies/find",
            params={"domain": domain},
            auth=(clearbit_key, ""),
            timeout=8,
        )
        if resp.status_code != 200:
            return {}
        data = resp.json()
        return {
            "description": (data.get("description") or "").strip(),
            "industry": (data.get("category") or {}).get("industry", ""),
            "employees": data.get("metrics", {}).get("employees"),
            "location": ", ".join(filter(None, [
                (data.get("geo") or {}).get("city"),
                (data.get("geo") or {}).get("country"),
            ])),
            "twitter_bio": ((data.get("twitter") or {}).get("bio") or "").strip(),
        }
    except Exception:
        return {}


def generate_opener(company_name: str, homepage_text: str, enrichment: dict) -> str:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    enrichment_block = ""
    if enrichment:
        parts = []
        if enrichment.get("description"):
            parts.append(f"Description: {enrichment['description']}")
        if enrichment.get("industry"):
            parts.append(f"Industry: {enrichment['industry']}")
        if enrichment.get("employees"):
            parts.append(f"Employees: {enrichment['employees']}")
        if enrichment.get("location"):
            parts.append(f"Location: {enrichment['location']}")
        if parts:
            enrichment_block = "\nLive company data:\n" + "\n".join(parts) + "\n"

    prompt = (
        f"Here is information about {company_name}:\n"
        f"{enrichment_block}"
        f"\nHomepage text:\n{homepage_text}\n\n"
        "Based on this, write a single, personalized, conversational sentence "
        "that could open a sponsorship ask email. Reference something specific "
        "about what the company does or values. Do not use generic phrases."
    )
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()


def main():
    parser = argparse.ArgumentParser(description="Generate a personalized sponsorship email opener.")
    parser.add_argument("company", help="Company name")
    parser.add_argument("url", help="Company homepage URL")
    args = parser.parse_args()

    print(f"Scraping homepage: {args.url}")
    try:
        text = scrape_homepage(args.url)
    except Exception as e:
        print(f"Error scraping page: {e}", file=sys.stderr)
        sys.exit(1)

    parsed = urlparse(args.url)
    domain = (parsed.netloc or parsed.path).removeprefix("www.")
    print(f"Enriching company data for {domain}...")
    enrichment = enrich_company(domain)
    if enrichment.get("industry"):
        print(f"  Found: {enrichment['industry']} | {enrichment.get('employees', '?')} employees")
    else:
        print("  No enrichment data (set CLEARBIT_API_KEY to enable).")

    print("Generating email opener...\n")
    opener = generate_opener(args.company, text, enrichment)
    print(f"Email opener for {args.company}:\n\n  {opener}\n")


if __name__ == "__main__":
    main()
