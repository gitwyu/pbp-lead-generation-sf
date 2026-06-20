import argparse
import asyncio
import csv
import logging
import os
from typing import Any, Dict, List

from lib import BaseAgent, extract_csv
from models import AAPIRelevance, CompanyLead, CompanyType
import contextlib


parser = argparse.ArgumentParser(
    description="Run the Discovery Agent to find company leads."
)
parser.add_argument(
    "--nomock",
    action="store_true",
    help="Disable mock mode and run live API calls.",
)
parser.add_argument(
    "--output",
    type=str,
    default="outreach_leads.csv",
    help="Path to the output CSV file.",
)
parser.add_argument(
    "--limit",
    type=int,
    default=10,
    help="Maximum number of companies to discover per query.",
)
parser.add_argument(
    "--model",
    type=str,
    default="gemini-3-flash-preview",
    help="The Gemini model to use.",
)
parser.add_argument(
    "--trace-file",
    type=str,
    default=None,
    help="Path to save the JSON trace of LLM search queries and URLs.",
)
parser.add_argument(
    "--overwrite", action="store_true", help="Overwrite existing output CSV."
)
parser.add_argument(
    "--verbose", action="store_true", help="Enable verbose logging."
)
parser.add_argument(
    "--num_parallel",
    type=int,
    default=1,
    help=(
        "Number of parallel workers. 1 = synchronous execution, >1 = async"
        " execution."
    ),
)

class DiscoveryAgent(BaseAgent):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.prompt_template = self.load_prompt("discover_companies.txt")

  def _parse_csv_to_dicts(self, text: str) -> List[Dict[str, Any]]:
    if not text:
      return []

    csv_text = extract_csv(text)
    data = []
    for line in csv_text.split("\n"):
      line = line.strip()
      if not line:
        continue

      parts = [p.strip() for p in line.split("|")]

      # Check if the LLM accidentally included the header row
      if (
          len(parts) >= 2
          and "title" in parts[0].lower()
          and "link" in parts[1].lower()
      ):
        continue

      if len(parts) >= 7:
        data.append({
            "title": parts[0],
            "link": parts[1],
            "hq_location": parts[2],
            "industry": parts[3],
            "company_type": parts[4],
            "aapi_relevance": parts[5],
            "aapi_notes": parts[6],
        })
      else:
        logging.warning(f"Skipping malformed row: {line}")

    return data


  async def _execute_search_async(self, query: str) -> List[Dict[str, Any]]:
    if self.mock_mode:
      await asyncio.sleep(0.5)
      return self._get_mock_results(query)

    if not self.prompt_template:
      return []

    prompt = self.prompt_template.format(query=query, limit=self.limit)

    async with self.semaphore if self.semaphore else contextlib.nullcontext():
      text, urls = await self._execute_llm_search_async(prompt, query)

    return self._parse_csv_to_dicts(text)

  def _get_mock_results(self, query: str) -> List[Dict[str, Any]]:
    # Dummy data for testing the categorization logic
    if (
        "fortune 500" in query.lower()
        or "national consumer brands" in query.lower()
    ):
      return [
          {
              "title": "JPMorgan Chase & Co.",
              "link": "https://www.jpmorganchase.com",
              "hq_location": "New York, NY",
              "industry": "Finance",
              "company_type": "Corporate Sponsor",
              "aapi_relevance": "Low",
              "aapi_notes": (
                  "General company with no strong AAPI tie, but relevant"
                  " industry."
              ),
          },
          {
              "title": "Pfizer Inc.",
              "link": "https://www.pfizer.com",
              "hq_location": "New York, NY",
              "industry": "Pharmaceuticals",
              "company_type": "Corporate Sponsor",
              "aapi_relevance": "Low",
              "aapi_notes": "Major medical brand open to general sponsorship.",
          },
      ]
    elif "mid-size" in query.lower() or "restaurant groups" in query.lower():
      return [
          {
              "title": "Joe's Coffee Company",
              "link": "https://joescoffeecompany.com",
              "hq_location": "New York, NY",
              "industry": "F&B",
              "company_type": "Restaurant / F&B",
              "aapi_relevance": "Low",
              "aapi_notes": "Local NYC business.",
          },
          {
              "title": "Katz's Delicatessen",
              "link": "https://local.katzsdelicatessen.com",
              "hq_location": "New York, NY",
              "industry": "Restaurant",
              "company_type": "Restaurant / F&B",
              "aapi_relevance": "Low",
              "aapi_notes": "Famous NYC kosher style delicatessen.",
          },
      ]
    elif "aapi" in query.lower() or "asian american" in query.lower():
      return [
          {
              "title": "Omsom",
              "link": "https://omsom.com",
              "hq_location": "New York, NY",
              "industry": "F&B",
              "company_type": "Product Donor (Small Biz)",
              "aapi_relevance": "High",
              "aapi_notes": (
                  "AAPI-founded pantry staples and Asian meal starters."
              ),
          },
          {
              "title": "Glow Recipe",
              "link": "https://www.glowrecipe.com",
              "hq_location": "New York, NY",
              "industry": "Cosmetics",
              "company_type": "Product Donor (Large)",
              "aapi_relevance": "High",
              "aapi_notes": (
                  "AAPI-owned skincare brand featuring fruit-forward cosmetics."
              ),
          },
      ]
    elif (
        "japanese" in query.lower()
        or "korean" in query.lower()
        or "asian international" in query.lower()
    ):
      return [
          {
              "title": "Sony Corporation of America",
              "link": "https://www.sony.com",
              "hq_location": "New York, NY",
              "industry": "Electronics / Media",
              "company_type": "Corporate Sponsor",
              "aapi_relevance": "Medium",
              "aapi_notes": (
                  "Japanese multinational with significant US HQ in NYC."
              ),
          },
          {
              "title": "Nongshim America",
              "link": "https://nongshimusa.com",
              "hq_location": "Rancho Cucamonga, CA",
              "industry": "F&B",
              "company_type": "Product Donor (Large)",
              "aapi_relevance": "Medium",
              "aapi_notes": (
                  "Major Korean food brand with large US distribution."
              ),
          },
      ]
    return []

  def _parse_leads(self, results: List[Dict[str, Any]]) -> List[CompanyLead]:
    leads = []
    for item in results:
      company_name = item.get("title", "").replace(" - Home", "")
      link = item.get("link", "")

      if not link:
        continue

      try:
        leads.append(
            CompanyLead(
                company_name=company_name,
                website=link,
                hq_location=item.get("hq_location", "Unknown"),
                industry=item.get("industry", "Unknown"),
                company_type=CompanyType(
                    item.get("company_type", "Corporate Sponsor")
                ),
                aapi_relevance=AAPIRelevance(item.get("aapi_relevance", "Low")),
                aapi_notes=item.get("aapi_notes", ""),
                source=link,
            )
        )
      except ValueError as e:
        logging.warning(
            f"Skipping lead {company_name} due to invalid enum value: {e}"
        )
    return leads

  def _save_partial(self, leads_so_far: List[CompanyLead], output_file: str | None):
    """Saves the unique leads discovered so far to the CSV."""
    if not output_file or not leads_so_far:
      return

    fieldnames = leads_so_far[0].to_csv_row().keys()
    with open(output_file, mode="w", newline="", encoding="utf-8") as f:
      writer = csv.DictWriter(f, fieldnames=fieldnames)
      writer.writeheader()
      for lead in leads_so_far:
        writer.writerow(lead.to_csv_row())

    logging.info(
        f"Saved partial discovery results ({len(leads_so_far)} leads) to"
        f" {output_file}"
    )

  async def _run_all_searches_async(
      self, queries: List[str], output_file: str | None = None
  ) -> List[CompanyLead]:
    all_leads = []
    seen_websites = set()
    completed_count = 0

    async def worker(query: str):
      return await self._execute_search_async(query)

    tasks = [asyncio.create_task(worker(q)) for q in queries]

    for f in asyncio.as_completed(tasks):
      results = await f
      new_leads = self._parse_leads(results)

      for lead in new_leads:
        if lead.website and lead.website not in seen_websites:
          seen_websites.add(lead.website)
          all_leads.append(lead)

      completed_count += 1
      if completed_count % 5 == 0 or completed_count == len(queries):
        self._save_partial(all_leads, output_file)
        self.export_traces()

    # Allow background SSL transports to close
    await asyncio.sleep(0.25)
    return all_leads

  def discover(
      self, queries: List[str], output_file: str | None = None
  ) -> List[CompanyLead]:
    """Executes a list of search queries and returns a deduplicated list of leads."""
    if self.num_parallel <= 1:
      all_leads = []
      seen_websites = set()
      for i, query in enumerate(queries):
        results = asyncio.run(self._execute_search_async(query))
        new_leads = self._parse_leads(results)
        for lead in new_leads:
          if lead.website and lead.website not in seen_websites:
            seen_websites.add(lead.website)
            all_leads.append(lead)

        if (i + 1) % 5 == 0 or (i + 1) == len(queries):
          self._save_partial(all_leads, output_file)
          self.export_traces()
      return all_leads
    else:
      return asyncio.run(self._run_all_searches_async(queries, output_file))

  def discover_all(
      self, plan: Dict[str, List[str]], output_file: str | None = None
  ) -> List[CompanyLead]:
    """Flattens a discovery plan and executes all queries concurrently."""
    all_queries = []
    for category_name, queries in plan.items():
      all_queries.extend(queries)

    return self.discover(all_queries, output_file)


def export_to_csv(
    leads: List[CompanyLead], filename: str = "outreach_leads.csv"
):
  if not leads:
    logging.info("No leads to export.")
    return

  fieldnames = leads[0].to_csv_row().keys()

  with open(filename, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for lead in leads:
      writer.writerow(lead.to_csv_row())

  logging.info(f"Exported {len(leads)} leads to {filename}")


if __name__ == "__main__":
  args = parser.parse_args()

  logging.basicConfig(
      level=logging.DEBUG if args.verbose else logging.INFO,
      format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
  )

  # Suppress noisy debug logs from underlying HTTP libraries
  if args.verbose:
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

  mock_mode = not args.nomock
  mode_str = "Mock Mode" if mock_mode else "Live Mode"
  logging.info(
      f"Running Discovery Agent in {mode_str} (Limit: {args.limit} per"
      f" query, num_parallel: {args.num_parallel})..."
  )

  if args.trace_file is None:
      base, _ = os.path.splitext(args.output)
      args.trace_file = f"{base}_discover_trace.json"

  if args.trace_file:
    logging.info(
        f"Tracing is ENABLED. Traces will be saved to {args.trace_file}"
    )

    agent = DiscoveryAgent(
        mock_mode=mock_mode,
        limit=args.limit,
        trace_file=args.trace_file,
        model=args.model,
        num_parallel=args.num_parallel,
    )

    # Define all the search categories and their associated queries
    DISCOVERY_PLAN = {
        # General wineries/breweries in Napa/Sonoma/Bay Area - not limited to AAPI.
        # Asian Wine Association is a key connection to tap for warm referrals.
        # Also look for vineyard stays/experiences as auction packages.
        "WINE_SPIRITS": [
            "wineries vineyards Napa Sonoma silent auction donation corporate gifting program",
            "craft breweries distilleries San Francisco Bay Area donation sponsorship program",
            "Asian Wine Association member wineries Bay Area Napa Sonoma",
            "vineyard stay experience packages silent auction Bay Area Napa Sonoma",
        ],
        "FOOD_COFFEE": [
            "specialty coffee roasters gift card donation program San Francisco Bay Area",
            "AAPI owned restaurants food businesses silent auction donation San Francisco",
            "Bay Area restaurant groups catering companies corporate gifting donation",
        ],
        # Warriors, Giants, Valkyries contacts already in hand - skip those.
        # Focus on other SF sports teams, minor leagues, and non-sports experiences.
        # These orgs typically give swag or signed memorabilia (benchwarmer-level).
        "EXPERIENCES_TICKETS": [
            "San Francisco Bay Area minor league sports teams silent auction ticket donation",
            "SF Bay Area concert venues theaters performing arts ticket donation program",
            "Bay Area unique experiences cooking classes pottery rock climbing silent auction donation",
            "San Francisco attractions museums tours gift experiences auction donation",
        ],
        # Added cookbooks and independent bookstores as a specific ask.
        "RETAIL_GIFTCARDS": [
            "AAPI owned retail boutique shops gift card donation San Francisco Bay Area",
            "Bay Area local small businesses silent auction gift basket donation program",
            "Asian American owned beauty wellness spa gift card donation San Francisco",
            "independent bookstores San Francisco Bay Area cookbook donation silent auction",
        ],
        # Added rock climbing gyms and pottery studios as creative auction experiences.
        "SERVICES_WELLNESS": [
            "Bay Area hotels spas wellness experiences silent auction donation program",
            "San Francisco yoga pilates fitness studios gift card donation",
            "rock climbing gyms San Francisco Bay Area gift card donation program",
            "pottery ceramics studios San Francisco Bay Area class donation silent auction",
            "Bay Area cooking classes culinary experiences gift donation silent auction",
        ],
        # Broadened beyond hospitality/culinary to capture more corporate cash sponsors.
        # Includes tech, finance, retail, and healthcare — all active in Bay Area philanthropy.
        # Note: team member to cross-reference with Built In SF list for additional leads.
        "CORPORATE_SPONSORS": [
            "San Francisco Bay Area hospitality culinary industry corporate sponsorship nonprofit",
            "restaurant food beverage companies Bay Area CSR community giving program",
            "hotel groups hospitality companies San Francisco corporate social responsibility sponsorship",
            "tech companies San Francisco Bay Area nonprofit gala sponsorship community giving",
            "financial services companies Bay Area corporate philanthropy nonprofit event sponsorship",
            "retail consumer brands San Francisco Bay Area nonprofit sponsorship donation program",
            "healthcare biotech companies Bay Area AAPI community nonprofit sponsorship",
            "Bay Area Fortune 500 companies corporate foundation nonprofit event cash sponsorship",
        ],
        # Family offices and AAPI executives who may sponsor or make major gifts.
        # Focused on publicly known AAPI founders, VCs, and executives in Bay Area.
        "FAMILY_OFFICES": [
            "San Francisco Bay Area family office philanthropic giving AAPI community",
            "Bay Area private foundation charitable giving Asian American nonprofits",
            "Silicon Valley family office impact investing community sponsorship",
            "AAPI venture capitalists investors San Francisco Bay Area philanthropic giving",
            "Asian American foundation endowment Bay Area charitable donations nonprofits",
        ],
        # AAPI executives and high net worth individuals known for Bay Area philanthropy.
        "AAPI_EXECUTIVES": [
            "AAPI tech executives founders San Francisco Bay Area philanthropic community giving",
            "Asian American C-suite executives Bay Area nonprofit board members donors",
            "Silicon Valley AAPI founders philanthropists charitable giving Bay Area",
            "Asian American venture capitalists Bay Area community impact nonprofit support",
            "notable AAPI business leaders San Francisco Bay Area philanthropic foundations",
        ],
        # Added SF and Napa tourism boards as institutional donors/sponsors.
        "AAPI_COMMUNITY": [
            "AAPI owned businesses San Francisco Bay Area community giving silent auction",
            "Asian American Chamber of Commerce San Francisco Bay Area member businesses",
            "Asian American nonprofit corporate partners sponsors San Francisco Bay Area",
            "San Francisco Travel tourism board nonprofit event sponsorship donation",
            "Visit Napa Valley tourism board community sponsorship nonprofit partnership",
        ],
    }
    if not args.overwrite and os.path.exists(args.output):
      raise FileExistsError(
          f"Output file {args.output} already exists. Use --overwrite to"
          " overwrite."
      )

    logging.info(
        "--- Executing Discovery Plan with"
        f" {sum(len(q) for q in DISCOVERY_PLAN.values())} total queries ---"
    )

    # Discovery loop with periodic saving
    all_discovered_leads = agent.discover_all(
        DISCOVERY_PLAN, output_file=args.output
    )

    for lead in all_discovered_leads:
      logging.info(
          f"{lead.company_name} ({lead.website}) - {lead.company_type.value}"
          f" [AAPI: {lead.aapi_relevance.value}]"
      )

    logging.info(
        f"--- Finished discovery, final CSV saved to ({args.output}) ---"
    )
