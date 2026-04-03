import argparse
import asyncio
import csv
import json
import logging
import os
from typing import Dict, List

from lib import BaseAgent
from models import EnrichedLeadData

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(
    description="Run the Enrichment Agent to find contact info."
)
parser.add_argument(
    "--input",
    type=str,
    required=True,
    help="Path to the input CSV file from Discovery phase.",
)
parser.add_argument(
    "--output",
    type=str,
    default="enriched_leads.csv",
    help="Path to save the enriched CSV.",
)
group = parser.add_mutually_exclusive_group()
group.add_argument(
    "--overwrite",
    action="store_true",
    help="Overwrite existing output files.",
)
group.add_argument(
    "--resume",
    action="store_true",
    help="Resume from previous state if output file exists.",
)
parser.add_argument(
    "--nomock",
    action="store_true",
    help="Disable mock mode and run live API calls.",
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
    help="Path to save the JSON trace (defaults to output name + _enrich_trace.json).",
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


class EnrichmentAgent(BaseAgent):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.prompt_template = self.load_prompt("enrich_contact.txt")
    self.system_instruction = self.load_prompt("enrich_contact_system.txt")

  async def _enrich_lead_async(
      self, row: Dict[str, str]
  ) -> List[Dict[str, str]]:
    """Asynchronously processes a single lead row to find contact info."""
    company_name = row.get("Company Name", "")
    website = row.get("Website", "")
    industry = row.get("Industry / Category", "")

    if not company_name or not website:
      return [row]

    query = (
        f"Find CSR, PR, or general contact info for {company_name} ({website})"
    )

    if not self.prompt_template:
      return [row]

    prompt = self.prompt_template.format(
        company_name=company_name, website=website, industry=industry
    )

    import contextlib

    async with self.semaphore if self.semaphore else contextlib.nullcontext():
      logger.debug(f"Enriching {company_name}...")

      if self.mock_mode:
        await asyncio.sleep(0.5)
        row["Contact 1 Name"] = "Jane Doe (Mock)"
        row["Contact 1 Title"] = "Manager of CSR"
        row["Contact 1 Email"] = f"jane.doe@{website.replace('https://www.', '').replace('http://', '').replace('/', '')}"
        row["Contact 1 LinkedIn"] = "https://linkedin.com/in/janedoemock"
        row["Contact 2 Name"] = ""
        row["Contact 2 Title"] = ""
        row["Contact 2 Email"] = ""
        row["Contact 2 LinkedIn"] = ""
        row["General Email"] = f"info@{website.replace('https://www.', '').replace('http://', '').replace('/', '')}"
        row["General Contact Webpage"] = f"{website}/contact"
        row["Source"] = website
        row["Date Sources Updated"] = "2024-01-01"
        row["Notes"] = "Mocked data. Confidence: high."
        return [row]

      # 1. Do the Search Grounding call to find the best contact details and URLs
      result_data, urls = await self._execute_llm_search_async(
          prompt=prompt,
          original_query=query,
          response_schema=EnrichedLeadData,
          use_search=True,
          system_instruction=self.system_instruction,
      )

      source_urls = ", ".join([u["uri"] for u in urls])
      row["Source"] = source_urls if source_urls else website

      if result_data:
        for c in [1, 2]:
          contact = getattr(result_data, f"contact_{c}")
          row[f"Contact {c} Name"] = contact.name
          row[f"Contact {c} Title"] = contact.title
          row[f"Contact {c} Email"] = contact.email
          row[f"Contact {c} LinkedIn"] = contact.linkedin
        row["General Email"] = result_data.general_email
        row["General Contact Webpage"] = result_data.general_contact_webpage
        row["Date Sources Updated"] = getattr(result_data, "date_sources_updated", "Unknown")
        row["Notes"] = result_data.notes
      else:
        for c in [1, 2]:
          row[f"Contact {c} Name"] = ""
          row[f"Contact {c} Title"] = ""
          row[f"Contact {c} Email"] = ""
          row[f"Contact {c} LinkedIn"] = ""
        row["General Email"] = ""
        row["General Contact Webpage"] = ""
        row["Date Sources Updated"] = "Unknown"
        row["Notes"] = ""
        logger.warning(f"No result data found for {company_name}.")

      return [row]

  def _save_partial(
      self,
      original_leads: List[Dict[str, str]],
      completed_results: Dict[int, List[Dict[str, str]]],
      output_file: str | None,
  ):
    """Saves the current state of leads (processed + pending) to the CSV."""
    if not output_file:
      return

    output_rows = []
    for i, original_row in enumerate(original_leads):
      if i in completed_results:
        output_rows.extend(completed_results[i])
      else:
        unprocessed = original_row.copy()
        for c in [1, 2]:
          unprocessed[f"Contact {c} Name"] = ""
          unprocessed[f"Contact {c} Title"] = ""
          unprocessed[f"Contact {c} Email"] = ""
          unprocessed[f"Contact {c} LinkedIn"] = ""
        unprocessed["General Email"] = ""
        unprocessed["General Contact Webpage"] = ""
        unprocessed["Date Sources Updated"] = ""
        unprocessed["Notes"] = ""
        output_rows.append(unprocessed)

    if output_rows:
      fieldnames = output_rows[0].keys()
      with open(output_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
      logger.info(
          "Saved partial results"
          f" ({len(completed_results)}/{len(original_leads)} completed) to"
          f" {output_file}"
      )

  async def _enrich_all_async(
      self, leads: List[Dict[str, str]], output_file: str | None = None, existing_results: Dict[int, List[Dict[str, str]]] | None = None
  ) -> List[Dict[str, str]]:
    completed_results: Dict[int, List[Dict[str, str]]] = existing_results or {}

    async def worker(index: int, row: Dict[str, str]):
      res = await self._enrich_lead_async(row)
      return index, res

    tasks = [
        asyncio.create_task(worker(i, row))
        for i, row in enumerate(leads)
        if i not in completed_results
    ]

    completed_count = len(completed_results)
    for f in asyncio.as_completed(tasks):
      i, res = await f
      completed_results[i] = res
      completed_count += 1

      if completed_count % 5 == 0 or completed_count == len(leads):
        self._save_partial(leads, completed_results, output_file)
        self.export_traces()

    flattened_results = []
    for i in range(len(leads)):
      flattened_results.extend(completed_results.get(i, []))

    # Allow background background SSL transports to close
    await asyncio.sleep(0.25)
    return flattened_results

  def enrich_all(
      self, leads: List[Dict[str, str]], output_file: str | None = None, existing_results: Dict[int, List[Dict[str, str]]] | None = None
  ) -> List[Dict[str, str]]:
    """Processes a list of leads based on num_parallel configuration."""
    completed_results = existing_results or {}

    if self.num_parallel <= 1:
      logger.info("Running enrichment sequentially.")

      for i, row in enumerate(leads):
        if i in completed_results:
          continue

        completed_results[i] = asyncio.run(self._enrich_lead_async(row))

        if (len(completed_results)) % 5 == 0 or len(completed_results) == len(leads):
          self._save_partial(leads, completed_results, output_file)
          self.export_traces()

      flattened = []
      for i in range(len(leads)):
        flattened.extend(completed_results.get(i, []))
      return flattened
    else:
      logger.info(
          f"Running enrichment concurrently with {self.num_parallel} workers."
      )
      return asyncio.run(self._enrich_all_async(leads, output_file, completed_results))


def main():
  args = parser.parse_args()

  logging.basicConfig(
      level=logging.DEBUG if args.verbose else logging.INFO,
      format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
  )

  if args.verbose:
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

  if args.trace_file is None:
      base, _ = os.path.splitext(args.output)
      args.trace_file = f"{base}_enrich_trace.json"

  existing_results = {}
  if os.path.exists(args.output):
      if args.overwrite:
          logger.info(f"Overwriting existing output file: {args.output}")
      elif args.resume:
          logger.info(f"Resuming from existing output file: {args.output}")
          try:
              with open(args.output, "r", encoding="utf-8") as f:
                  reader = list(csv.DictReader(f))
                  for i, row in enumerate(reader):
                      if row.get("Date Sources Updated", "") != "":
                          existing_results[i] = [row]
              logger.info(f"Found {len(existing_results)} previously enriched leads to skip.")
          except Exception as e:
              logger.error(f"Failed to read existing output file: {e}")
              return
      else:
          logger.error(f"Output file {args.output} already exists. Use --overwrite to overwrite or --resume to resume.")
          return

  existing_traces = []
  if args.resume and os.path.exists(args.trace_file):
      try:
          with open(args.trace_file, "r", encoding="utf-8") as f:
              existing_traces = json.load(f)
      except Exception as e:
          logger.warning(f"Failed to load existing trace file {args.trace_file}: {e}")

  mock_mode = not args.nomock
  mode_str = "Mock Mode" if mock_mode else "Live Mode"
  logger.info(
      f"Running Enrichment Agent in {mode_str} (num_parallel:"
      f" {args.num_parallel})..."
  )

  if not os.path.exists(args.input):
    logger.error(f"Input file {args.input} does not exist.")
    return

  # Read input CSV
  leads = []
  with open(args.input, mode="r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
      leads.append(row)

  if not leads:
    logger.error("Input CSV is empty.")
    return

  logger.info(f"Loaded {len(leads)} leads for enrichment.")

  agent = EnrichmentAgent(
      mock_mode=mock_mode,
      trace_file=args.trace_file,
      model=args.model,
      num_parallel=args.num_parallel,
  )
  if existing_traces:
      agent.traces.extend(existing_traces)

  # Run enrichment loop with periodic saving
  enriched_leads = agent.enrich_all(leads, output_file=args.output, existing_results=existing_results)
  logger.info("Enrichment complete.")

  # Suppress noisy unclosed socket warnings on exit
  import sys
  if sys.platform.startswith("linux") or sys.platform.startswith("darwin"):
      import warnings
      warnings.filterwarnings("ignore", category=ResourceWarning, message="unclosed.*<socket.socket.*>")
  try:
      # Clear the client to trigger GC before loop state gets weird
      if hasattr(agent, "client"):
          del agent.client
  except AttributeError:
      pass


if __name__ == "__main__":
  main()
