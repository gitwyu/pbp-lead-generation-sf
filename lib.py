import json
import logging
import os
import re
from typing import Any, Dict, List, Optional
from google import genai
from google.genai import types
import httpx
import asyncio

logger = logging.getLogger(__name__)


def extract_csv(text: str) -> str:
  """Safely extracts a CSV block using regex if there is conversational filler."""
  match = re.search(r"```(?:csv|text)?\s*(.*?)\s*```", text, re.DOTALL)
  return match.group(1) if match else text.strip()




class BaseAgent:

  def __init__(
      self,
      mock_mode: bool = False,
      limit: int = 10,
      trace_file: str = "",
      model: str = "gemini-3-flash-preview",
      num_parallel: int = 1,
  ):
    self.mock_mode = mock_mode
    self.limit = limit
    self.trace_file = trace_file
    self.model = model
    self.num_parallel = num_parallel
    self._semaphore = None
    self.traces = []
    self.api_key = os.environ.get("GEMINI_API_KEY")

    if not self.mock_mode and not self.api_key:
      raise ValueError("GEMINI_API_KEY must be set when not in mock_mode")

    if not self.mock_mode:
      self.client = genai.Client(api_key=self.api_key)

  def load_prompt(self, filename: str) -> str:
    """Loads a prompt template from the prompts/ directory."""
    prompt_path = os.path.join(os.path.dirname(__file__), "prompts", filename)
    try:
      with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()
    except FileNotFoundError:
      logger.error(f"Prompt file not found at {prompt_path}")
      return ""

  def _build_config_kwargs(
      self, response_schema: Any = None, use_search: bool = True, system_instruction: str = ''
  ) -> Dict[str, Any]:
    config_kwargs: Dict[str, Any] = {
        "temperature": 0.2,
    }
    if use_search:
      config_kwargs["tools"] = [types.Tool(google_search=types.GoogleSearch())]

    if response_schema:
      config_kwargs["response_mime_type"] = "application/json"
      config_kwargs["response_schema"] = response_schema

    if system_instruction:
      config_kwargs["system_instruction"] = system_instruction

    return config_kwargs

  async def _resolve_redirect(self, url: str, client: httpx.AsyncClient) -> str:
    """Resolves the Vertex AI Grounding obfuscated redirect URLs."""
    if not url.startswith(
        "https://vertexaisearch.cloud.google.com/grounding-api-redirect/"
    ):
      return url
    try:
      # Follow redirects=False lets us capture the Location header immediately
      response = await client.get(url, follow_redirects=False, timeout=5.0)
      if response.status_code in (301, 302, 303, 307, 308):
        return response.headers.get("Location", url)
      return url
    except httpx.RequestError as e:
      logger.debug(f"Failed to resolve URL {url}: {e}")
      return url

  async def _process_llm_response(
      self, response: Any, original_query: str, response_schema: Any = None
  ) -> tuple[Any, List[Dict[str, str]]]:
    urls_retrieved = []
    if response.candidates:
      metadata = response.candidates[0].grounding_metadata
      if metadata:
        if metadata.web_search_queries:
          logger.debug(f"LLM searched for: {list(metadata.web_search_queries)}")

        if metadata.grounding_chunks:
          # Resolve all redirect URLs concurrently
          async with httpx.AsyncClient() as http_client:
            resolve_tasks = [
                self._resolve_redirect(chunk.web.uri, http_client)
                if chunk.web
                else self._resolve_redirect("", http_client)
                for chunk in metadata.grounding_chunks
            ]
            resolved_uris = await asyncio.gather(*resolve_tasks)

          urls_retrieved = [
              {"title": chunk.web.title, "uri": resolved_uri}
              for chunk, resolved_uri in zip(
                  metadata.grounding_chunks, resolved_uris
              )
              if chunk.web and resolved_uri
          ]

        if self.trace_file:
          candidate = response.candidates[0]
          trace_entry = {
              "original_prompt_query": original_query,
              "search_queries_executed_by_llm": (
                  list(metadata.web_search_queries)
                  if metadata.web_search_queries
                  else []
              ),
              "urls_retrieved_as_context": urls_retrieved,
              "model_version": getattr(response, "model_version", "Unknown"),
              "response_id": getattr(response, "response_id", "Unknown"),
              "finish_reason": (
                  candidate.finish_reason.name
                  if getattr(candidate, "finish_reason", None)
                  else "Unknown"
              ),
          }

          if getattr(candidate, "safety_ratings", None):
            trace_entry["safety_ratings"] = [
                {
                    "category": (
                        r.category.name
                        if hasattr(r.category, "name")
                        else str(r.category)
                    ),
                    "probability": (
                        r.probability.name
                        if hasattr(r.probability, "name")
                        else str(r.probability)
                    ),
                }
                for r in candidate.safety_ratings
            ]

          if hasattr(response, "usage_metadata") and response.usage_metadata:
            trace_entry["usage_metadata"] = {
                "prompt_tokens": response.usage_metadata.prompt_token_count,
                "response_tokens": (
                    response.usage_metadata.candidates_token_count
                ),
                "total_tokens": response.usage_metadata.total_token_count,
            }
          self.traces.append(trace_entry)

    elif self.trace_file:
      # Also log traces for calls that don't use search (like the deep scrape)
      candidate = response.candidates[0] if response.candidates else None
      trace_entry = {
          "original_prompt_query": original_query,
          "search_queries_executed_by_llm": [],
          "urls_retrieved_as_context": [],
          "model_version": getattr(response, "model_version", "Unknown"),
          "response_id": getattr(response, "response_id", "Unknown"),
      }

      if candidate:
        trace_entry["finish_reason"] = (
            candidate.finish_reason.name
            if getattr(candidate, "finish_reason", None)
            else "Unknown"
        )
        if getattr(candidate, "safety_ratings", None):
          trace_entry["safety_ratings"] = [
              {
                  "category": (
                      r.category.name
                      if hasattr(r.category, "name")
                      else str(r.category)
                  ),
                  "probability": (
                      r.probability.name
                      if hasattr(r.probability, "name")
                      else str(r.probability)
                  ),
              }
              for r in candidate.safety_ratings
          ]

      if hasattr(response, "usage_metadata") and response.usage_metadata:
        trace_entry["usage_metadata"] = {
            "prompt_tokens": response.usage_metadata.prompt_token_count,
            "response_tokens": response.usage_metadata.candidates_token_count,
            "total_tokens": response.usage_metadata.total_token_count,
        }
      self.traces.append(trace_entry)

    if (
        response_schema
        and hasattr(response, "parsed")
        and response.parsed is not None
    ):
      return response.parsed, urls_retrieved
    return response.text, urls_retrieved

  @property
  def semaphore(self) -> Optional[asyncio.Semaphore]:
    """Lazily creates the semaphore within the event loop."""
    if self.num_parallel > 1 and self._semaphore is None:
      self._semaphore = asyncio.Semaphore(self.num_parallel)
    return self._semaphore

  async def _execute_llm_search_async(
      self,
      prompt: str,
      original_query: str,
      response_schema: Any = None,
      use_search: bool = True,
      system_instruction: str = '',
  ) -> tuple[Any, List[Dict[str, str]]]:
    """Executes an async LLM call (optionally with Google Search) and handles tracing."""
    logger.debug(
        f"Executing async LLM search for query: '{original_query}',"
        f" use_search={use_search}"
    )

    try:
      response = await self.client.aio.models.generate_content(
          model=self.model,
          contents=prompt,
          config=types.GenerateContentConfig(
              **self._build_config_kwargs(response_schema, use_search, system_instruction)
          ),
      )
      logger.debug("Received async LLM response.")
      return await self._process_llm_response(
          response, original_query, response_schema
      )
    except Exception as e:
      logger.error(
          "Error executing async Gemini search for query"
          f" '{original_query}': {e}"
      )
      return None, []

  def export_traces(self):
    if not self.trace_file or not self.traces:
      return
    with open(self.trace_file, "w", encoding="utf-8") as f:
      json.dump(self.traces, f, indent=2)
    logger.info(f"Saved search grounding traces to {self.trace_file}")
