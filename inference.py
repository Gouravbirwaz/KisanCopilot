"""
KisanAgent Inference Script
=============================
Agent entry point — ReAct loop over the KisanAgent FastMCP server.

The server (server/app.py) runs FastMCP over HTTP (JSON-RPC 2.0).
All env control and tool calls go through the MCP protocol:

  POST http://localhost:7860/mcp   with JSON-RPC bodies

MCP tools available:
  reset_env(difficulty, seed)    — start new season
  observe()                      — read state (no tool budget cost)
  weather(days_ahead)            — weather forecast
  soil()                         — soil moisture
  mandi_price()                  — tomato market price
  govt_scheme()                  — government schemes
  pest_alert()                   — pest surveillance
  credit(amount_inr)             — loan check
  step_env(farm_decision,        — advance one day
           reasoning)

Environment variables (loaded from .env):
  ENV_SERVER_URL   — KisanAgent server (default: http://localhost:7860)
  API_KEY          — LLM API key (LLM_API_KEY also accepted)
  API_BASE_URL     — LLM base URL (default: https://integrate.api.nvidia.com/v1)
  MODEL_NAME       — model name (default: google/gemma-3n-e4b-it)
  DIFFICULTY       — easy | medium | hard (default: medium)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("kisanagent.inference")

# ── Config ────────────────────────────────────────────────────────────────────
ENV_SERVER   = os.getenv("ENV_SERVER_URL", "http://localhost:7860")
MCP_URL      = f"{ENV_SERVER}/mcp"
API_KEY      = os.getenv("API_KEY") or os.getenv("LLM_API_KEY") or "sk-placeholder"
API_BASE_URL = os.getenv("API_BASE_URL", "https://integrate.api.nvidia.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "google/gemma-3n-e4b-it")
DIFFICULTY   = os.getenv("DIFFICULTY", "medium")

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
logger.info("KisanAgent inference  model=%s  server=%s", MODEL_NAME, ENV_SERVER)

# ── Valid values ───────────────────────────────────────────────────────────────
VALID_TOOLS = {"weather", "soil", "mandi_price", "govt_scheme", "pest_alert", "credit"}
VALID_DECISIONS = {
    "irrigate", "fertilize", "spray_pesticide", "sell_now",
    "hold_crop", "apply_scheme", "take_loan", "do_nothing",
}

# ── MCP JSON-RPC client ────────────────────────────────────────────────────────

class MCPClient:
    """
    Stateful MCP HTTP client.
    Handles session init + all tool calls via JSON-RPC 2.0.
    """

    def __init__(self, url: str):
        self.url = url
        self.session_id: Optional[str] = None
        self._req_id = 0
        self._connect()

    def _next_id(self) -> int:
        self._req_id += 1
        return self._req_id

    def _headers(self) -> Dict[str, str]:
        h = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self.session_id:
            h["mcp-session-id"] = self.session_id
        return h

    def _connect(self):
        """Perform MCP initialize + notifications/initialized handshake."""
        resp = requests.post(
            self.url,
            headers=self._headers(),
            json={
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "kisanagent-inference", "version": "1.0.0"},
                },
            },
            timeout=30,
        )
        resp.raise_for_status()
        self.session_id = resp.headers.get("mcp-session-id")

        # Required handshake notification (no id = notification, not a request)
        requests.post(
            self.url,
            headers=self._headers(),
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
            timeout=10,
        )
        logger.info("MCP session established: %s", self.session_id)

    def call(self, tool_name: str, arguments: Dict[str, Any] = None, retries: int = 3) -> Dict[str, Any]:
        """
        Call an MCP tool by name with given arguments.
        Returns the parsed result dict from the tool response.
        Retries on network errors with exponential backoff.
        """
        arguments = arguments or {}
        payload = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        }

        for attempt in range(retries):
            try:
                resp = requests.post(
                    self.url,
                    headers=self._headers(),
                    json=payload,
                    timeout=30,
                )
                resp.raise_for_status()
                return self._parse_mcp_response(resp.text, tool_name)
            except Exception as exc:
                if attempt < retries - 1:
                    wait = 2 ** attempt
                    logger.warning(
                        "MCP call %s attempt %d failed: %s — retry in %ds",
                        tool_name, attempt + 1, exc, wait,
                    )
                    time.sleep(wait)
                else:
                    logger.error("MCP call %s exhausted retries: %s", tool_name, exc)
                    return {"error": str(exc)}

    @staticmethod
    def _parse_mcp_response(raw_text: str, tool_name: str) -> Dict[str, Any]:
        """
        Parse MCP SSE or plain JSON response.
        FastMCP returns: event: message\\ndata: {...}
        """
        text = raw_text.strip()

        # Strip SSE envelope if present
        if text.startswith("event:"):
            for line in text.splitlines():
                if line.startswith("data:"):
                    text = line[5:].strip()
                    break

        try:
            envelope = json.loads(text)
        except json.JSONDecodeError:
            logger.error("Unparseable MCP response for %s: %s", tool_name, raw_text[:200])
            return {"error": "unparseable_response"}

        if "error" in envelope:
            err = envelope["error"]
            logger.error("MCP error for %s: %s", tool_name, err)
            return {"error": err.get("message", "mcp_error"), "code": err.get("code")}

        result = envelope.get("result", {})

        # Prefer structuredContent (richer), fall back to content[0].text
        if "structuredContent" in result:
            return result["structuredContent"]

        content = result.get("content", [])
        if content and content[0].get("type") == "text":
            try:
                return json.loads(content[0]["text"])
            except (json.JSONDecodeError, KeyError):
                return {"raw": content[0].get("text", "")}

        return result


# Module-level MCP client (initialised once, reused for the whole episode)
_mcp: Optional[MCPClient] = None


def get_mcp() -> MCPClient:
    global _mcp
    if _mcp is None:
        _mcp = MCPClient(MCP_URL)
    return _mcp


# ── Env helpers ────────────────────────────────────────────────────────────────

def reset_env(difficulty: str = "medium", seed: int = 0) -> Dict:
    return get_mcp().call("reset_env", {"difficulty": difficulty, "seed": seed or 0})


def observe() -> Dict:
    return get_mcp().call("observe")


def call_tool(tool_name: str, args: Dict = None) -> Dict:
    return get_mcp().call(tool_name, args or {})


def step_env(farm_decision: str, reasoning: str = "") -> Dict:
    return get_mcp().call("step_env", {"farm_decision": farm_decision, "reasoning": reasoning})


# ── LLM helpers ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are KisanAgent, an AI agricultural advisor for Harish —
a smallholder tomato farmer with 2 acres in Kolar district, Karnataka, India.

Goal: maximize Harish's net income across a 90-day tomato growing season.
Starting capital: ₹15,000. Good season: ₹25,000–₹40,000 net income.

════════════════════════════════════════
AVAILABLE TOOLS  (call via tool_to_call)
════════════════════════════════════════
weather      — IMD Karnataka 3-day forecast (noisy, can be unavailable)
soil         — IoT soil moisture, pH, nitrogen readings (±5% sensor noise)
mandi_price  — Today's tomato price at KR Puram Bangalore (₹12–35/kg)
govt_scheme  — Karnataka Raitha Seva Kendra — active subsidies & deadlines
pest_alert   — Dept of Agriculture pest surveillance — Kolar region
credit       — KCC / NABARD microfinance — loan eligibility check

════════════════════════════════════════
FARM DECISIONS  (exactly one per day)
════════════════════════════════════════
irrigate         — ₹200, adds 20% soil moisture
fertilize        — ₹600, boosts yield in vegetative/fruiting stage
spray_pesticide  — ₹800, treats pests (WARNING: penalised if flowering + no pest)
sell_now         — sell yield at today's mandi price (harvest stage only)
hold_crop        — wait for better price (harvest stage)
apply_scheme     — claim active government scheme benefit
take_loan        — borrow KCC loan up to ₹25,000 at 7% p.a.
do_nothing       — no action

════════════════════════════════════════
DECISION RULES
════════════════════════════════════════
1. Call at least one relevant tool before deciding (budget: 3/day).
2. NEVER spray_pesticide in flowering stage (days 41–60) without checking pest_alert first.
3. ALWAYS check mandi_price before sell_now or hold_crop.
4. Check govt_scheme every 10 days — missing deadlines loses ₹1,500–₹5,000.
5. Soil moisture below 40% during fruiting (days 61–80) = permanent yield loss.
6. Pest outbreaks escalate LOW→MEDIUM→HIGH→CRITICAL over 6 days. Treat within 4 days.

════════════════════════════════════════
RESPONSE FORMAT  (strict JSON, no prose, no markdown fences)
════════════════════════════════════════
{
  "reasoning": "2-3 sentences: what you observed and why",
  "tool_to_call": "tool_name OR null",
  "tool_args": {},
  "farm_decision": "decision_name OR null"
}

Exactly one of tool_to_call or farm_decision must be non-null.
"""


def _strip_fences(text: str) -> str:
    t = text.strip()
    for prefix in ("```json", "```"):
        if t.startswith(prefix):
            t = t[len(prefix):]
            break
    if t.endswith("```"):
        t = t[:-3]
    return t.strip()


def llm_call(messages: List[Dict], retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.2,
                top_p=0.7,
                max_tokens=512,
            )
            raw = resp.choices[0].message.content
            clean = _strip_fences(raw)
            json.loads(clean)   # validate — raises if not JSON
            return clean
        except json.JSONDecodeError:
            logger.warning("LLM non-JSON on attempt %d — injecting reminder.", attempt + 1)
            if attempt < retries - 1:
                msgs = list(messages)
                if msgs and msgs[-1]["role"] == "user":
                    msgs[-1] = {
                        "role": "user",
                        "content": msgs[-1]["content"]
                        + "\n\nIMPORTANT: reply with valid JSON only — no prose, no markdown.",
                    }
                messages = msgs
                time.sleep(1)
            else:
                return json.dumps({
                    "reasoning": "JSON parse error — safe fallback.",
                    "tool_to_call": None,
                    "tool_args": {},
                    "farm_decision": "do_nothing",
                })
        except Exception as exc:
            if attempt < retries - 1:
                wait = 2 ** attempt
                logger.warning("LLM attempt %d failed: %s — retry in %ds", attempt + 1, exc, wait)
                time.sleep(wait)
            else:
                raise RuntimeError(f"LLM failed after {retries} attempts: {exc}") from exc


def safe_parse(raw: str) -> Dict:
    try:
        return json.loads(_strip_fences(raw))
    except json.JSONDecodeError:
        logger.error("Unparseable LLM output: %s", raw[:200])
        return {"reasoning": "parse error", "tool_to_call": None, "tool_args": {}, "farm_decision": "do_nothing"}


# ── Main episode loop ─────────────────────────────────────────────────────────

def run_episode(difficulty: str = "medium", seed: Optional[int] = None, verbose: bool = True) -> Dict:
    if verbose:
        print(f"\n{'🌾 ' * 20}")
        print(f"🌾  KisanAgent — Season Start")
        print(f"    Difficulty: {difficulty.upper()} | Model: {MODEL_NAME}")
        print(f"{'🌾 ' * 20}\n")

    # ── Reset via MCP ─────────────────────────────────────────────
    reset_resp = reset_env(difficulty=difficulty, seed=seed or 0)

    if "error" in reset_resp:
        raise RuntimeError(f"reset_env failed: {reset_resp}")

    season_id = reset_resp.get("season_id", "N/A")

    if verbose:
        print(f"Season ID  : {season_id}")
        print(f"Difficulty : {reset_resp.get('difficulty', difficulty)}")
        print(f"Seed       : {reset_resp.get('seed', '?')}")
        print(f"Optimal    : ₹{reset_resp.get('optimal_income_inr', 40000):,.0f}")
        print(f"Schemes    : {', '.join(reset_resp.get('schemes_available', []))}")
        print(f"Pest events: {reset_resp.get('pest_events_scheduled', 0)}\n")

    # Initial observation
    observation = observe()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    episode_rewards: List[float] = []

    # ── 90-day loop ───────────────────────────────────────────────
    for _iter in range(90):
        day         = observation.get("day", 0)
        stage       = observation.get("crop_stage", "unknown")
        balance     = observation.get("bank_balance_inr", 0.0)
        yield_kg    = observation.get("estimated_yield_kg", 0.0)
        moisture    = observation.get("soil_moisture_pct", 0.0)
        weather_sum = observation.get("weather_summary", "dry")
        tools_left  = observation.get("tool_calls_remaining", 3)
        days_left   = observation.get("days_to_harvest", 90 - day)
        alerts      = observation.get("active_alerts", [])
        season_on   = observation.get("season_active", True)

        if not season_on:
            logger.info("Season inactive at day %d — exiting.", day)
            break

        if verbose:
            print(
                f"📅 Day {day:>2} | {stage:<12} | "
                f"₹{balance:>8,.0f} | {yield_kg:>6.0f}kg | 💧{moisture:.0f}%"
                + (f" | ⚠ {alerts[0]}" if alerts else "")
            )

        user_msg = (
            f"CURRENT STATE — Day {day}\n"
            f"Crop stage       : {stage}\n"
            f"Estimated yield  : {yield_kg:.0f} kg\n"
            f"Bank balance     : ₹{balance:,.0f}\n"
            f"Days to harvest  : {days_left}\n"
            f"Soil moisture    : {moisture:.1f}%\n"
            f"Weather summary  : {weather_sum}\n"
            f"Tool calls left  : {tools_left}/3\n"
            f"Active alerts    : {alerts if alerts else 'None'}\n\n"
            "Call a tool (tool_to_call) OR make your farm_decision."
        )
        messages.append({"role": "user", "content": user_msg})

        # ── ReAct inner loop ──────────────────────────────────────
        tool_calls_made: List[str] = []
        farm_decision: Optional[str] = None
        day_reasoning = ""

        while farm_decision is None:
            raw = llm_call(messages)
            parsed = safe_parse(raw)

            day_reasoning = parsed.get("reasoning", "")
            tool_to_call  = parsed.get("tool_to_call") or None
            tool_args     = parsed.get("tool_args") or {}
            farm_decision = parsed.get("farm_decision") or None

            if isinstance(tool_to_call, str) and tool_to_call.lower() in ("null", "none", ""):
                tool_to_call = None
            if isinstance(farm_decision, str) and farm_decision.lower() in ("null", "none", ""):
                farm_decision = None

            if tool_to_call in VALID_DECISIONS and not farm_decision:
                logger.warning("LLM put decision '%s' in tool_to_call — swapping.", tool_to_call)
                farm_decision = tool_to_call
                tool_to_call  = None

            if tool_to_call and tool_to_call not in VALID_TOOLS:
                logger.warning("Unknown tool '%s' — ignoring.", tool_to_call)
                tool_to_call = None

            if farm_decision and farm_decision not in VALID_DECISIONS:
                logger.warning("Invalid decision '%s' — defaulting to do_nothing.", farm_decision)
                farm_decision = "do_nothing"

            messages.append({"role": "assistant", "content": raw})

            if tool_to_call and not farm_decision:
                tool_result = call_tool(tool_to_call, tool_args)

                if tool_result.get("error") == "tool_budget_exceeded":
                    messages.append({
                        "role": "user",
                        "content": (
                            "⚠️ Tool budget exhausted (3/3 used today). "
                            "You MUST choose a farm_decision now. "
                            "Do NOT set tool_to_call."
                        ),
                    })
                else:
                    tool_calls_made.append(tool_to_call)
                    messages.append({
                        "role": "user",
                        "content": (
                            f"Tool result ({tool_to_call}):\n"
                            f"{json.dumps(tool_result, indent=2, ensure_ascii=False)}\n\n"
                            "Call another tool OR set farm_decision to proceed."
                        ),
                    })
                    logger.info("  Tool %-14s called (%d/3)", tool_to_call, len(tool_calls_made))

            elif farm_decision and tool_to_call:
                logger.debug("Both set — using decision, ignoring tool.")
                tool_to_call = None

            elif not farm_decision and not tool_to_call:
                logger.warning("LLM gave neither — defaulting to do_nothing.")
                farm_decision = "do_nothing"

        # ── Step via MCP ──────────────────────────────────────────
        try:
            step_resp = step_env(farm_decision=farm_decision, reasoning=day_reasoning)
        except Exception as exc:
            logger.error("step_env failed on day %d: %s", day, exc)
            break

        reward = float(step_resp.get("reward", 0.0))
        done   = bool(step_resp.get("done", False))
        episode_rewards.append(reward)

        # Refresh observation for next iteration
        obs_from_step = step_resp.get("observation", {})
        observation = obs_from_step if obs_from_step else observe()

        if verbose:
            cost = step_resp.get("cost_incurred_inr", 0)
            print(
                f"   ↳ {farm_decision:<18} | tools: {tool_calls_made or ['—']} | "
                f"cost: ₹{cost:>5,.0f} | reward: {reward:+.3f}"
            )

        if done:
            net_income   = step_resp.get("net_income_inr") or 0.0
            final_scores = step_resp.get("final_scores") or {}

            if verbose:
                print(f"\n{'═' * 56}")
                print(f"🏁  SEASON COMPLETE — Day 90")
                print(f"{'═' * 56}")
                print(f"💰  Net Income      : ₹{net_income:>10,.0f}")
                print(f"📊  Composite Score : {final_scores.get('composite_score', 0):.4f}")
                print(f"    Income Score    : {final_scores.get('income_score', 0):.4f}")
                print(f"    Tool Quality    : {final_scores.get('tool_use_quality', 0):.4f}")
                print(f"    Pest Response   : {final_scores.get('pest_response_accuracy', 0):.4f}")
                print(f"    Scheme Capture  : {final_scores.get('scheme_capture_rate', 0):.4f}")
                print(f"    Sustainability  : {final_scores.get('sustainability_score', 0):.4f}")
                print(f"{'═' * 56}\n")

            return {
                "season_id"     : season_id,
                "net_income_inr": net_income,
                "final_scores"  : final_scores,
                "total_reward"  : round(sum(episode_rewards), 4),
                "episode_length": day + 1,
            }

    return {"error": "Episode did not terminate cleanly within 90 steps."}


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    result = run_episode(difficulty=DIFFICULTY)
    print(f"\nFinal result:\n{json.dumps(result, indent=2, ensure_ascii=False)}")