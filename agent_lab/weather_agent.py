"""Building AI Agents with APIs — local version of the Husky Coding Project solution.

The model never sees the Python. It sees a tool description, decides when to
call it, and answers from the live Open-Meteo result. A second tool reports
the local time, and the chat at the bottom keeps its message list so it
remembers the conversation.

Colab uses a sidebar secret. Here the same names are environment variables
(or a .env file):

    GEMINI_API_KEY   default provider
    GROQ_API_KEY     used when PROVIDER=groq

    python -m agent_lab.weather_agent --weather Seattle
    python -m agent_lab.weather_agent --ask "Do I need an umbrella in Seattle right now?"
    python -m agent_lab.weather_agent --demo
    python -m agent_lab.weather_agent --chat
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import requests

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Part 0: provider config
# ---------------------------------------------------------------------------

PROVIDER = os.environ.get("PROVIDER", "gemini")

PROVIDERS = {
    "gemini": {
        "secret": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-3.8-flash",
    },
    "groq": {
        "secret": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "model": "openai/gpt-oss-120b",
    },
}


class MissingAPIKey(RuntimeError):
    """Raised when the selected provider has no key in the environment."""


def load_config(provider: str | None = None) -> dict:
    name = (provider or os.environ.get("PROVIDER") or PROVIDER).lower()
    if name not in PROVIDERS:
        known = ", ".join(PROVIDERS)
        raise MissingAPIKey(f"Unknown PROVIDER {name!r}. Use one of: {known}.")
    return {"name": name, **PROVIDERS[name]}


def build_client(provider: str | None = None):
    """OpenAI-compatible client for Gemini or Groq. Needs the matching API key."""
    from openai import OpenAI

    config = load_config(provider)
    api_key = os.environ.get(config["secret"])
    if not api_key:
        raise MissingAPIKey(
            f"No {config['secret']} in the environment. "
            f"Add it to .env or export it. "
            f"Gemini keys come from aistudio.google.com/apikey (personal Gmail, not a UW account). "
            f"If Gemini is blocked, export PROVIDER=groq and GROQ_API_KEY from console.groq.com/keys."
        )
    client = OpenAI(api_key=api_key, base_url=config["base_url"])
    return client, config["model"]


def check_provider(provider: str | None = None) -> str:
    """One tiny call to prove the key works. Returns the model's reply."""
    client, model = build_client(provider)
    reply = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Say hi in five words or fewer."}],
    )
    return reply.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Step 1: a normal Python function
# ---------------------------------------------------------------------------

GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"


def get_weather(city: str) -> dict:
    """City name → coordinates → current temperature, rain, and wind."""
    geo = requests.get(GEO_URL, params={"name": city, "count": 1}, timeout=10).json()
    if not geo.get("results"):
        return {"error": f"Couldn't find a city called {city!r}"}

    place = geo["results"][0]
    weather = requests.get(
        WEATHER_URL,
        params={
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "current": "temperature_2m,rain,wind_speed_10m",
        },
        timeout=10,
    ).json()

    current = weather["current"]
    return {
        "city": place["name"],
        "temp_c": current["temperature_2m"],
        "rain_mm": current["rain"],
        "wind_kmh": current["wind_speed_10m"],
    }


# ---------------------------------------------------------------------------
# Step 2: describe the tool to the model
# ---------------------------------------------------------------------------

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "Get the current weather for a city: temperature in Celsius, "
                "rain in mm, and wind speed in km/h. Use this whenever the user "
                "asks about weather, temperature, rain, or what to wear."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name, e.g. Seattle"}
                },
                "required": ["city"],
            },
        },
    }
]

# Map tool names to real functions, so adding a tool later is one line.
TOOL_FUNCTIONS = {"get_weather": get_weather}


# ---------------------------------------------------------------------------
# Step 3: the loop
# ---------------------------------------------------------------------------

def _parse_args(raw) -> dict:
    """Tool arguments arrive as a JSON string. Some providers already send a dict."""
    if isinstance(raw, str):
        return json.loads(raw) if raw else {}
    return dict(raw or {})


def run_tool_loop(
    messages: list,
    *,
    max_turns: int = 6,
    tool_specs: list | None = None,
    tool_functions: dict | None = None,
    client=None,
    model: str | None = None,
) -> str:
    """Ask the model. If it wants a tool, run it and send the result back.

    Repeat until it answers in plain text, or until max_turns.
    `messages` is updated in place so a caller can keep it for the next question.
    """
    specs = tools if tool_specs is None else tool_specs
    functions = TOOL_FUNCTIONS if tool_functions is None else tool_functions
    if client is None:
        client, default_model = build_client()
        model = model or default_model

    for _ in range(max_turns):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=specs,
        )
        message = response.choices[0].message
        messages.append(message)  # keep the model's turn exactly as it came back

        if not message.tool_calls:
            return message.content or ""

        for call in message.tool_calls:
            args = _parse_args(call.function.arguments)
            print(f"  [tool] {call.function.name}({args})")
            output = functions[call.function.name](**args)
            print(f"  [result] {output}")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,  # must match the request's id
                    "content": json.dumps(output),
                }
            )

    return "Stopped after too many tool calls."


def run_agent(
    question: str,
    max_turns: int = 6,
    *,
    client=None,
    model: str | None = None,
    tool_specs: list | None = None,
    tool_functions: dict | None = None,
) -> str:
    """One question, no memory of earlier questions."""
    messages = [{"role": "user", "content": question}]
    return run_tool_loop(
        messages,
        max_turns=max_turns,
        tool_specs=tool_specs,
        tool_functions=tool_functions,
        client=client,
        model=model,
    )


# ---------------------------------------------------------------------------
# Extra: a second tool, a personality, and a chat that remembers
# ---------------------------------------------------------------------------

def get_time(timezone: str) -> dict:
    """Current local time and weekday in an IANA timezone."""
    try:
        now = datetime.now(ZoneInfo(timezone))
    except Exception:
        return {
            "error": (
                f"Unknown timezone {timezone!r}. "
                "Use a name like America/Los_Angeles."
            )
        }
    return {
        "timezone": timezone,
        "time": now.strftime("%I:%M %p"),
        "day": now.strftime("%A"),
    }


chat_tools = tools + [
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": (
                "Current local time and weekday in an IANA timezone "
                "like Asia/Tokyo or America/New_York."
            ),
            "parameters": {
                "type": "object",
                "properties": {"timezone": {"type": "string"}},
                "required": ["timezone"],
            },
        },
    }
]

CHAT_FUNCTIONS = {"get_weather": get_weather, "get_time": get_time}

SYSTEM_PROMPT = (
    "You are Dubs, the UW husky, helping students plan their day. "
    "Keep answers to two or three sentences. Use your tools instead of guessing."
)


def new_chat() -> list:
    """Fresh conversation. Keep this list and pass it to every turn."""
    return [{"role": "system", "content": SYSTEM_PROMPT}]


def run_chat_turn(
    chat: list,
    question: str,
    max_turns: int = 8,
    *,
    client=None,
    model: str | None = None,
    tool_functions: dict | None = None,
) -> str:
    """Append one user question, run tools as needed, and return Dubs' answer.

    `chat` lives outside this function, so the next question still sees
    the earlier turns.
    """
    chat.append({"role": "user", "content": question})
    return run_tool_loop(
        chat,
        max_turns=max_turns,
        tool_specs=chat_tools,
        tool_functions=tool_functions or CHAT_FUNCTIONS,
        client=client,
        model=model,
    )


def chat_loop(client=None, model: str | None = None) -> None:
    """Interactive chat. Type quit to stop."""
    chat = new_chat()
    print("Dubs is here. Ask about weather or the time. Type quit to stop.")
    while True:
        try:
            question = input("you> ").strip()
        except EOFError:
            print()
            break
        if question.lower() in {"quit", "exit", ""}:
            break
        answer = run_chat_turn(chat, question, client=client, model=model)
        print(f"dubs> {answer}")


# Questions used by the notebook checkpoints and the "two calls" experiment.
DEMO_QUESTIONS = [
    "Do I need an umbrella in Seattle right now?",
    "What's the weather in Seatle?",
    "Is it warmer in Tokyo or Seattle right now?",
]

DEMO_CHAT = [
    "What's the weather in Seattle right now?",
    "What time is it there?",
]


def run_demo(client=None, model: str | None = None) -> None:
    """Checkpoint 1 always hits the live API. The rest need a model key."""
    print("Checkpoint 1: live weather")
    print(get_weather("Seattle"))

    if client is None:
        try:
            client, model = build_client()
        except MissingAPIKey as exc:
            print()
            print(exc)
            print("Checkpoint 1 ran. Set a key and re-run --demo for the model checkpoints.")
            return

    print()
    print("Provider check")
    reply = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Say hi in five words or fewer."}],
    )
    print(f"OK: {load_config()['name']} says: {reply.choices[0].message.content}")

    for question in DEMO_QUESTIONS:
        print()
        print(f"you> {question}")
        print(run_agent(question, client=client, model=model))

    print()
    print("Chat that remembers")
    chat = new_chat()
    for question in DEMO_CHAT:
        print(f"you> {question}")
        print(f"dubs> {run_chat_turn(chat, question, client=client, model=model)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Weather agent that calls tools on its own.")
    parser.add_argument("--weather", metavar="CITY", help="Call get_weather directly. No model needed.")
    parser.add_argument("--time", metavar="TIMEZONE", help="Call get_time directly. No model needed.")
    parser.add_argument("--ask", metavar="QUESTION", help="One question through the agent loop.")
    parser.add_argument("--demo", action="store_true", help="Run the notebook checkpoints.")
    parser.add_argument("--chat", action="store_true", help="Chat with Dubs until you type quit.")
    parser.add_argument("--check", action="store_true", help="Prove the model key with a five-word hello.")
    args = parser.parse_args(argv)

    try:
        if args.weather:
            print(get_weather(args.weather))
            return 0
        if args.time:
            print(get_time(args.time))
            return 0
        if args.check:
            config = load_config()
            print(f"OK: {config['name']} says: {check_provider(config['name'])}")
            return 0
        if args.ask:
            print(run_agent(args.ask))
            return 0
        if args.chat:
            chat_loop()
            return 0
        # --demo, or no flags at all: live weather, then the model checkpoints.
        run_demo()
        return 0
    except MissingAPIKey as exc:
        print(exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
