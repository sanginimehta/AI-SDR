"""Weather tool, clock tool, and the decide-then-call loop.

Live tests hit Open-Meteo. Loop tests use a scripted stand-in for the model
so they don't need a Gemini or Groq key.
"""

from unittest.mock import patch

from agent_lab.weather_agent import (
    chat_tools,
    get_time,
    get_weather,
    new_chat,
    run_agent,
    run_chat_turn,
)


class _Fn:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _Call:
    def __init__(self, call_id, name, arguments):
        self.id = call_id
        self.function = _Fn(name, arguments)


class _Message:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _Response:
    def __init__(self, message):
        self.choices = [type("Choice", (), {"message": message})()]


class FakeClient:
    """Returns scripted assistant messages and records each request."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.script:
            raise AssertionError("the model was called more times than the test scripted")
        return _Response(self.script.pop(0))


def _weather_response(url, params=None, timeout=None):
    class Response:
        def json(self):
            if "geocoding" in url:
                name = params["name"]
                if name.lower() == "seatle":
                    return {}
                catalog = {
                    "seattle": ("Seattle", 47.6, -122.3),
                    "tokyo": ("Tokyo", 35.7, 139.7),
                }
                found = catalog.get(name.lower())
                if not found:
                    return {}
                city, lat, lon = found
                return {"results": [{"name": city, "latitude": lat, "longitude": lon}]}
            temps = {47.6: 16.0, 35.7: 22.5}
            return {
                "current": {
                    "temperature_2m": temps.get(params["latitude"], 10.0),
                    "rain": 0.0,
                    "wind_speed_10m": 5.0,
                }
            }

    return Response()


def test_get_weather_reads_current_conditions():
    with patch("agent_lab.weather_agent.requests.get", side_effect=_weather_response):
        data = get_weather("Seattle")
    assert data == {"city": "Seattle", "temp_c": 16.0, "rain_mm": 0.0, "wind_kmh": 5.0}


def test_get_weather_reports_an_unknown_city():
    with patch("agent_lab.weather_agent.requests.get", side_effect=_weather_response):
        data = get_weather("Seatle")
    assert data == {"error": "Couldn't find a city called 'Seatle'"}


def test_live_seattle_weather():
    data = get_weather("Seattle")
    assert data["city"] == "Seattle"
    assert isinstance(data["temp_c"], (int, float))
    # Open-Meteo omits rain as null when the station has no reading.
    assert data["rain_mm"] is None or isinstance(data["rain_mm"], (int, float))
    assert isinstance(data["wind_kmh"], (int, float))


def test_get_time_uses_the_timezone_clock():
    data = get_time("America/Los_Angeles")
    assert data["timezone"] == "America/Los_Angeles"
    assert data["day"]
    assert data["time"]


def test_get_time_rejects_an_unknown_zone():
    data = get_time("Not/AZone")
    assert "Unknown timezone" in data["error"]
    assert "America/Los_Angeles" in data["error"]


def test_agent_calls_the_tool_then_answers(capsys):
    client = FakeClient(
        [
            _Message(tool_calls=[_Call("call_1", "get_weather", '{"city": "Seattle"}')]),
            _Message(content="No umbrella. Seattle is 16°C and dry."),
        ]
    )
    with patch("agent_lab.weather_agent.requests.get", side_effect=_weather_response):
        answer = run_agent(
            "Do I need an umbrella in Seattle right now?",
            client=client,
            model="test-model",
        )

    assert answer == "No umbrella. Seattle is 16°C and dry."
    assert "[tool] get_weather({'city': 'Seattle'})" in capsys.readouterr().out

    first, second = client.calls
    assert first["tools"][0]["function"]["name"] == "get_weather"
    assert first["model"] == "test-model"
    tool_msg = second["messages"][-1]
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"] == "call_1"
    assert '"temp_c": 16.0' in tool_msg["content"]


def test_agent_answers_without_a_tool_when_the_model_does_not_ask():
    client = FakeClient([_Message(content="I don't need a tool for that.")])
    answer = run_agent("What is 2 + 2?", client=client, model="test-model")
    assert answer == "I don't need a tool for that."
    assert len(client.calls) == 1


def test_typo_sends_the_error_back_and_the_model_tries_again():
    client = FakeClient(
        [
            _Message(tool_calls=[_Call("c1", "get_weather", '{"city": "Seatle"}')]),
            _Message(tool_calls=[_Call("c2", "get_weather", '{"city": "Seattle"}')]),
            _Message(content="Seattle is 16°C."),
        ]
    )
    with patch("agent_lab.weather_agent.requests.get", side_effect=_weather_response):
        answer = run_agent("What's the weather in Seatle?", client=client, model="test-model")

    assert answer == "Seattle is 16°C."
    error_turn = client.calls[1]["messages"][-1]["content"]
    assert "Couldn't find a city called 'Seatle'" in error_turn
    success_turn = client.calls[2]["messages"][-1]["content"]
    assert '"city": "Seattle"' in success_turn


def test_two_cities_in_one_turn():
    client = FakeClient(
        [
            _Message(
                tool_calls=[
                    _Call("t", "get_weather", '{"city": "Tokyo"}'),
                    _Call("s", "get_weather", '{"city": "Seattle"}'),
                ]
            ),
            _Message(content="Tokyo is warmer: 22.5°C versus 16°C in Seattle."),
        ]
    )
    with patch("agent_lab.weather_agent.requests.get", side_effect=_weather_response):
        answer = run_agent(
            "Is it warmer in Tokyo or Seattle right now?",
            client=client,
            model="test-model",
        )

    assert "Tokyo is warmer" in answer
    tool_msgs = [m for m in client.calls[1]["messages"] if isinstance(m, dict) and m.get("role") == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["t", "s"]
    assert '"temp_c": 22.5' in tool_msgs[0]["content"]
    assert '"temp_c": 16.0' in tool_msgs[1]["content"]


def test_loop_stops_after_too_many_tool_calls():
    client = FakeClient(
        [
            _Message(tool_calls=[_Call("1", "get_weather", '{"city": "Seattle"}')]),
            _Message(tool_calls=[_Call("2", "get_weather", '{"city": "Seattle"}')]),
        ]
    )
    with patch("agent_lab.weather_agent.requests.get", side_effect=_weather_response):
        answer = run_agent("weather?", max_turns=2, client=client, model="test-model")
    assert answer == "Stopped after too many tool calls."


def test_chat_remembers_the_previous_turn_and_can_call_both_tools():
    client = FakeClient(
        [
            _Message(tool_calls=[_Call("w", "get_weather", '{"city": "Seattle"}')]),
            _Message(content="Seattle is 16°C and dry."),
            _Message(tool_calls=[_Call("t", "get_time", '{"timezone": "America/Los_Angeles"}')]),
            _Message(content="It's evening on the west coast."),
        ]
    )
    chat = new_chat()
    with patch("agent_lab.weather_agent.requests.get", side_effect=_weather_response):
        first = run_chat_turn(chat, "What's the weather in Seattle right now?", client=client, model="test-model")
        second = run_chat_turn(chat, "What time is it there?", client=client, model="test-model")

    assert first == "Seattle is 16°C and dry."
    assert second == "It's evening on the west coast."
    assert chat[0]["role"] == "system"
    assert "Dubs" in chat[0]["content"]
    assert {spec["function"]["name"] for spec in chat_tools} == {"get_weather", "get_time"}

    # The clock request still contains the earlier weather question.
    time_request = client.calls[2]["messages"]
    contents = []
    for message in time_request:
        content = message["content"] if isinstance(message, dict) else message.content
        contents.append(content)
    assert "What's the weather in Seattle right now?" in contents
    assert "What time is it there?" in contents


def test_arguments_may_already_be_a_dict():
    client = FakeClient(
        [
            _Message(tool_calls=[_Call("1", "get_weather", {"city": "Tokyo"})]),
            _Message(content="Tokyo is 22.5°C."),
        ]
    )
    with patch("agent_lab.weather_agent.requests.get", side_effect=_weather_response):
        answer = run_agent("Tokyo weather?", client=client, model="test-model")
    assert answer == "Tokyo is 22.5°C."


def test_cli_weather_does_not_need_a_model_key(capsys):
    from agent_lab.weather_agent import main

    with patch("agent_lab.weather_agent.requests.get", side_effect=_weather_response):
        code = main(["--weather", "Seattle"])
    assert code == 0
    assert "temp_c" in capsys.readouterr().out


def test_cli_ask_without_a_key_exits(monkeypatch, capsys):
    from agent_lab.weather_agent import main

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("PROVIDER", "gemini")
    code = main(["--ask", "Is it raining in Seattle?"])
    assert code == 1
    assert "GEMINI_API_KEY" in capsys.readouterr().err
