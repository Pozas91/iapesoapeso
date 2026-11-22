"""A ReAct loop written by hand: the model only asks for the tool, this program runs it.

The model gets the question and the description of one tool, tiempo(ciudad). When it needs the
weather it does not fetch anything: it answers with a request (a tool_use block: the tool's name and
its input) and stops. This program reads that request, calls the weather service and sends back what
it returned; the model reads it and writes the answer. Every step, the model's exact output and the
tokens of each call are printed and saved to trace.json, which is where the Short's figures come from.

The original ReAct paper (Yao et al., 2022) wrote the same loop as free text («Thought / Action /
Observation»); today's APIs carry the request as a structured block, with the same split of roles.

Needs ANTHROPIC_API_KEY in the environment. Weather from Open-Meteo (free, no key).
"""

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import anthropic

MODEL = "claude-opus-5-5"
QUESTION = "¿Qué tiempo hace ahora mismo en Madrid?"
SYSTEM = (
    "Respondes en español, en una o dos frases. No sabes el tiempo actual de ningún sitio salvo por la herramienta."
)
TOOL_SPECS = [
    {
        "name": "tiempo",
        "description": "Devuelve la temperatura y el estado del cielo ahora mismo en una ciudad.",
        "input_schema": {
            "type": "object",
            "properties": {"ciudad": {"type": "string", "description": "Nombre de la ciudad"}},
            "required": ["ciudad"],
            "additionalProperties": False,
        },
        "strict": True,
    }
]
WEATHER_CODES = {
    0: "despejado",
    1: "casi despejado",
    2: "parcialmente nublado",
    3: "nublado",
    45: "niebla",
    48: "niebla con escarcha",
    51: "llovizna débil",
    53: "llovizna",
    55: "llovizna intensa",
    61: "lluvia débil",
    63: "lluvia",
    65: "lluvia intensa",
    71: "nieve débil",
    73: "nieve",
    75: "nieve intensa",
    80: "chubascos débiles",
    81: "chubascos",
    82: "chubascos fuertes",
    95: "tormenta",
}


def get_json(url: str, params: dict, attempts: int = 3) -> dict:
    request = urllib.request.Request(f"{url}?{urllib.parse.urlencode(params)}", headers={"User-Agent": "react-demo"})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code < 500 or attempt == attempts - 1:
                raise
            time.sleep(2)
    raise RuntimeError("unreachable")


def tiempo(ciudad: str) -> str:
    """The tool: current temperature and sky in a city, from Open-Meteo."""
    place = get_json("https://geocoding-api.open-meteo.com/v1/search", {"name": ciudad, "count": 1, "language": "es"})[
        "results"
    ][0]
    now = get_json(
        "https://api.open-meteo.com/v1/forecast",
        {
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "current": "temperature_2m,weather_code",
            "timezone": "auto",
        },
    )["current"]
    sky = WEATHER_CODES.get(now["weather_code"], f"código {now['weather_code']}")
    return f"{place['name']}: {now['temperature_2m']} °C, {sky} ({now['time']} hora local)"


TOOLS = {"tiempo": tiempo}


def main() -> None:
    client = anthropic.Anthropic()
    messages: list = [{"role": "user", "content": QUESTION}]
    steps = []
    print(f"Pregunta: {QUESTION}", end="\n\n")
    for call in range(1, 6):
        response = client.beta.messages.create(
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",  # if the model declines, the API retries on a fallback model
            model=MODEL,
            max_tokens=2000,
            system=SYSTEM,
            output_config={"effort": "low"},
            tools=TOOL_SPECS,
            messages=messages,
        )
        if response.stop_reason == "refusal":
            raise SystemExit(f"Refused: {response.stop_details}")
        text = "".join(block.text for block in response.content if block.type == "text").strip()
        requests = [block for block in response.content if block.type == "tool_use"]
        usage = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
        print(
            f"── llamada {call} al modelo ({usage['input_tokens']} tokens de entrada, {usage['output_tokens']} de salida)"
        )
        if text:
            print(f"Texto: {text}")
        step = {"call": call, "stop_reason": response.stop_reason, "model_text": text, "usage": usage}
        messages.append({"role": "assistant", "content": response.content})  # thinking blocks kept as returned
        if response.stop_reason != "tool_use":
            steps.append(step)
            print()
            break

        results = []
        step["tool_requests"] = []
        for request in requests:
            observation = TOOLS[request.name](**request.input)
            shown = json.dumps(request.input, ensure_ascii=False)
            print(f"Petición del modelo: {request.name}({shown})")
            print(f"── el programa la ejecuta y devuelve: {observation}")
            step["tool_requests"].append({"name": request.name, "input": request.input, "result": observation})
            results.append({"type": "tool_result", "tool_use_id": request.id, "content": observation})
        print()
        steps.append(step)
        messages.append({"role": "user", "content": results})

    total_in = sum(s["usage"]["input_tokens"] for s in steps)
    total_out = sum(s["usage"]["output_tokens"] for s in steps)
    print(
        f"Llamadas al modelo: {len(steps)} · herramientas ejecutadas por el programa: "
        f"{sum(len(s.get('tool_requests', [])) for s in steps)} · tokens: {total_in} de entrada, {total_out} de salida"
    )
    trace = {
        "run_at": datetime.now(ZoneInfo("Europe/Madrid")).isoformat(timespec="seconds"),
        "model": MODEL,
        "question": QUESTION,
        "system": SYSTEM,
        "tools": TOOL_SPECS,
        "steps": steps,
    }
    out = Path(__file__).with_name("trace.json")
    out.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Traza guardada en {out.name}")


if __name__ == "__main__":
    main()
