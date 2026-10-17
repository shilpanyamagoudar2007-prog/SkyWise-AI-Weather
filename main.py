
import os

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types

load_dotenv()

app = FastAPI(title="SkyWise AI Weather Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini AI
api_key = os.getenv("GEMINI_API_KEY")
AI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

print("Gemini API key loaded:", bool(api_key))
print("Gemini model configured:", AI_MODEL)

ai_client = genai.Client(api_key=api_key) if api_key else None


def get_weather_description(code: int) -> str:
    descriptions = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Fog",
        48: "Depositing rime fog",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Heavy drizzle",
        61: "Light rain",
        63: "Moderate rain",
        65: "Heavy rain",
        71: "Light snow",
        73: "Moderate snow",
        75: "Heavy snow",
        80: "Rain showers",
        81: "Moderate rain showers",
        82: "Heavy rain showers",
        95: "Thunderstorm",
        96: "Thunderstorm with hail",
        99: "Thunderstorm with heavy hail",
    }
    return descriptions.get(code, "Unknown weather condition")


@app.get("/")
def home():
    return {"message": "SkyWise AI Weather Assistant is running"}


@app.get("/weather")
async def get_weather(
    city: str = Query(..., min_length=2, description="Enter a city name")
):
    city = city.strip()

    if len(city) < 2:
        raise HTTPException(
            status_code=400,
            detail="Please enter a valid city name.",
        )

    # Step 1: Find the city
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            geo_response = await client.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={
                    "name": city,
                    "count": 1,
                    "language": "en",
                    "format": "json",
                },
            )
            geo_response.raise_for_status()

            results = geo_response.json().get("results", [])

            if not results:
                raise HTTPException(
                    status_code=404,
                    detail="City not found. Please try another city.",
                )

            location = results[0]

            # Step 2: Fetch current weather
            weather_response = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": location["latitude"],
                    "longitude": location["longitude"],
                    "current": (
                        "temperature_2m,"
                        "relative_humidity_2m,"
                        "apparent_temperature,"
                        "precipitation,"
                        "weather_code,"
                        "wind_speed_10m"
                    ),
                    "timezone": "auto",
                },
            )
            weather_response.raise_for_status()

            current = weather_response.json().get("current")

            if not current:
                raise HTTPException(
                    status_code=502,
                    detail="Weather data is unavailable.",
                )

    except HTTPException:
        raise
    except httpx.HTTPError as error:
        print("Weather API error:", repr(error))
        raise HTTPException(
            status_code=502,
            detail="Unable to connect to the weather service.",
        ) from error
    except (KeyError, TypeError, ValueError) as error:
        print("Weather data error:", repr(error))
        raise HTTPException(
            status_code=502,
            detail="The weather service returned unexpected data.",
        ) from error

    # Step 3: Prepare weather details
    weather = {
        "city": location["name"],
        "country": location.get("country", "Unknown"),
        "temperature": current["temperature_2m"],
        "feels_like": current["apparent_temperature"],
        "humidity": current["relative_humidity_2m"],
        "wind_speed": current["wind_speed_10m"],
        "precipitation": current["precipitation"],
        "condition": get_weather_description(current["weather_code"]),
        "time": current["time"],
    }

    # Step 4: Check Gemini configuration
    if ai_client is None:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is missing from your .env file.",
        )

    # Step 5: Prepare the AI prompt using actual weather measurements
    prompt = f"""
You are SkyWise, a helpful AI weather assistant.

Use only the weather measurements provided.
Do not invent weather conditions or measurements.

Provide:
1. A brief weather summary.
2. Suitable clothing suggestions.
3. Outdoor activity advice.
4. Whether an umbrella may be useful.

Keep the answer concise and practical.

City: {weather['city']}, {weather['country']}
Temperature: {weather['temperature']} Celsius
Feels like: {weather['feels_like']} Celsius
Condition: {weather['condition']}
Humidity: {weather['humidity']} percent
Wind speed: {weather['wind_speed']} km/h
Precipitation: {weather['precipitation']} mm
"""

    # Step 6: Generate advice using Gemini
    try:
        print("Calling Gemini AI...")

        response = ai_client.models.generate_content(
            model=AI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You are SkyWise, an AI weather assistant. "
                    "Give clear, concise and useful weather advice."
                ),
                temperature=0.4,
                max_output_tokens=1000,
            ),
        )

        ai_advice = (response.text or "").strip()

        if not ai_advice:
            print("Gemini returned no text.")
            print("Response:", response.model_dump())

            raise HTTPException(
                status_code=502,
                detail=(
                    "Gemini returned an empty response. "
                    "Check the backend terminal."
                ),
            )

        print("Gemini response received successfully.")

    except HTTPException:
        raise
    except Exception as error:
        print(
            f"Gemini API error: {type(error).__name__}: {error}"
        )
        raise HTTPException(
            status_code=502,
            detail=(
                "Gemini AI request failed. "
                "Check the backend terminal for details."
            ),
        ) from error

    # Step 7: Return results to React
    return {
        "weather": weather,
        "ai_advice": ai_advice,
    }
