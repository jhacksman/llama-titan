"""Madrone Way dashboard server.

Single-page real-time dashboard for 7223 NW Madrone Way, Corvallis OR
(Vineyard Mountain, unincorporated Benton County, Corvallis Rural Fire
Protection District).

Run:  uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import sources
from .sun import sun_times

TZ = ZoneInfo("America/Los_Angeles")
LAT, LON = sources.LAT, sources.LON

STATIC = Path(__file__).resolve().parent / "static"

app = FastAPI(title="7223 NW Madrone Way dashboard")


@app.get("/api/now")
async def now():
    t = datetime.now(TZ)
    sunrise, noon, sunset = sun_times(t.date(), LAT, LON)
    rise_tom, _, set_tom = sun_times(t.date() + timedelta(days=1), LAT, LON)

    sr, ss = sunrise.astimezone(TZ), sunset.astimezone(TZ)
    is_day = sr <= t <= ss
    day_len = ss - sr
    return {
        "iso": t.isoformat(),
        "tz": "America/Los_Angeles",
        "sunrise": sr.isoformat(),
        "sunset": ss.isoformat(),
        "solarNoon": noon.astimezone(TZ).isoformat(),
        "sunriseTomorrow": rise_tom.astimezone(TZ).isoformat(),
        "sunsetTomorrow": set_tom.astimezone(TZ).isoformat(),
        "isDaytime": is_day,
        "dayLengthMinutes": round(day_len.total_seconds() / 60),
    }


@app.get("/api/weather")
async def weather():
    return await sources.weather()


@app.get("/api/burn")
async def burn():
    return await sources.burn()


@app.get("/api/fire")
async def fire():
    return await sources.fire()


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
