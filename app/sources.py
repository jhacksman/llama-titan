"""Live data fetchers for the Madrone Way dashboard.

Every fetcher is async, keyless, and cached with a TTL so a page reload
never hammers the upstream services. Failures return a payload with an
"error" field instead of raising, so one dead upstream never takes the
page down.
"""

import asyncio
import html as html_mod
import re
import time

import httpx

LAT = 44.643735
LON = -123.242621

UA = {"User-Agent": "madrone-dashboard/1.0 (self-hosted home dashboard)"}

NWS_POINT = f"https://api.weather.gov/points/{LAT},{LON}"
NWS_FORECAST = "https://api.weather.gov/gridpoints/PQR/86,67/forecast"
NWS_HOURLY = "https://api.weather.gov/gridpoints/PQR/86,67/forecast/hourly"
NWS_ALERTS = f"https://api.weather.gov/alerts/active?point={LAT},{LON}"
NWS_FIRE_ZONE_ALERTS = "https://api.weather.gov/alerts/active?zone=ORZ683"
ODF_BURN = "https://smkmgt.com/burn.php"
WFIGS = (
    "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/"
    "WFIGS_Incident_Locations_Current/FeatureServer/0/query"
)
OPEN_METEO_AQ = (
    "https://air-quality-api.open-meteo.com/v1/air-quality"
    f"?latitude={LAT}&longitude={LON}"
    "&current=us_aqi,pm2_5,pm10,ozone"
)

_cache: dict[str, tuple[float, dict]] = {}
_locks: dict[str, asyncio.Lock] = {}


async def _cached(key: str, ttl: float, fetch):
    lock = _locks.setdefault(key, asyncio.Lock())
    async with lock:
        hit = _cache.get(key)
        if hit and time.monotonic() - hit[0] < ttl:
            return hit[1]
        try:
            value = await fetch()
        except Exception as e:  # serve stale data over an error if we have it
            if hit:
                return hit[1]
            value = {"error": f"{type(e).__name__}: {e}"}
        _cache[key] = (time.monotonic(), value)
        return value


async def _get_json(client: httpx.AsyncClient, url: str) -> dict:
    r = await client.get(url, headers=UA, timeout=25, follow_redirects=True)
    r.raise_for_status()
    return r.json()


async def weather() -> dict:
    async def fetch():
        async with httpx.AsyncClient() as client:
            fc, hr, al = await asyncio.gather(
                _get_json(client, NWS_FORECAST),
                _get_json(client, NWS_HOURLY),
                _get_json(client, NWS_ALERTS),
            )
        periods = fc["properties"]["periods"][:8]
        hours = hr["properties"]["periods"][:24]
        return {
            "updated": fc["properties"].get("updateTime"),
            "periods": [
                {
                    "name": p["name"],
                    "isDaytime": p["isDaytime"],
                    "tempF": p["temperature"],
                    "wind": f"{p['windSpeed']} {p['windDirection']}",
                    "short": p["shortForecast"],
                    "detail": p["detailedForecast"],
                    "rainChance": (p.get("probabilityOfPrecipitation") or {}).get("value"),
                }
                for p in periods
            ],
            "hourly": [
                {
                    "time": h["startTime"],
                    "tempF": h["temperature"],
                    "short": h["shortForecast"],
                    "rainChance": (h.get("probabilityOfPrecipitation") or {}).get("value"),
                    "wind": f"{h['windSpeed']} {h['windDirection']}",
                }
                for h in hours
            ],
            "alerts": [
                {
                    "event": a["properties"]["event"],
                    "severity": a["properties"]["severity"],
                    "headline": a["properties"]["headline"],
                    "expires": a["properties"]["expires"],
                }
                for a in al.get("features", [])
            ],
            "source": "National Weather Service (api.weather.gov), Portland OR office, grid PQR 86,67",
        }

    return await _cached("weather", 300, fetch)


def _strip_html(raw: str) -> str:
    txt = re.sub(r"<script.*?</script>", "", raw, flags=re.S | re.I)
    txt = re.sub(r"<style.*?</style>", "", txt, flags=re.S | re.I)
    txt = re.sub(r"<[^>]+>", " ", txt)
    return html_mod.unescape(re.sub(r"\s+", " ", txt)).strip()


async def burn() -> dict:
    """Scrape the ODF/DEQ daily Willamette Valley open-burning announcement."""
    async def fetch():
        async with httpx.AsyncClient() as client:
            r = await client.get(ODF_BURN, headers=UA, timeout=25, follow_redirects=True)
            r.raise_for_status()
        text = _strip_html(r.text)

        m = re.search(r"Open Burn Announcement for ([A-Za-z]+,? [A-Za-z]+ \d+\w*,? \d{4})", text)
        announce_date = m.group(1) if m else None

        # Pull the operative lines about what burning is allowed/recommended.
        statements = []
        for pat in (
            r"Agricultural burning:?\s*(.*?)(?=Backyard burning|➤|$)",
            r"Backyard burning[^:]*:?\s*(.*?)(?=➤|Go to|$)",
            r"(No open burning[^.]*\.)",
            r"(Burning is not (?:recommended|advised)[^.]*\.)",
        ):
            mm = re.search(pat, text, flags=re.I)
            if mm:
                s = mm.group(1).strip()
                if s and s not in statements:
                    statements.append(s[:400])

        return {
            "announcementDate": announce_date,
            "statements": statements,
            "advisory": "This announcement is not approval to burn. Local rules and "
                        "regulations still apply — call before you burn.",
            "burnLine": {
                "label": "Corvallis-area daily burn advisory line (updated ~8:15 AM)",
                "phone": "541-766-6971",
            },
            "links": [
                {"label": "Corvallis Rural Fire Protection District — burn regulations",
                 "url": "https://www.corvallisrfpd.com/burn-regulations"},
                {"label": "Benton County outdoor burning / wildfire protection",
                 "url": "https://cd.bentoncountyor.gov/wildfire-protection/outdoor-burning/"},
                {"label": "ODF Willamette Valley daily open burning announcement",
                 "url": "https://smkmgt.com/burn.php"},
                {"label": "DEQ outdoor & open burning rules",
                 "url": "https://www.oregon.gov/deq/aq/pages/burning.aspx"},
            ],
            "seasons": "DEQ backyard burn seasons in the Corvallis area: spring "
                       "Mar 1 – Jun 15, fall Oct 1 – Dec 15. Burn days are declared daily.",
            "source": "Oregon Dept. of Forestry meteorologist daily announcement (smkmgt.com)",
        }

    return await _cached("burn", 900, fetch)


async def fire() -> dict:
    """Fire-hazard picture: nearby active wildfires, fire-weather alerts, air quality."""
    async def fetch():
        params = {
            "geometry": f"{LON},{LAT}",
            "geometryType": "esriGeometryPoint",
            "inSR": "4326",
            "distance": "100000",
            "units": "esriSRUnit_Meter",
            "outFields": "IncidentName,FireDiscoveryDateTime,IncidentSize,PercentContained",
            "returnGeometry": "true",
            "outSR": "4326",
            "f": "json",
        }
        async with httpx.AsyncClient() as client:
            inc_r, fz, aq = await asyncio.gather(
                client.get(WFIGS, params=params, headers=UA, timeout=30),
                _get_json(client, NWS_FIRE_ZONE_ALERTS),
                _get_json(client, OPEN_METEO_AQ),
            )
        inc = inc_r.json()

        import math
        def miles(lat2, lon2):
            p1, p2 = math.radians(LAT), math.radians(lat2)
            a = (math.sin(math.radians(lat2 - LAT) / 2) ** 2
                 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - LON) / 2) ** 2)
            return 2 * 3958.7613 * math.asin(math.sqrt(a))

        fires = []
        for f in inc.get("features", []):
            a, g = f["attributes"], f.get("geometry") or {}
            if "y" not in g:
                continue
            fires.append({
                "name": a.get("IncidentName"),
                "acres": a.get("IncidentSize"),
                "contained": a.get("PercentContained"),
                "discovered": a.get("FireDiscoveryDateTime"),
                "miles": round(miles(g["y"], g["x"]), 1),
                "lat": g["y"], "lon": g["x"],
            })
        fires.sort(key=lambda x: x["miles"])

        cur = aq.get("current", {})
        return {
            "activeFiresWithin60mi": fires[:12],
            "fireWeatherAlerts": [
                {
                    "event": a["properties"]["event"],
                    "headline": a["properties"]["headline"],
                    "expires": a["properties"]["expires"],
                }
                for a in fz.get("features", [])
            ],
            "airQuality": {
                "usAqi": cur.get("us_aqi"),
                "pm2_5": cur.get("pm2_5"),
                "pm10": cur.get("pm10"),
                "ozone": cur.get("ozone"),
                "time": cur.get("time"),
                "source": "Open-Meteo air quality (CAMS)",
            },
            "links": [
                {"label": "ODF fire restrictions & danger levels",
                 "url": "https://www.oregon.gov/odf/fire/pages/restrictions.aspx"},
                {"label": "Benton County evacuation & emergency alerts (sign up)",
                 "url": "https://www.bentoncountyor.gov/sheriff/emergency-management/"},
                {"label": "Oregon wildfire map (ODF/NIFC)",
                 "url": "https://www.oregon.gov/odf/fire/pages/firestats.aspx"},
                {"label": "NW Madrone Way is bordered by McDonald-Dunn Research Forest (OSU)",
                 "url": "https://cf.forestry.oregonstate.edu/visit"},
            ],
            "source": "NIFC WFIGS current incidents; NWS fire zone ORZ683; Open-Meteo AQ",
        }

    return await _cached("fire", 600, fetch)
