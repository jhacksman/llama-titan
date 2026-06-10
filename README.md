# 7223 NW Madrone Way — live dashboard

A single self-hosted web page for **7223 NW Madrone Way, Corvallis OR 97330**
(Vineyard Mountain, unincorporated Benton County, Corvallis Rural Fire
Protection District). Every load renders from real-time data:

| Card | Live source |
|---|---|
| **Burn status today** | Oregon Dept. of Forestry daily Willamette Valley open-burning announcement (smkmgt.com), scraped server-side and cached 15 min, plus the Corvallis-area burn advisory line **541-766-6971** and CRFPD / Benton County / DEQ links |
| **Weather** | National Weather Service `api.weather.gov` — hourly + 7-day forecast and active alerts for this exact gridpoint (PQR 86,67) |
| **Sun & sky** | Sunrise / solar noon / sunset / day length computed on the server with the NOAA solar algorithm for the rooftop coordinates — no external API |
| **Fire hazard** | Active wildfires within 60 mi (NIFC WFIGS), Red Flag / fire-weather alerts for NWS fire zone ORZ683, US AQI + PM2.5 (Open-Meteo) |
| **Clock / theme** | Live Pacific-time clock; the whole page switches between day and night themes automatically at the real sunrise/sunset |
| **The neighborhood** | Custom pan/zoom canvas map (no map tiles): 933 Benton County taxlot polygons, 663 real address points, and the street network — with ¼ / ½ / 1 mile rings quantizing the neighbors (94 / 244 / 563 addresses) |

Neighborhood data is baked into `app/static/data/neighborhood.json` from:

* **Parcels** — Benton County Taxlots via ODOT's public ArcGIS server
* **Addresses** — USDOT National Address Database (Benton County submission)
* **Streets** — OpenStreetMap (Overpass API)

Refresh it any time with `python3 tools/build_dataset.py`.

## Run it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
```

then open http://localhost:8000.

### Docker

```bash
docker compose up -d
```

### systemd (bare metal)

See `deploy/madrone-dashboard.service`.

## Layout

```
app/main.py        FastAPI app + /api/now (time & sun)
app/sources.py     cached fetchers: NWS weather/alerts, ODF burn scrape, WFIGS fires, Open-Meteo AQ
app/sun.py         NOAA sunrise/sunset math (pure stdlib)
app/static/        index.html, style.css, app.js, data/neighborhood.json
tools/build_dataset.py   re-bake the neighborhood dataset
archive/           the previous llama-titan project, untouched
```

All upstream calls are keyless, cached server-side (5–15 min TTLs), and fail
soft — a dead upstream shows an error card, never a dead page.
