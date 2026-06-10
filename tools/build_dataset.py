#!/usr/bin/env python3
"""Build the static neighborhood dataset for 7223 NW Madrone Way, Corvallis, OR.

Fetches real plat/parcel, address, and street data from public services and
bakes it into app/static/data/neighborhood.json:

  * Taxlot polygons  — Benton County Taxlots, mirrored by ODOT's public
                       ArcGIS server (layer "ames/ames/MapServer/29").
  * Address points   — USDOT National Address Database (NAD) hosted view
                       on ArcGIS Online (Benton County submits its address
                       points to the NAD).
  * Streets          — OpenStreetMap via the Overpass API.

The neighborhood doesn't change much, so this is run manually whenever a
refresh is wanted:  python3 tools/build_dataset.py
"""

import json
import math
import sys
import urllib.parse
import urllib.request
from pathlib import Path

HOUSE = {
    "address": "7223 NW Madrone Way, Corvallis, OR 97330",
    "lat": 44.643735286056646,
    "lon": -123.24262074444363,
}

# ~1 mile around the house; covers the whole Vineyard Mountain neighborhood.
BBOX = (-123.2630, 44.6294, -123.2226, 44.6583)  # W, S, E, N

TAXLOT_URL = (
    "https://gis.odot.state.or.us/arcgis1006/rest/services/ames/ames/MapServer/29/query"
)
NAD_URL = (
    "https://services.arcgis.com/xOi1kZaI0eWDREZv/ArcGIS/rest/services/"
    "Address_Points_from_National_Address_Database_view/FeatureServer/0/query"
)
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

OUT = Path(__file__).resolve().parent.parent / "app" / "static" / "data" / "neighborhood.json"

RING_MILES = [0.25, 0.5, 1.0]


def post(url: str, params: dict) -> dict:
    data = urllib.parse.urlencode(params).encode()
    req = urllib.request.Request(url, data=data, headers={"User-Agent": "madrone-dashboard/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def dist_miles(lat1, lon1, lat2, lon2):
    r = 3958.7613
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def fetch_taxlots():
    d = post(TAXLOT_URL, {
        "geometry": ",".join(str(v) for v in BBOX),
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "MapTaxlot",
        "outSR": "4326",
        "returnGeometry": "true",
        "maxAllowableOffset": "0.00002",
        "f": "json",
    })
    lots = []
    for f in d["features"]:
        rings = [[[round(x, 6), round(y, 6)] for x, y in ring]
                 for ring in f["geometry"]["rings"]]
        lots.append({"id": f["attributes"]["MapTaxlot"], "rings": rings})
    return lots


def fetch_addresses():
    d = post(NAD_URL, {
        "geometry": ",".join(str(v) for v in BBOX),
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "Add_Number,StNam_Full,Post_City,Zip_Code",
        "outSR": "4326",
        "returnGeometry": "true",
        "resultRecordCount": "2000",
        "f": "json",
    })
    pts = []
    for f in d["features"]:
        a = f["attributes"]
        g = f["geometry"]
        pts.append({
            "num": a.get("Add_Number"),
            "street": a.get("StNam_Full"),
            "lat": round(g["y"], 6),
            "lon": round(g["x"], 6),
            "mi": round(dist_miles(HOUSE["lat"], HOUSE["lon"], g["y"], g["x"]), 4),
        })
    pts.sort(key=lambda p: p["mi"])
    return pts


def fetch_streets():
    w, s, e, n = BBOX
    q = (
        f'[out:json][timeout:90];'
        f'(way["highway"]({s},{w},{n},{e}););out tags geom;'
    )
    data = urllib.parse.urlencode({"data": q}).encode()
    req = urllib.request.Request(OVERPASS_URL, data=data,
                                 headers={"User-Agent": "madrone-dashboard/1.0"})
    with urllib.request.urlopen(req, timeout=150) as r:
        d = json.loads(r.read())
    streets = []
    for el in d["elements"]:
        tags = el.get("tags", {})
        hw = tags.get("highway", "")
        if hw in ("footway", "steps", "bridleway", "cycleway"):
            continue
        pts = [[round(p["lon"], 6), round(p["lat"], 6)] for p in el.get("geometry", [])]
        if len(pts) < 2:
            continue
        streets.append({
            "name": tags.get("name", ""),
            "kind": hw,
            "pts": pts,
        })
    return streets


def main():
    print("fetching taxlots (Benton County via ODOT ArcGIS) ...")
    lots = fetch_taxlots()
    print(f"  {len(lots)} parcels")

    print("fetching address points (USDOT National Address Database) ...")
    addrs = fetch_addresses()
    print(f"  {len(addrs)} addresses")

    print("fetching streets (OpenStreetMap Overpass) ...")
    streets = fetch_streets()
    print(f"  {len(streets)} street segments")

    rings = []
    for rmi in RING_MILES:
        inside = [a for a in addrs if a["mi"] <= rmi]
        per_street = {}
        for a in inside:
            per_street[a["street"]] = per_street.get(a["street"], 0) + 1
        rings.append({
            "miles": rmi,
            "addresses": len(inside),
            "streets": dict(sorted(per_street.items(), key=lambda kv: -kv[1])),
        })

    out = {
        "house": HOUSE,
        "bbox": BBOX,
        "rings": rings,
        "addresses": addrs,
        "streets": streets,
        "parcels": lots,
        "sources": {
            "parcels": "Benton County Taxlots via ODOT ArcGIS (gis.odot.state.or.us)",
            "addresses": "USDOT National Address Database hosted view (ArcGIS Online)",
            "streets": "OpenStreetMap via Overpass API",
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {OUT} ({OUT.stat().st_size/1024:.0f} KB)")
    for r in rings:
        print(f"  within {r['miles']} mi: {r['addresses']} addresses")


if __name__ == "__main__":
    sys.exit(main())
