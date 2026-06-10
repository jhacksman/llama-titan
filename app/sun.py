"""Sunrise/sunset/solar-noon computation (NOAA solar calculator algorithm).

Pure stdlib so the dashboard never depends on an external sunrise API.
Accurate to well under a minute at this latitude.
"""

import math
from datetime import date, datetime, timedelta, timezone


def _julian_day(d: date) -> float:
    y, m = d.year, d.month
    if m <= 2:
        y -= 1
        m += 12
    a = y // 100
    b = 2 - a + a // 4
    return int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d.day + b - 1524.5


def _sun_geometry(jc: float):
    """Return (declination_deg, eq_of_time_minutes) for a Julian century."""
    gmls = (280.46646 + jc * (36000.76983 + jc * 0.0003032)) % 360
    gmas = 357.52911 + jc * (35999.05029 - 0.0001537 * jc)
    eeo = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)
    seoc = (
        math.sin(math.radians(gmas)) * (1.914602 - jc * (0.004817 + 0.000014 * jc))
        + math.sin(math.radians(2 * gmas)) * (0.019993 - 0.000101 * jc)
        + math.sin(math.radians(3 * gmas)) * 0.000289
    )
    stl = gmls + seoc
    sal = stl - 0.00569 - 0.00478 * math.sin(math.radians(125.04 - 1934.136 * jc))
    moe = 23 + (26 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60) / 60
    oc = moe + 0.00256 * math.cos(math.radians(125.04 - 1934.136 * jc))
    decl = math.degrees(math.asin(math.sin(math.radians(oc)) * math.sin(math.radians(sal))))
    vary = math.tan(math.radians(oc / 2)) ** 2
    eot = 4 * math.degrees(
        vary * math.sin(2 * math.radians(gmls))
        - 2 * eeo * math.sin(math.radians(gmas))
        + 4 * eeo * vary * math.sin(math.radians(gmas)) * math.cos(2 * math.radians(gmls))
        - 0.5 * vary * vary * math.sin(4 * math.radians(gmls))
        - 1.25 * eeo * eeo * math.sin(2 * math.radians(gmas))
    )
    return decl, eot


def sun_times(d: date, lat: float, lon: float):
    """Return (sunrise, solar_noon, sunset) as aware UTC datetimes.

    Returns None for sunrise/sunset if the sun never rises/sets that day
    (not possible at 44°N, but handled anyway).
    """
    jd = _julian_day(d)
    jc = (jd - 2451545) / 36525
    decl, eot = _sun_geometry(jc)

    noon_min = 720 - 4 * lon - eot  # minutes after 00:00 UTC
    midnight = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
    noon = midnight + timedelta(minutes=noon_min)

    cos_ha = (
        math.cos(math.radians(90.833))
        / (math.cos(math.radians(lat)) * math.cos(math.radians(decl)))
        - math.tan(math.radians(lat)) * math.tan(math.radians(decl))
    )
    if cos_ha > 1 or cos_ha < -1:
        return None, noon, None
    ha = math.degrees(math.acos(cos_ha))
    sunrise = midnight + timedelta(minutes=noon_min - ha * 4)
    sunset = midnight + timedelta(minutes=noon_min + ha * 4)
    return sunrise, noon, sunset
