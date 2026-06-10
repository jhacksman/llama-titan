/* Live dashboard for 7223 NW Madrone Way. Everything renders from real-time
   API calls on every load; the neighborhood map renders baked county/NAD/OSM
   data client-side on a pan-zoomable canvas. */

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

const fmtTime = (iso) => new Date(iso).toLocaleTimeString("en-US",
  { hour: "numeric", minute: "2-digit", timeZone: "America/Los_Angeles" });

/* ---------------- clock + sun + auto day/night theme ---------------- */

let sunInfo = null;

function tickClock() {
  const now = new Date();
  $("clock").textContent = now.toLocaleTimeString("en-US",
    { hour12: true, hour: "numeric", minute: "2-digit", second: "2-digit", timeZone: "America/Los_Angeles" });
  $("clockdate").textContent = now.toLocaleDateString("en-US",
    { weekday: "long", month: "long", day: "numeric", year: "numeric", timeZone: "America/Los_Angeles" }) + " — Pacific";

  if (sunInfo) {
    const rise = new Date(sunInfo.sunrise), set = new Date(sunInfo.sunset);
    const isDay = now >= rise && now <= set;
    document.body.dataset.theme = isDay ? "day" : "night";
    // sun position dot: 10%..90% of the track maps sunrise..sunset
    let frac;
    if (isDay) frac = 0.1 + 0.8 * ((now - rise) / (set - rise));
    else if (now < rise) frac = 0.05;
    else frac = 0.95;
    $("sundot").style.left = (frac * 100).toFixed(2) + "%";
  }
}

async function loadNow() {
  const d = await (await fetch("/api/now")).json();
  sunInfo = d;
  $("sunline").textContent =
    `sunrise ${fmtTime(d.sunrise)} · sunset ${fmtTime(d.sunset)}`;

  const h = Math.floor(d.dayLengthMinutes / 60), m = d.dayLengthMinutes % 60;
  $("sun-body").innerHTML = `
    <div class="kv"><span class="k">Sunrise today</span><b>${fmtTime(d.sunrise)}</b></div>
    <div class="kv"><span class="k">Solar noon</span><b>${fmtTime(d.solarNoon)}</b></div>
    <div class="kv"><span class="k">Sunset today</span><b>${fmtTime(d.sunset)}</b></div>
    <div class="kv"><span class="k">Day length</span><b>${h} h ${m} min</b></div>
    <div class="kv"><span class="k">Sunrise tomorrow</span><b>${fmtTime(d.sunriseTomorrow)}</b></div>
    <div class="kv"><span class="k">Sunset tomorrow</span><b>${fmtTime(d.sunsetTomorrow)}</b></div>
    <p class="statline" style="color:var(--ink-soft)">Computed for this exact rooftop
    (44.6437°N, 123.2426°W) — NOAA solar algorithm, no external API.</p>`;
  tickClock();
}

/* ---------------- weather ---------------- */

async function loadWeather() {
  try {
    const d = await (await fetch("/api/weather")).json();
    if (d.error) throw new Error(d.error);

    if (d.alerts && d.alerts.length) {
      $("alertbar").classList.remove("hidden");
      $("alertbar").innerHTML = d.alerts.map(a =>
        `⚠ ${esc(a.event)} — ${esc(a.headline)}`).join("<br>");
    }

    const nowH = d.hourly[0];
    const strip = d.hourly.slice(0, 18).map(hh => {
      const t = new Date(hh.time).toLocaleTimeString("en-US",
        { hour: "numeric", timeZone: "America/Los_Angeles" }).replace(" ", "");
      const px = Math.max(4, Math.round((hh.tempF - 30) * 0.9));
      return `<div class="hourcell"><div class="t">${hh.tempF}°</div>
        <div class="bar"><i style="height:${px}px"></i></div>${t}</div>`;
    }).join("");

    const periods = d.periods.slice(0, 4).map(p => `
      <div class="kv"><span class="k">${esc(p.name)}</span>
      <b>${p.tempF}° · ${esc(p.short)}${p.rainChance ? ` · ${p.rainChance}% rain` : ""}</b></div>`).join("");

    $("weather-body").innerHTML = `
      <div class="bignum">${nowH.tempF}°F</div>
      <div class="statline">${esc(nowH.short)} · wind ${esc(nowH.wind)}</div>
      <div class="hourstrip">${strip}</div>
      ${periods}
      <p class="statline" style="color:var(--ink-soft)">${esc(d.periods[0].detail)}</p>`;
  } catch (e) {
    $("weather-body").innerHTML = `<p class="err">NWS unavailable: ${esc(e.message)}</p>`;
  }
}

/* ---------------- burn status ---------------- */

async function loadBurn() {
  try {
    const d = await (await fetch("/api/burn")).json();
    if (d.error) throw new Error(d.error);

    const txt = (d.statements || []).join(" ").toLowerCase();
    let cls = "unk", verdict = "Check before you burn";
    if (/no open burning|not recommended|not advised|prohibited/.test(txt)) {
      cls = "no"; verdict = "Burning NOT advised today";
    } else if (/allowed|permitted/.test(txt)) {
      cls = "ok"; verdict = "Burning conditionally allowed today";
    }
    $("burn-dot").className = "dot " + cls;

    $("burn-body").innerHTML = `
      <div class="bignum" style="font-size:1.35rem">${verdict}</div>
      <div class="statline" style="color:var(--ink-soft)">ODF announcement for ${esc(d.announcementDate || "today")}</div>
      ${(d.statements || []).map(s => `<p class="statline">• ${esc(s)}</p>`).join("")}
      <p class="statline"><b>${esc(d.advisory)}</b></p>
      <div class="kv"><span class="k">${esc(d.burnLine.label)}</span>
        <b><a href="tel:${d.burnLine.phone.replace(/-/g, "")}">${esc(d.burnLine.phone)}</a></b></div>
      <p class="statline" style="color:var(--ink-soft)">${esc(d.seasons)}</p>
      <ul class="linklist">${d.links.map(l =>
        `<li><a target="_blank" rel="noopener" href="${esc(l.url)}">${esc(l.label)}</a></li>`).join("")}</ul>`;
  } catch (e) {
    $("burn-dot").className = "dot unk";
    $("burn-body").innerHTML = `<p class="err">Burn advisory fetch failed: ${esc(e.message)}</p>
      <p>Call the burn line: <a href="tel:5417666971"><b>541-766-6971</b></a> (updated ~8:15 AM)</p>`;
  }
}

/* ---------------- fire hazard ---------------- */

function aqiLabel(v) {
  if (v == null) return ["–", ""];
  if (v <= 50) return [v, "Good"];
  if (v <= 100) return [v, "Moderate"];
  if (v <= 150) return [v, "Unhealthy (sensitive)"];
  if (v <= 200) return [v, "Unhealthy"];
  return [v, "Very unhealthy"];
}

async function loadFire() {
  try {
    const d = await (await fetch("/api/fire")).json();
    if (d.error) throw new Error(d.error);

    const [aqi, aqiTxt] = aqiLabel(d.airQuality.usAqi);
    const fwa = d.fireWeatherAlerts || [];
    const fires = d.activeFiresWithin60mi || [];

    $("fire-body").innerHTML = `
      <div class="kv"><span class="k">Red Flag / fire-weather alerts (zone ORZ683)</span>
        <b>${fwa.length ? "" : "none active"}</b></div>
      ${fwa.map(a => `<p class="statline err">⚠ ${esc(a.event)} — ${esc(a.headline)}</p>`).join("")}
      <div class="kv"><span class="k">Active wildfires within 60 mi (NIFC)</span><b>${fires.length}</b></div>
      <div class="firelist">${fires.slice(0, 5).map(f => `
        <div class="kv"><span class="k">${esc(f.name)} · ${f.acres ?? "?"} ac</span>
        <b>${f.miles} mi</b></div>`).join("")}</div>
      <div class="kv"><span class="k">Air quality (US AQI)</span><b>${aqi} ${esc(aqiTxt)}</b></div>
      <div class="kv"><span class="k">PM2.5</span><b>${d.airQuality.pm2_5 ?? "–"} µg/m³</b></div>
      <p class="statline" style="color:var(--ink-soft)">This house sits against the
      McDonald-Dunn Research Forest — wildland-urban interface. Keep defensible space;
      sign up for Benton County emergency alerts.</p>
      <ul class="linklist">${d.links.map(l =>
        `<li><a target="_blank" rel="noopener" href="${esc(l.url)}">${esc(l.label)}</a></li>`).join("")}</ul>`;
  } catch (e) {
    $("fire-body").innerHTML = `<p class="err">Fire data unavailable: ${esc(e.message)}</p>`;
  }
}

/* ---------------- neighborhood map ---------------- */

const RING_COLORS = ["#e4572e", "#f2a93b", "#57a773"];

async function loadMap() {
  const d = await (await fetch("/static/data/neighborhood.json")).json();
  renderRingStats(d);

  const canvas = $("map"), tip = $("map-tip");
  const ctx = canvas.getContext("2d");
  const H = d.house;
  const cosLat = Math.cos(H.lat * Math.PI / 180);
  const MI_PER_DEG = 69.0469;

  // local mile coordinates centered on the house
  const toXY = (lon, lat) => [
    (lon - H.lon) * MI_PER_DEG * cosLat,
    (lat - H.lat) * MI_PER_DEG,
  ];

  const parcels = d.parcels.map(p => p.rings.map(r => r.map(([x, y]) => toXY(x, y))));
  const streets = d.streets.map(s => ({ ...s, xy: s.pts.map(([x, y]) => toXY(x, y)) }));
  const addrs = d.addresses.map(a => ({ ...a, xy: toXY(a.lon, a.lat) }));

  let scale = 0, cx = 0, cy = 0; // pixels per mile, view center in mile coords

  function resize() {
    const r = canvas.getBoundingClientRect();
    canvas.width = r.width * devicePixelRatio;
    canvas.height = r.height * devicePixelRatio;
    if (!scale) scale = canvas.height / 1.15; // start ~0.55 mi above/below
    draw();
  }

  const sx = (x) => canvas.width / 2 + (x - cx) * scale;
  const sy = (y) => canvas.height / 2 - (y - cy) * scale;

  function path(pts) {
    ctx.beginPath();
    ctx.moveTo(sx(pts[0][0]), sy(pts[0][1]));
    for (let i = 1; i < pts.length; i++) ctx.lineTo(sx(pts[i][0]), sy(pts[i][1]));
  }

  function draw() {
    const css = getComputedStyle(document.body);
    ctx.fillStyle = css.getPropertyValue("--map-bg");
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // parcels (the plat)
    ctx.strokeStyle = css.getPropertyValue("--parcel");
    ctx.lineWidth = Math.max(0.6, scale * 0.0006);
    for (const rings of parcels) for (const r of rings) { path(r); ctx.closePath(); ctx.stroke(); }

    // streets
    for (const s of streets) {
      const major = ["primary", "secondary", "tertiary"].includes(s.kind);
      ctx.strokeStyle = css.getPropertyValue(major ? "--street-major" : "--street");
      ctx.lineWidth = Math.max(major ? 2.5 : 1.2, scale * (major ? 0.004 : 0.002));
      ctx.lineCap = "round"; ctx.lineJoin = "round";
      if (s.kind === "track" || s.kind === "path") ctx.setLineDash([4, 6]); else ctx.setLineDash([]);
      path(s.xy); ctx.stroke();
    }
    ctx.setLineDash([]);

    // distance rings
    d.rings.forEach((r, i) => {
      ctx.strokeStyle = RING_COLORS[i] + "90";
      ctx.lineWidth = 1.5 * devicePixelRatio;
      ctx.setLineDash([8, 8]);
      ctx.beginPath();
      ctx.arc(sx(0), sy(0), r.miles * scale, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = RING_COLORS[i];
      ctx.font = `${12 * devicePixelRatio}px sans-serif`;
      ctx.fillText(`${r.miles} mi · ${r.addresses} addresses`,
        sx(0) + 6 * devicePixelRatio, sy(r.miles) - 6 * devicePixelRatio);
    });

    // address dots colored by ring
    for (const a of addrs) {
      const i = d.rings.findIndex(r => a.mi <= r.miles);
      ctx.fillStyle = i < 0 ? "#88888880" : RING_COLORS[i];
      ctx.beginPath();
      ctx.arc(sx(a.xy[0]), sy(a.xy[1]), Math.max(2, scale * 0.004), 0, Math.PI * 2);
      ctx.fill();
    }

    // the house
    const hx = sx(0), hy = sy(0), R = Math.max(6, scale * 0.009);
    ctx.fillStyle = "#1565c0";
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 2 * devicePixelRatio;
    ctx.beginPath(); ctx.arc(hx, hy, R, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    ctx.fillStyle = css.getPropertyValue("--ink");
    ctx.font = `bold ${13 * devicePixelRatio}px sans-serif`;
    ctx.fillText("7223", hx + R + 4, hy + 4);
  }

  /* pan / zoom / hover */
  let dragging = false, lx = 0, ly = 0;
  canvas.addEventListener("pointerdown", (e) => { dragging = true; lx = e.clientX; ly = e.clientY; canvas.setPointerCapture(e.pointerId); });
  canvas.addEventListener("pointerup", () => dragging = false);
  canvas.addEventListener("pointermove", (e) => {
    if (dragging) {
      cx -= (e.clientX - lx) * devicePixelRatio / scale;
      cy += (e.clientY - ly) * devicePixelRatio / scale;
      lx = e.clientX; ly = e.clientY; draw(); return;
    }
    // hover tooltip: nearest address within 14px
    const r = canvas.getBoundingClientRect();
    const mx = (e.clientX - r.left) * devicePixelRatio, my = (e.clientY - r.top) * devicePixelRatio;
    let best = null, bd = 14 * devicePixelRatio;
    for (const a of addrs) {
      const dx = sx(a.xy[0]) - mx, dy = sy(a.xy[1]) - my;
      const dist = Math.hypot(dx, dy);
      if (dist < bd) { bd = dist; best = a; }
    }
    if (best) {
      tip.classList.remove("hidden");
      tip.style.left = (e.clientX - r.left + 14) + "px";
      tip.style.top = (e.clientY - r.top - 10) + "px";
      tip.innerHTML = `<b>${best.num} ${esc(best.street)}</b><br>${best.mi.toFixed(2)} mi from the house`;
    } else tip.classList.add("hidden");
  });
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    const f = e.deltaY < 0 ? 1.15 : 1 / 1.15;
    scale = Math.min(canvas.height * 12, Math.max(canvas.height / 3, scale * f));
    draw();
  }, { passive: false });

  $("map-legend").innerHTML =
    `<b>Vineyard Mountain</b><br>` +
    d.rings.map((r, i) => `<span class="lg" style="background:${RING_COLORS[i]}"></span>≤ ${r.miles} mi — ${r.addresses} addr`).join("<br>") +
    `<br><span style="opacity:.7">${d.parcels.length} county taxlots · drag to pan · scroll to zoom</span>`;

  new ResizeObserver(resize).observe(canvas);
  resize();
  // redraw when theme flips at sunset/sunrise
  new MutationObserver(draw).observe(document.body, { attributes: true, attributeFilter: ["data-theme"] });
}

function renderRingStats(d) {
  $("ring-stats").innerHTML = `<div class="ringrow">` + d.rings.map((r, i) => {
    const streets = Object.entries(r.streets).slice(0, 6)
      .map(([s, n]) => `${esc(s.replace("Northwest", "NW").replace("Northeast", "NE"))} (${n})`).join(", ");
    return `<div class="ringbox">
      <div class="bignum" style="color:${RING_COLORS[i]}">${r.addresses}</div>
      <div>addresses within <b>${r.miles} mi</b></div>
      <div class="streets">${streets}${Object.keys(r.streets).length > 6 ? "…" : ""}</div>
    </div>`;
  }).join("") + `</div>
  <p class="statline" style="color:var(--ink-soft)">Sources: ${esc(d.sources.parcels)};
  ${esc(d.sources.addresses)}; ${esc(d.sources.streets)}.</p>`;
}

/* ---------------- boot ---------------- */

loadNow().then(() => {
  setInterval(tickClock, 1000);
  setInterval(loadNow, 10 * 60 * 1000);
});
loadWeather(); setInterval(loadWeather, 10 * 60 * 1000);
loadBurn(); setInterval(loadBurn, 30 * 60 * 1000);
loadFire(); setInterval(loadFire, 15 * 60 * 1000);
loadMap();
$("foot-updated").textContent = "page generated live " + new Date().toLocaleString("en-US", { timeZone: "America/Los_Angeles" });
