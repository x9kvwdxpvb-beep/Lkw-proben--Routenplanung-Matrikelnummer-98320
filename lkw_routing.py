#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LKW Proben-Routenplanung (Studentenprojekt)

Funktion:
- Liest config.json (Depot, Probenzeit, API-Endpunkte)
- Liest sites.json (bis zu 5 Baustellen-Adressen)
- Geocodiert Depot + Baustellen (Nominatim -> lat/lon)
- Holt Distanz- & Zeitmatrix (OSRM /table)
- Findet:
  (A) zeitlich schnellste Rundtour (Fahrzeit + Probenzeit)
  (B) kürzeste Rundtour (Kilometer)
- Erstellt:
  - Textreport (out_report.txt)
  - Interaktive Karte (out_map.html)

Hinweis:
- Max 5 Baustellen => wir können alle Permutationen exakt durchprobieren
  (5! = 120 Möglichkeiten).
"""
# -----------------------------------------------------------------------------
# Projekt: LKW-Proben-Routenplanung
# Autor: Malte Wirth
# Kontext: Studienprojekt (Routenoptimierung / Logistik)
#
# Ziel:
# Automatisierte Berechnung optimaler Rundtouren für mehrere Baustellen
# unter Berücksichtigung von Fahrzeit, Probenzeit und Distanz.
#
# Technische Grundlage:
# - Geocoding über Nominatim (OpenStreetMap)
# - Routing & Zeitmatrix über OSRM
# -----------------------------------------------------------------------------

from __future__ import annotations
import json
import logging
import time
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import folium


# -----------------------------
# Logging / Hilfsfunktionen
# -----------------------------

log = logging.getLogger("lkw-routing")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )


def fmt_minutes(m: float) -> str:
    """Minuten als 'Xh YYm' formatieren."""
    m_int = int(round(m))
    h = m_int // 60
    mm = m_int % 60
    return f"{h}h {mm:02d}m"


def read_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def sleep(seconds: float) -> None:
    """Wrapper, damit Throttling/Backoff im Code klar ist."""
    time.sleep(seconds)


# -----------------------------
# Datenmodelle
# -----------------------------

@dataclass(frozen=True)
class Point:
    lat: float
    lon: float


@dataclass(frozen=True)
class SiteInput:
    name: str
    address: str


@dataclass(frozen=True)
class Site:
    name: str
    address: str
    point: Point


@dataclass(frozen=True)
class TourResult:
    order: Tuple[int, ...]      # Indizes 1..n (Depot ist 0)
    total_km: float
    total_minutes: float
    drive_minutes: float
    service_minutes: float


# -----------------------------
# HTTP Helper (Retry)
# -----------------------------

@dataclass
class RetryConfig:
    timeout_seconds: int = 20
    max_retries: int = 3
    backoff_seconds: float = 0.8


def http_get_json(
    url: str,
    params: Optional[Dict[str, str]] = None,
    headers: Optional[Dict[str, str]] = None,
    retry: Optional[RetryConfig] = None
) -> Any:
    """HTTP GET mit Retries + Backoff."""
    retry = retry or RetryConfig()
    last_err: Optional[Exception] = None

    for attempt in range(1, retry.max_retries + 1):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=retry.timeout_seconds)

            # Rate limit (429) -> retry
            if r.status_code == 429:
                raise requests.HTTPError("429 Too Many Requests", response=r)

            r.raise_for_status()
            return r.json()

        except Exception as e:
            last_err = e
            wait = retry.backoff_seconds * attempt
            log.warning("HTTP Fehler %s (attempt %d/%d). Warte %.1fs ...",
                        type(e).__name__, attempt, retry.max_retries, wait)
            sleep(wait)

    assert last_err is not None
    raise last_err


# -----------------------------
# Geocoding (Nominatim)
# -----------------------------

@dataclass
class NominatimConfig:
    base_url: str
    user_agent: str
    min_seconds_between_requests: float = 1.0
    timeout_seconds: int = 20
    max_retries: int = 3
    backoff_seconds: float = 0.8


class GeocodeError(RuntimeError):
    pass


class Geocoder:
    """
    Nominatim Geocoder mit:
    - Rate limiting (>=1s zwischen Requests)
    - einfachem JSON-Cache (damit Tests stabil sind)
    """

    def __init__(self, cfg: NominatimConfig, cache_path: str = "geocode_cache.json"):
        self.cfg = cfg
        self.cache_path = Path(cache_path)
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._last_ts: float = 0.0
        self._load_cache()

    def _load_cache(self) -> None:
        if self.cache_path.exists():
            try:
                self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except Exception:
                log.warning("Cache unlesbar, starte leer.")
                self._cache = {}

    def _save_cache(self) -> None:
        self.cache_path.write_text(
            json.dumps(self._cache, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    def _throttle(self) -> None:
        dt = time.time() - self._last_ts
        wait = self.cfg.min_seconds_between_requests - dt
        if wait > 0:
            sleep(wait)

            pass  # dann normal weiter mit Nominatim
    def geocode (self, address: str, region_hint: str | None = None): 
    #1) Cache-Key
        key = (address.strip() + (" " + region_hint.strip() if region_hint else "")).lower()
        if key in self._cache:
       	    hit = self._cache[key]
            return Point(lat=float(hit["lat"]), lon=float(hit["lon"]))

    #2) Rate limit
        self._throttle()

    #3) Nominatim Request
        url = f"{self.cfg.base_url.rstrip('/')}/search"
        params = {
            "q": f"{address}, {region_hint}" if region_hint else address, 
            "format": "jsonv2",
            "limit": 1, 
            "addressdetails": 1,
    }
        headers = {
            "User-Agent": self.cfg.user_agent,
            "Accept-Language": "de, en;q=0.8",
    }

        data = http_get_json(
            url,
            params=params,
            headers=headers,
            retry=RetryConfig(
            timeout_seconds=self.cfg.timeout_seconds,
            max_retries=self.cfg.max_retries,
            backoff_seconds=self.cfg.backoff_seconds,
        ),
    )

        if not isinstance(data, list) or not data:
            raise GeocodeError(f"Adresse nicht gefunden: {address}")

        lat = float(data[0]["lat"])
        lon = float(data[0]["lon"])

        self._cache[key] = {"lat": lat, "lon": lon}
        self._save_cache()

        return Point(lat=lat, lon=lon)




# -----------------------------
# Routing (OSRM)
# -----------------------------

@dataclass
class OSRMConfig:
    base_url: str
    profile: str = "car"
    timeout_seconds: int = 20
    max_retries: int = 3
    backoff_seconds: float = 0.8


class RoutingError(RuntimeError):
    pass


def _coord_str(points: List[Point]) -> str:
    """OSRM erwartet 'lon,lat;lon,lat;...'"""
    return ";".join(f"{p.lon:.6f},{p.lat:.6f}" for p in points)


def osrm_table(cfg: OSRMConfig, points: List[Point]) -> Tuple[List[List[float]], List[List[float]]]:
    """
    /table liefert Distanz (m) & Dauer (s) -> wir geben km & Minuten zurück.
    """
    coords = _coord_str(points)
    url = f"{cfg.base_url.rstrip('/')}/table/v1/{cfg.profile}/{coords}"
    params = {"annotations": "distance,duration"}

    data = http_get_json(url, params=params, retry=RetryConfig(cfg.timeout_seconds, cfg.max_retries, cfg.backoff_seconds))

    if not isinstance(data, dict) or "distances" not in data or "durations" not in data:
        raise RoutingError("OSRM Table Antwort ungültig.")

    distances = data["distances"]
    durations = data["durations"]

    n = len(points)
    dist_km = [[0.0] * n for _ in range(n)]
    time_min = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            d = distances[i][j]
            t = durations[i][j]
            if d is None or t is None:
                raise RoutingError("OSRM konnte keine Verbindung berechnen.")
            dist_km[i][j] = float(d) / 1000.0
            time_min[i][j] = float(t) / 60.0

    return dist_km, time_min


def osrm_route_geometry(cfg: OSRMConfig, points_in_order: List[Point]) -> List[Tuple[float, float]]:
    """/route liefert GeoJSON-Linie -> Rückgabe als (lat, lon) Liste."""
    coords = _coord_str(points_in_order)
    url = f"{cfg.base_url.rstrip('/')}/route/v1/{cfg.profile}/{coords}"
    params = {"overview": "full", "geometries": "geojson"}

    data = http_get_json(url, params=params, retry=RetryConfig(cfg.timeout_seconds, cfg.max_retries, cfg.backoff_seconds))
    if not isinstance(data, dict) or "routes" not in data or not data["routes"]:
        raise RoutingError("OSRM Route Antwort ungültig.")

    geom = data["routes"][0].get("geometry")
    if not geom or geom.get("type") != "LineString":
        raise RoutingError("OSRM geometry fehlt.")

    coords = geom.get("coordinates", [])  # (lon, lat)
    return [(float(lat), float(lon)) for lon, lat in coords]


# -----------------------------
# TSP / Optimierung (Permutation)
# -----------------------------

def evaluate_tour(
    order: Tuple[int, ...],
    dist_km: List[List[float]],
    time_min: List[List[float]],
    probe_minutes_per_site: float
) -> TourResult:
    """Depot -> order -> Depot bewerten."""
    prev = 0
    km = 0.0
    drive = 0.0

    for idx in order:
        km += dist_km[prev][idx]
        drive += time_min[prev][idx]
        prev = idx

    km += dist_km[prev][0]
    drive += time_min[prev][0]

    service = probe_minutes_per_site * len(order)
    total = drive + service

    return TourResult(order=order, total_km=km, total_minutes=total, drive_minutes=drive, service_minutes=service)


def solve_exact(
    dist_km: List[List[float]],
    time_min: List[List[float]],
    n_sites: int,
    probe_minutes_per_site: float
) -> Tuple[TourResult, TourResult]:
    """
    Exakt durch Enumeration aller Permutationen.
    n_sites <= 5 => max 120 Routen.
    """
    if not (1 <= n_sites <= 5):
        raise ValueError("Es sind 1 bis 5 Baustellen erlaubt.")

    indices = list(range(1, n_sites + 1))
    best_fast: Optional[TourResult] = None
    best_short: Optional[TourResult] = None

    for order in permutations(indices):
        res = evaluate_tour(order, dist_km, time_min, probe_minutes_per_site)

        if best_fast is None or res.total_minutes < best_fast.total_minutes:
            best_fast = res
        if best_short is None or res.total_km < best_short.total_km:
            best_short = res

    assert best_fast is not None and best_short is not None
    return best_fast, best_short


# -----------------------------
# Reporting + Karte
# -----------------------------

def route_names(order: Tuple[int, ...], sites: List[Site]) -> str:
    seq = [sites[i - 1].name for i in order]
    return "Depot -> " + " -> ".join(seq) + " -> Depot"


def build_report(res: TourResult, title: str, sites: List[Site]) -> str:
    lines = []
    lines.append(title)
    lines.append("-" * len(title))
    lines.append(f"Route:      {route_names(res.order, sites)}")
    lines.append(f"Kilometer:  {res.total_km:.2f} km")
    lines.append(f"Fahrzeit:   {fmt_minutes(res.drive_minutes)} ({res.drive_minutes:.1f} min)")
    lines.append(f"Probenzeit: {fmt_minutes(res.service_minutes)} ({res.service_minutes:.1f} min)")
    lines.append(f"Gesamt:     {fmt_minutes(res.total_minutes)} ({res.total_minutes:.1f} min)")
    return "\n".join(lines)


def build_map(
    depot: Site,
    sites: List[Site],
    fast_line: List[Tuple[float, float]],
    short_line: List[Tuple[float, float]],
    out_html: str
) -> None:
    """Interaktive Karte als HTML."""
    m = folium.Map(location=[depot.point.lat, depot.point.lon], zoom_start=11, control_scale=True)

    fg_fast = folium.FeatureGroup(name="Schnellste Route (Zeit)", show=True)
    fg_short = folium.FeatureGroup(name="Kürzeste Route (km)", show=True)
    fg_points = folium.FeatureGroup(name="Punkte", show=True)

    # Depot
    folium.Marker(
        [depot.point.lat, depot.point.lon],
        popup=f"Depot<br>{depot.address}",
        tooltip="Depot",
        icon=folium.Icon(color="red", icon="home")
    ).add_to(fg_points)

    # Baustellen
    for i, s in enumerate(sites, start=1):
        folium.Marker(
            [s.point.lat, s.point.lon],
            popup=f"{s.name}<br>{s.address}",
            tooltip=f"{i}. {s.name}",
            icon=folium.Icon(color="blue", icon="wrench")
        ).add_to(fg_points)

    folium.PolyLine(fast_line, weight=5, opacity=0.85, color="blue").add_to(fg_fast)
    folium.PolyLine(short_line, weight=5, opacity=0.85, color="green").add_to(fg_short)

    fg_points.add_to(m)
    fg_fast.add_to(m)
    fg_short.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    m.get_root().html.add_child(folium.Element(
        "<div style='position:fixed; bottom:10px; left:10px; z-index:9999; "
        "background:white; padding:6px 8px; border:1px solid #aaa; font-size:12px;'>"
        "© OpenStreetMap contributors | Routing: OSRM</div>"
    ))

    m.save(out_html)


# -----------------------------
# Main
# -----------------------------

def load_sites(path: str = "sites.json") -> List[SiteInput]:
    data = read_json(path)
    raw = data.get("sites")
    if not isinstance(raw, list) or not raw:
        raise ValueError("sites.json braucht Feld 'sites' als Liste mit mindestens einem Eintrag.")

    sites: List[SiteInput] = []
    for i, s in enumerate(raw, start=1):
        if not isinstance(s, dict):
            raise ValueError(f"sites[{i}] ist kein Objekt.")
        name = str(s.get("name") or f"Baustelle {i}").strip()
        address = str(s.get("address") or "").strip()
        if not address:
            raise ValueError(f"sites[{i}] hat keine address.")
        sites.append(SiteInput(name=name, address=address))

    if not (1 <= len(sites) <= 5):
        raise ValueError("Es sind 1 bis 5 Baustellen erlaubt.")
    return sites


def validate_bbox(pt: Point, bbox: Optional[List[float]]) -> bool:
    """bbox = [min_lon, min_lat, max_lon, max_lat]"""
    if not bbox or len(bbox) != 4:
        return True
    min_lon, min_lat, max_lon, max_lat = bbox
    return (min_lat <= pt.lat <= max_lat) and (min_lon <= pt.lon <= max_lon)


def ordered_points(depot: Site, sites: List[Site], order: Tuple[int, ...]) -> List[Point]:
    seq = [depot.point]
    for idx in order:
        seq.append(sites[idx - 1].point)
    seq.append(depot.point)
    return seq

def prompt_sites_interactive(max_sites: int = 5) -> list[SiteInput]:
    """Fragt Baustellen interaktiv im Terminal ab und gibt SiteInput-Liste zurück."""
    while True:
        raw = input(f"Wie viele Baustellen (1-{max_sites})? ").strip()
        try:
            n = int(raw)
            if 1 <= n <= max_sites:
                break
        except ValueError:
            pass
        print(f"Bitte eine Zahl zwischen 1 und {max_sites} eingeben.")

    sites: list[SiteInput] = []
    for i in range(1, n + 1):
        name = input(f"Name Baustelle {i} (Enter = 'Baustelle {i}'): ").strip()
        if not name:
            name = f"Baustelle {i}"

        while True:
            addr = input(f"Adresse Baustelle {i}: ").strip()
            if addr:
                break
            print("Adresse darf nicht leer sein.")

        sites.append(SiteInput(name=name, address=addr))

    return sites

def main() -> int:
    setup_logging()

    cfg = read_json("config.json")

    sites_in = prompt_sites_interactive(max_sites=5)

    depot_addr = cfg["depot_address"]
    probe_minutes = float(cfg["probe_minutes_per_site"])
    region_hint = cfg.get("region_hint")

    bbox = cfg.get("validation", {}).get("bbox_hint")
    reject_outside = bool(cfg.get("validation", {}).get("reject_outside_bbox", False))

    geo_cfg = NominatimConfig(**cfg["nominatim"])
    osrm_cfg = OSRMConfig(**cfg["osrm"])

    geocoder = Geocoder(geo_cfg)

    # Depot geocoden
    depot_pt = geocoder.geocode(depot_addr, region_hint=region_hint)
    if not validate_bbox(depot_pt, bbox):
        msg = "Depot liegt außerhalb der Region-BBox (Hinweis)."
        if reject_outside:
            log.error(msg)
            return 2
        log.warning(msg)

    depot = Site(name="Depot", address=depot_addr, point=depot_pt)

    # Baustellen geocoden
    sites: List[Site] = []
    for s in sites_in:
        pt = geocoder.geocode(s.address, region_hint=region_hint)
        if not validate_bbox(pt, bbox):
            msg = f"Baustelle '{s.name}' liegt evtl. außerhalb der Region."
            if reject_outside:
                log.error(msg)
                return 3
            log.warning(msg)

        sites.append(Site(name=s.name, address=s.address, point=pt))

    # Distanz/Zeitmatrix (Depot + Sites)
    points = [depot.point] + [s.point for s in sites]
    dist_km, time_min = osrm_table(osrm_cfg, points)

    # Optimierung
    fastest, shortest = solve_exact(dist_km, time_min, n_sites=len(sites), probe_minutes_per_site=probe_minutes)

    # Linien für Karte
    fast_line = osrm_route_geometry(osrm_cfg, ordered_points(depot, sites, fastest.order))
    short_line = osrm_route_geometry(osrm_cfg, ordered_points(depot, sites, shortest.order))

    # Outputs
    report_path = "out_report.txt"
    map_path = "out_map.html"

    report = []
    report.append(build_report(fastest, "Schnellste Route (Zeit)", sites))
    report.append("")
    report.append(build_report(shortest, "Kürzeste Route (Kilometer)", sites))
    Path(report_path).write_text("\n".join(report), encoding="utf-8")

    build_map(depot, sites, fast_line, short_line, map_path)

    print(f"Report geschrieben: {report_path}")
    print(f"Karte geschrieben:  {map_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

