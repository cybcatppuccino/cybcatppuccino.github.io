#!/usr/bin/env python3
"""Download OSM railway snapshots and selected official reference files.

Standard-library only. Run from the project root:
    python scripts/refresh_data.py --all
"""
from __future__ import annotations
import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGIONS = {
    "guangzhou": (112.75, 22.48, 114.15, 23.78),
    "foshan": (112.50, 22.50, 113.50, 23.45),
    "dongguan": (113.35, 22.58, 114.28, 23.25),
    "shenzhen": (113.68, 22.32, 114.72, 22.92),
    "hongkong": (113.78, 22.08, 114.48, 22.63),
    "macau": (113.45, 22.03, 113.66, 22.25),
}
OVERPASS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"]
OFFICIAL = {
    "data/reference/hk/mtr_lines_and_stations.csv": "https://opendata.mtr.com.hk/data/mtr_lines_and_stations.csv",
    "data/reference/hk/light_rail_routes_and_stops.csv": "https://opendata.mtr.com.hk/data/light_rail_routes_and_stops.csv",
    "data/reference/hk/STOP_TRAM.xml": "https://static.data.gov.hk/td/routes-fares-xml/STOP_TRAM.xml",
}
USER_AGENT = "GBA-Rail-Map-Research/1.0 (local educational project)"

def query(bbox: tuple[float,float,float,float]) -> str:
    w,s,e,n = bbox
    return f'''[out:json][timeout:180][bbox:{s},{w},{n},{e}];
(
  way["railway"~"^(rail|subway|light_rail|tram|monorail|narrow_gauge|construction)$"];
  relation["type"="route"]["route"~"^(subway|light_rail|tram)$"];
  relation["type"="route"]["route"="train"]["network"~"广东城际|Guangdong Intercity|珠三角城际",i];
  node["railway"~"^(station|halt|tram_stop)$"];
);
out body geom;'''

def request(url: str, data: bytes | None = None, timeout: int = 240) -> bytes:
    req = urllib.request.Request(url, data=data, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as res:
        return res.read()

def download_region(region: str) -> None:
    payload = urllib.parse.urlencode({"data": query(REGIONS[region])}).encode()
    error = None
    for endpoint in OVERPASS:
        try:
            print(f"[{region}] querying {endpoint}")
            raw = request(endpoint, payload)
            parsed = json.loads(raw)
            out = ROOT / "data" / "osm" / f"{region}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(parsed, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
            print(f"[{region}] saved {len(parsed.get('elements', []))} elements -> {out.relative_to(ROOT)}")
            return
        except Exception as exc:  # noqa: BLE001
            error = exc
            print(f"[{region}] failed: {exc}")
    raise RuntimeError(f"all Overpass endpoints failed for {region}: {error}")

def download_official() -> None:
    for rel, url in OFFICIAL.items():
        out = ROOT / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {url}")
        out.write_bytes(request(url, timeout=120))
        print(f"saved -> {out.relative_to(ROOT)}")

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("regions", nargs="*", choices=sorted(REGIONS))
    p.add_argument("--all", action="store_true", help="download all six regions and official references")
    p.add_argument("--official", action="store_true", help="download official reference files")
    args = p.parse_args()
    regions = list(REGIONS) if args.all else args.regions
    if not regions and not args.official:
        p.error("specify one or more regions, --official, or --all")
    for i, region in enumerate(regions):
        download_region(region)
        if i < len(regions)-1:
            time.sleep(8)  # be considerate to public Overpass instances
    if args.all or args.official:
        download_official()

if __name__ == "__main__":
    main()
