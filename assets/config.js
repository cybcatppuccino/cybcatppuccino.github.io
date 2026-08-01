window.GBA_RAIL_CONFIG = {
  version: "2026.08.01-GBARailv2.1.4-overview-stations",
  defaultBasemap: "positron",
  bundledGeoJSONPath: "data/rail_snapshot.geojson",
  lazyDataManifest: "data/chunks/manifest.json",
  basemaps: {
    positron: "https://tiles.openfreemap.org/styles/positron",
    liberty: "https://tiles.openfreemap.org/styles/liberty"
  },
  overpassEndpoints: [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter"
  ],
  initialView: { center: [113.2644, 23.1291], zoom: 11.2, pitch: 0, bearing: 0 },
  regions: [
    { id: "guangzhou", name: "广州", bbox: [112.75, 22.48, 114.15, 23.78], center: [113.2644, 23.1291], zoom: 9.2 },
    { id: "foshan", name: "佛山", bbox: [112.50, 22.50, 113.50, 23.45], center: [113.1214, 23.0215], zoom: 9.4 },
    { id: "dongguan", name: "东莞", bbox: [113.35, 22.58, 114.28, 23.25], center: [113.7518, 23.0207], zoom: 9.4 },
    { id: "shenzhen", name: "深圳", bbox: [113.68, 22.32, 114.72, 22.92], center: [114.0579, 22.5431], zoom: 9.5 },
    { id: "hongkong", name: "香港", bbox: [113.78, 22.08, 114.48, 22.63], center: [114.1694, 22.3193], zoom: 9.7 },
    { id: "macau", name: "澳门", bbox: [113.45, 22.03, 113.66, 22.25], center: [113.5439, 22.1987], zoom: 11.2 }
  ],
  localSnapshotPath: "data/osm/{id}.json",
  officialCatalogs: {
    mtr: "data/reference/hk/mtr_lines_and_stations.csv",
    lightRail: "data/reference/hk/light_rail_routes_and_stops.csv"
  }
};
