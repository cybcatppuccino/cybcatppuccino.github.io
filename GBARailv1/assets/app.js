/* Bay Area Rail Map v2 — static client, no build step required. */
(() => {
  "use strict";

  const CFG = window.GBA_RAIL_CONFIG;
  const EMPTY = { type: "FeatureCollection", features: [] };
  const featureStore = new Map();
  const regionStates = new Map(CFG.regions.map(r => [r.id, "idle"]));
  const searchEntries = [];
  const railLayerIds = new Set();
  const railLineDefaults = new Map();
  const stationPaintDefaults = new Map();
  const baseLayerSnapshots = new Map();
  const lineStyleWidthFactors = new Map();
  const lineStyleOpacityFactors = new Map();
  const tramLineColorHints = new Map();

  let map;
  let currentStyle = CFG.defaultBasemap || "positron";
  let officialCatalog = { mtr: [], lightRail: [] };
  let toastTimer;
  let dbPromise;
  let loadingAll = false;
  let baseApplyFrame = 0;
  let bundledGeoJSONPromise = null;
  let searchRebuildHandle = 0;
  let cachedRailGeoJSON = EMPTY;
  let cachedStationGeoJSON = EMPTY;

  const settings = {
    lineWidth: 1,
    lineOpacity: 1,
    stationSize: 1,
    stationStyle: "hollow",
    metroLineStyle: "slim",
    tramLineStyle: "thin",
    nationalLineStyle: "railway",
    intercityLineStyle: "double",
    constructionLineStyle: "dots",
    constructionDashLength: 1,
    constructionDashGap: 1,
    constructionLineCap: "round",
    labelDetail: "standard",
    baseOpacity: 0.82,
    baseGrayscale: 0.35,
    baseContrast: 1,
    baseSaturation: 1,
    baseDetail: "standard"
  };

  const $ = sel => document.querySelector(sel);
  const $$ = sel => [...document.querySelectorAll(sel)];
  const deepClone = value => value == null ? value : JSON.parse(JSON.stringify(value));

  function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>'"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[c]));
  }

  function showToast(message, duration = 3200) {
    const el = $("#statusToast");
    clearTimeout(toastTimer);
    el.textContent = message;
    el.hidden = false;
    toastTimer = setTimeout(() => { el.hidden = true; }, duration);
  }

  function normalizeText(value) {
    return String(value || "").toLowerCase().replace(/[\s·・‧—–_()（）\[\]【】'’.-]/g, "");
  }

  function hashNumber(text) {
    let h = 2166136261;
    for (const ch of String(text || "rail")) { h ^= ch.charCodeAt(0); h = Math.imul(h, 16777619); }
    return h >>> 0;
  }

  const fallbackPalette = ["#d94c4c", "#e7822f", "#d7a51e", "#52a458", "#148e84", "#3387c8", "#5969c9", "#8c5abe", "#c94f8c", "#9a5b3f"];
  function safeColor(input, seed) {
    if (typeof input === "string") {
      const c = input.trim();
      if (/^#[0-9a-f]{3,8}$/i.test(c) || /^(rgb|hsl)a?\(/i.test(c)) return c;
      const named = { red:"#d94c4c", blue:"#3387c8", green:"#52a458", orange:"#e7822f", yellow:"#d7a51e", purple:"#8c5abe", pink:"#c94f8c", brown:"#9a5b3f", cyan:"#00a6b2", teal:"#148e84" };
      if (named[c.toLowerCase()]) return named[c.toLowerCase()];
    }
    return fallbackPalette[hashNumber(seed) % fallbackPalette.length];
  }

  const UNKNOWN_CONSTRUCTION_COLOR = "#8a9298";
  const NATIONAL_RAIL_COLOR = "#E54848";
  const GENERIC_DATA_COLORS = new Set(["#52636f", "#00a6b2", "#8d5ac7", "#5e8fc6"]);
  const tramServiceColorCorrections = [
    [/深圳(?:龙华)?有轨电车.*(?:1号线|清湖[^\n]*下围|下围[^\n]*清湖)|深圳有轨电车[^\n]*\s1(?:\s|$)/i, "#D7A51E"],
    [/深圳(?:龙华)?有轨电车.*(?:2号线|清湖[^\n]*新澜|新澜[^\n]*清湖)|深圳有轨电车[^\n]*\s2(?:\s|$)/i, "#FFCC99"]
  ];

  const lineColorRules = [
    [/海珠有轨(?:电车)?1号线|THZ1/i, "#43B02A"],
    [/黄埔有轨(?:电车)?1号线|THP1/i, "#D42D1B"],
    [/黄埔有轨(?:电车)?2号线|THP2/i, "#E55C9A"],
    [/南海有轨(?:电车)?1号线|TNH1/i, "#5EB3E4"],
    [/高明(?:区现代)?有轨电车|TGM1/i, "#4CA585"],
    [/澳门轻轨.*氹仔线|凼仔线|Taipa/i, "#96C93C"],
    [/澳门轻轨.*石排湾线|石排灣線|Seac Pai Van/i, "#8966C3"],
    [/澳门轻轨.*横琴线|橫琴線|Hengqin/i, "#B82A3C"],
    [/珠江新城旅客自动输送系统|地铁\s*APM线|\bAPM\b/i, "#00B5E2"],

    [/广清城际/i, "#2F80C3"],
    [/广珠城际/i, "#1687A7"],
    [/珠机城际/i, "#4B78B7"],
    [/广州东环城际|新白广城际/i, "#D85757"],
    [/琶莲城际/i, "#8C62B5"],
    [/广肇城际|佛肇城际/i, "#C97932"],
    [/广惠城际|莞惠城际|佛莞城际/i, "#2C956F"],
    [/穗深城际/i, "#7456A6"],
    [/广佛(?:南环|西环|环线)城际/i, "#9A78C7"],
    [/深惠城际/i, "#2B8C8A"],
    [/深大城际/i, "#2A78B8"],
    [/南沙至珠海|南珠高速线/i, "#397F9D"],

    [/广州地铁.*(?:^|\D)1号线|广州地铁集团.*(?:^|\D)1号线/i, "#F3D03E"],
    [/广州地铁.*(?:^|\D)2号线|广州地铁集团.*(?:^|\D)2号线/i, "#00629B"],
    [/广州地铁.*(?:^|\D)3号线|广州地铁集团.*(?:^|\D)3号线/i, "#ECA154"],
    [/广州地铁.*(?:^|\D)4号线|广州地铁集团.*(?:^|\D)4号线/i, "#00843D"],
    [/广州地铁.*(?:^|\D)5号线|广州地铁集团.*(?:^|\D)5号线/i, "#C5003E"],
    [/广州地铁.*(?:^|\D)6号线|广州地铁集团.*(?:^|\D)6号线/i, "#80225F"],
    [/广州地铁.*(?:^|\D)7号线|广州地铁集团.*(?:^|\D)7号线/i, "#97D700"],
    [/广州地铁.*(?:^|\D)8号线|广州地铁集团.*(?:^|\D)8号线/i, "#008C95"],
    [/广州地铁.*(?:^|\D)9号线|广州地铁集团.*(?:^|\D)9号线/i, "#71CC98"],
    [/广州地铁.*(?:^|\D)10号线|广州地铁集团.*(?:^|\D)10号线/i, "#7389B2"],
    [/广州地铁.*(?:^|\D)11号线|广州地铁集团.*(?:^|\D)11号线/i, "#FAC525"],
    [/广州地铁.*(?:^|\D)12号线|广州地铁集团.*(?:^|\D)12号线/i, "#435428"],
    [/广州地铁.*(?:^|\D)13号线|广州地铁集团.*(?:^|\D)13号线/i, "#8E8C13"],
    [/广州地铁.*(?:^|\D)14号线|广州地铁集团.*(?:^|\D)14号线/i, "#81312F"],
    [/广州地铁.*(?:^|\D)16号线/i, "#9E652E"],
    [/广州地铁.*(?:^|\D)18号线|广州地铁集团.*(?:^|\D)18号线/i, "#0047BA"],
    [/广州地铁.*(?:^|\D)21号线|广州地铁集团.*(?:^|\D)21号线/i, "#201747"],
    [/广州地铁.*(?:^|\D)22号线|广州地铁集团.*(?:^|\D)22号线|芳白城际.*22号线/i, "#CD5228"],
    [/广州地铁.*APM线|广州地铁集团.*APM/i, "#00B5E2"],

    [/深圳地铁.*(?:^|\D)13号线/i, "#DE7C00"],
    [/深圳地铁.*(?:^|\D)15号线/i, "#78B943"],
    [/深圳地铁.*(?:^|\D)17号线/i, "#E9A3B8"],
    [/深圳地铁.*(?:^|\D)19号线/i, "#A83B68"],
    [/深圳地铁.*(?:^|\D)20号线/i, "#88DBDF"],
    [/深圳地铁.*(?:^|\D)22号线/i, "#F3C300"],
    [/深圳地铁.*(?:^|\D)25号线/i, "#EFA15B"],
    [/深圳地铁.*(?:^|\D)29号线/i, "#8BCB8B"],
    [/深圳地铁.*(?:^|\D)32号线/i, "#65483C"],

    [/佛山地铁.*(?:^|\D)2号线|佛山市轨道交通.*(?:^|\D)2号线/i, "#C10230"],
    [/佛山地铁.*(?:^|\D)3号线|佛山市轨道交通.*(?:^|\D)3号线/i, "#002F87"],
    [/东莞轨道交通.*(?:^|\D)1号线/i, "#3190CB"],
    [/东莞轨道交通.*(?:^|\D)2号线/i, "#ED1C24"]
  ];

  const numberedCityLineColors = {
    guangzhou: {
      1:"#F3D03E", 2:"#00629B", 3:"#ECA154", 4:"#00843D", 5:"#C5003E", 6:"#80225F", 7:"#97D700", 8:"#008C95", 9:"#71CC98",
      10:"#7389B2", 11:"#FAC525", 12:"#435428", 13:"#8E8C13", 14:"#81312F", 16:"#9E652E", 18:"#0047BA", 21:"#201747", 22:"#CD5228"
    },
    shenzhen: { 13:"#DE7C00", 15:"#78B943", 17:"#E9A3B8", 19:"#A83B68", 20:"#88DBDF", 22:"#F3C300", 25:"#EFA15B", 29:"#8BCB8B", 32:"#65483C" },
    foshan: { 2:"#C10230", 3:"#002F87" },
    dongguan: { 1:"#3190CB", 2:"#ED1C24" }
  };

  function numberedCityLineColor(text) {
    let palette = null;
    if (/广州地铁|广州地铁集团/.test(text)) palette = numberedCityLineColors.guangzhou;
    else if (/深圳地铁|深圳市地铁集团|深圳市城市轨道/.test(text)) palette = numberedCityLineColors.shenzhen;
    else if (/佛山地铁|佛山市轨道交通/.test(text)) palette = numberedCityLineColors.foshan;
    else if (/东莞轨道交通|东莞市轨道交通/.test(text)) palette = numberedCityLineColors.dongguan;
    if (!palette) return null;
    for (const match of text.matchAll(/(?:^|[^0-9])(\d{1,2})号线/g)) {
      const color = palette[Number(match[1])];
      if (color) return color;
    }
    return null;
  }

  function featureColorText(properties = {}) {
    return [properties.name, properties.nameEn, properties.ref, properties.operator, properties.network, properties.mode].filter(Boolean).join(" ");
  }

  function isDirectLineColor(value) {
    const color = typeof value === "string" ? value.trim() : "";
    return color && (/^#[0-9a-f]{3,8}$/i.test(color) || /^(rgb|hsl)a?\(/i.test(color)) ? color : "";
  }

  function tramLineIdentityKeys(properties = {}) {
    const keys = [];
    const ref = normalizeText(properties.ref);
    if (ref && (/[a-z]/i.test(ref) || ref.length >= 3)) keys.push(`ref:${ref}`);
    let name = String(properties.name || "").split(/[：:（(]/, 1)[0].replace(/^(?:地铁|輕鐵|轻轨|有轨)\s*/i, "");
    name = normalizeText(name);
    const genericNetworkName = /^(?:未命名|轻铁|輕鐵|香港电车|香港電車|深圳有轨电车|深圳龍華有軌電車|深圳龙华有轨电车|华为松山湖有轨电车)$/;
    if (name && !genericNetworkName.test(name)) keys.push(`name:${name}`);
    return keys;
  }

  function rebuildTramLineColorHints(features = []) {
    tramLineColorHints.clear();
    const scores = new Map();
    for (const feature of features) {
      const properties = feature?.properties || {};
      if (properties.featureType !== "service" || !["tram", "light_rail"].includes(properties.railClass)) continue;
      const color = isDirectLineColor(properties.color);
      if (!color || GENERIC_DATA_COLORS.has(color.toLowerCase())) continue;
      const score = 1 + (properties.operator ? 2 : 0) + (properties.network ? 1 : 0);
      for (const key of tramLineIdentityKeys(properties)) {
        if (score > (scores.get(key) || -1)) {
          scores.set(key, score);
          tramLineColorHints.set(key, color);
        }
      }
    }
  }

  function hintedTramLineColor(properties = {}) {
    for (const key of tramLineIdentityKeys(properties)) {
      const color = tramLineColorHints.get(key);
      if (color) return color;
    }
    return "";
  }

  function resolvedLineColor(properties = {}) {
    const text = featureColorText(properties);
    const existing = isDirectLineColor(properties.color);
    const validExisting = existing && safeColor(existing, text);
    const existingIsGeneric = existing && GENERIC_DATA_COLORS.has(existing.toLowerCase());

    if (properties.railClass === "national" || properties.railClass === "highspeed") return NATIONAL_RAIL_COLOR;
    const isTramLike = properties.railClass === "tram" || properties.railClass === "light_rail";

    if (isTramLike) {
      // 对少数同一路线双向数据颜色不一致的记录先做配对修复；其余服务线直接使用快照自带线路色。
      if (properties.featureType === "service") {
        for (const [pattern, color] of tramServiceColorCorrections) if (pattern.test(text)) return color;
        if (validExisting && !existingIsGeneric) return validExisting;
      }
      const hinted = hintedTramLineColor(properties);
      if (hinted) return hinted;
      for (const [pattern, color] of lineColorRules) if (pattern.test(text)) return color;
      if (validExisting && !existingIsGeneric) return validExisting;
      return "#74818A";
    }

    const numberedColor = numberedCityLineColor(text);
    if (numberedColor) return numberedColor;
    for (const [pattern, color] of lineColorRules) if (pattern.test(text)) return color;
    if (properties.featureType === "service" && validExisting && !existingIsGeneric) return validExisting;

    if (properties.railClass === "construction" || properties.status === "construction") {
      return validExisting && !existingIsGeneric ? validExisting : UNKNOWN_CONSTRUCTION_COLOR;
    }
    if (validExisting && !existingIsGeneric) return validExisting;
    if (properties.railClass === "intercity") return "#147D92";
    return existing || "#52636F";
  }

  function normalizeFeatureAppearance(feature) {
    const properties = feature?.properties;
    if (!properties || !["service", "infrastructure"].includes(properties.featureType)) return feature;
    properties.color = resolvedLineColor(properties);
    return feature;
  }

  function pickName(tags = {}) {
    return tags["name:zh-Hans"] || tags["name:zh"] || tags["name:zh-Hant"] || tags.name || tags["name:en"] || tags.ref || "未命名";
  }

  function isConstruction(tags = {}) {
    return tags.railway === "construction" || tags.construction || tags.proposed || tags.status === "construction";
  }

  function classify(tags = {}, relationRoute = "") {
    const railway = tags.railway || "";
    const route = relationRoute || tags.route || "";
    const text = `${pickName(tags)} ${tags.network || ""} ${tags.operator || ""} ${tags.ref || ""}`;
    if (isConstruction(tags)) return "construction";
    if (route === "tram" || railway === "tram" || railway === "tram_stop") return "tram";
    if (route === "light_rail" || railway === "light_rail" || railway === "monorail" || tags.station === "light_rail" || tags.light_rail === "yes") return "light_rail";
    if (route === "subway" || railway === "subway" || tags.station === "subway") return "metro";
    if (route === "train" && /城际|intercity|广东城际|珠三角城际/i.test(text)) return "intercity";
    if (/城际|intercity|广东城际|珠三角城际/i.test(text)) return "intercity";
    if (tags.highspeed === "yes" || /高速铁路|高铁|high.?speed/i.test(text) || Number(tags.maxspeed) >= 200) return "highspeed";
    if (railway === "rail" && (tags.usage === "main" || (!tags.usage && !tags.service))) return "national";
    if (["station", "halt", "tram_stop"].includes(railway)) return "station";
    return "minor";
  }

  function makeOverpassQuery(region) {
    const [w, s, e, n] = region.bbox;
    return `[out:json][timeout:180][bbox:${s},${w},${n},${e}];\n(\n` +
      `  way["railway"~"^(rail|subway|light_rail|tram|monorail|narrow_gauge|construction)$"];\n` +
      `  relation["type"="route"]["route"~"^(subway|light_rail|tram)$"];\n` +
      `  relation["type"="route"]["route"="train"]["network"~"广东城际|Guangdong Intercity|珠三角城际",i];\n` +
      `  node["railway"~"^(station|halt|tram_stop)$"];\n` +
      `);\nout body geom;`;
  }

  function lineGeometry(geometry) {
    if (!Array.isArray(geometry)) return null;
    const coords = geometry.filter(p => Number.isFinite(p.lon) && Number.isFinite(p.lat)).map(p => [p.lon, p.lat]);
    return coords.length >= 2 ? coords : null;
  }

  function parseOverpass(json, regionId) {
    const features = [];
    for (const el of json.elements || []) {
      const tags = el.tags || {};
      if (el.type === "way") {
        const coords = lineGeometry(el.geometry);
        if (!coords) continue;
        const cls = classify(tags);
        features.push({
          type: "Feature", id: `w${el.id}`,
          geometry: { type: "LineString", coordinates: coords },
          properties: {
            key: `way:${el.id}`, osmType: "way", osmId: el.id, sourceRegion: regionId,
            featureType: "infrastructure", railClass: cls, mode: tags.railway || "rail",
            name: pickName(tags), nameEn: tags["name:en"] || "", ref: tags.ref || "",
            operator: tags.operator || "", network: tags.network || "", usage: tags.usage || "",
            service: tags.service || "", highspeed: tags.highspeed || "", tunnel: tags.tunnel || "",
            bridge: tags.bridge || "", status: isConstruction(tags) ? "construction" : "operational",
            color: cls === "intercity" ? "#00a6b2" : cls === "highspeed" ? "#e54848" : cls === "tram" ? "#8d5ac7" : "#52636f"
          }
        });
      } else if (el.type === "relation") {
        const route = tags.route || "";
        const lines = (el.members || []).filter(m => m.type === "way" && Array.isArray(m.geometry)).map(m => lineGeometry(m.geometry)).filter(Boolean);
        if (!lines.length) continue;
        const cls = classify(tags, route);
        const name = pickName(tags);
        features.push({
          type: "Feature", id: `r${el.id}`,
          geometry: { type: "MultiLineString", coordinates: lines },
          properties: {
            key: `relation:${el.id}`, osmType: "relation", osmId: el.id, sourceRegion: regionId,
            featureType: "service", railClass: cls, mode: route, name,
            nameEn: tags["name:en"] || "", ref: tags.ref || "", operator: tags.operator || "", network: tags.network || "",
            status: isConstruction(tags) ? "construction" : "operational",
            color: safeColor(tags.colour || tags.color, `${name}-${tags.ref || el.id}`)
          }
        });
      } else if (el.type === "node" && Number.isFinite(el.lon) && Number.isFinite(el.lat)) {
        const name = pickName(tags);
        const cls = classify(tags);
        features.push({
          type: "Feature", id: `n${el.id}`,
          geometry: { type: "Point", coordinates: [el.lon, el.lat] },
          properties: {
            key: `node:${el.id}`, osmType: "node", osmId: el.id, sourceRegion: regionId,
            featureType: "station", railClass: cls, mode: tags.station || tags.railway || "station",
            name, nameEn: tags["name:en"] || "", ref: tags.ref || tags["ref:crs"] || "",
            operator: tags.operator || "", network: tags.network || "", color: "#126d75",
            status: isConstruction(tags) ? "construction" : "operational"
          }
        });
      }
    }
    return features;
  }

  function geometryHasCoordinates(geometry) {
    let valid = false;
    const walk = coords => {
      if (valid) return;
      if (Array.isArray(coords) && coords.length >= 2 && Number.isFinite(coords[0]) && Number.isFinite(coords[1])) valid = true;
      else if (Array.isArray(coords)) coords.forEach(walk);
    };
    walk(geometry?.coordinates);
    return valid;
  }

  function rebuildGeoJSONCaches() {
    const features = [...featureStore.values()];
    cachedRailGeoJSON = { type: "FeatureCollection", features };
    cachedStationGeoJSON = { type: "FeatureCollection", features: features.filter(f => f.properties.featureType === "station") };
  }

  function scheduleSearchRebuild() {
    if (searchRebuildHandle) return;
    const run = () => { searchRebuildHandle = 0; rebuildSearch(); };
    if ("requestIdleCallback" in window) searchRebuildHandle = requestIdleCallback(run, { timeout: 1200 });
    else searchRebuildHandle = setTimeout(run, 0);
  }

  function mergeFeatures(features, { trusted = false } = {}) {
    const incoming = features || [];
    rebuildTramLineColorHints(incoming);
    if (trusted && featureStore.size === 0) {
      // rail_snapshot.geojson 的 key 已唯一：首载走单次批量路径，避免几何 JSON.stringify 比较和重复数组复制。
      const prepared = [];
      const stations = [];
      for (const f of incoming) {
        if (!f?.properties?.key) continue;
        normalizeFeatureAppearance(f);
        featureStore.set(f.properties.key, f);
        prepared.push(f);
        if (f.properties.featureType === "station") stations.push(f);
      }
      cachedRailGeoJSON = { type: "FeatureCollection", features: prepared };
      cachedStationGeoJSON = { type: "FeatureCollection", features: stations };
    } else {
      for (const f of incoming) {
        if (!f?.properties?.key || (!trusted && !geometryHasCoordinates(f.geometry))) continue;
        normalizeFeatureAppearance(f);
        const key = f.properties.key;
        const old = featureStore.get(key);
        if (!old || JSON.stringify(f.geometry).length > JSON.stringify(old.geometry).length) featureStore.set(key, f);
      }
      rebuildGeoJSONCaches();
    }
    enrichStationsFromOfficialCatalog();
    updateMapSources();
    updateSummary();
    scheduleSearchRebuild();
  }

  function currentGeoJSON() { return cachedRailGeoJSON; }
  function currentStationsGeoJSON() { return cachedStationGeoJSON; }

  function updateMapSources() {
    const rail = map?.getSource("rail-data");
    if (rail) rail.setData(currentGeoJSON());
    const rawStations = map?.getSource("station-data");
    if (rawStations) rawStations.setData(currentStationsGeoJSON());
    if (map) requestAnimationFrame(applyStationVisibility);
  }

  function updateSummary() {
    let routes = 0, tracks = 0, stations = 0;
    for (const f of featureStore.values()) {
      if (f.properties.featureType === "service") routes++;
      else if (f.properties.featureType === "station") stations++;
      else tracks++;
    }
    $("#dataSummary").textContent = featureStore.size ? `${routes} 条线路 · ${tracks} 段轨道 · ${stations} 个车站` : "等待本地数据或永久缓存";
  }

  function styleVisibility(id, visible) {
    if (map?.getLayer(id)) map.setLayoutProperty(id, "visibility", visible ? "visible" : "none");
  }

  const layerGroups = {
    service: ["service-halo", "service-lines", "service-labels", "metro-infra"],
    intercity: ["intercity-halo", "intercity-lines", "intercity-center", "intercity-labels"],
    national: ["national-halo", "national-lines", "national-detail", "highspeed-halo", "highspeed-lines", "highspeed-detail"],
    tram: ["tram-infra-halo", "tram-infra", "tram-service-halo", "tram-service-lines", "tram-labels"],
    construction: ["construction-halo", "construction-lines"],
    minor: ["minor-lines"]
  };

  const stationLayerIds = ["stations-halo", "stations", "station-labels"];
  const railLayerOrder = [
    "minor-lines", "national-halo", "national-lines", "national-detail", "highspeed-halo", "highspeed-lines", "highspeed-detail",
    "intercity-halo", "intercity-lines", "intercity-center", "metro-infra", "tram-infra-halo", "tram-infra",
    "service-halo", "service-lines", "tram-service-halo", "tram-service-lines", "construction-halo", "construction-lines",
    "service-labels", "intercity-labels", "tram-labels", "stations-halo", "stations", "station-labels"
  ];

  function addRailLayer(layer) {
    railLayerIds.add(layer.id);
    if (layer.type === "line") railLineDefaults.set(layer.id, {
      width: deepClone(layer.paint?.["line-width"] ?? 1),
      opacity: deepClone(layer.paint?.["line-opacity"] ?? 1)
    });
    if (layer.type === "circle" && layer.id.startsWith("stations")) stationPaintDefaults.set(layer.id, {
      radius: deepClone(layer.paint?.["circle-radius"] ?? 1),
      strokeWidth: deepClone(layer.paint?.["circle-stroke-width"] ?? 0)
    });
    map.addLayer(layer);
  }

  function addRailLayers() {
    if (!map || map.getSource("rail-data")) return;
    map.addSource("rail-data", { type: "geojson", data: currentGeoJSON(), promoteId: "key" });
    map.addSource("station-data", { type: "geojson", data: currentStationsGeoJSON(), promoteId: "key" });

    const isInfra = ["==", ["get", "featureType"], "infrastructure"];
    const isService = ["==", ["get", "featureType"], "service"];
    const cls = name => ["==", ["get", "railClass"], name];
    const both = (a, b) => ["all", a, b];

    addRailLayer({ id:"minor-lines", type:"line", source:"rail-data", filter:both(isInfra, cls("minor")), minzoom:9, layout:{visibility:"none", "line-cap":"round", "line-join":"round"}, paint:{"line-color":"#71808a", "line-opacity":.52, "line-width":["interpolate",["linear"],["zoom"],9,.7,14,1.8]} });
    addRailLayer({ id:"national-halo", type:"line", source:"rail-data", filter:both(isInfra, cls("national")), layout:{"line-cap":"round","line-join":"round"}, paint:{"line-color":"#fff","line-opacity":.9,"line-width":["interpolate",["linear"],["zoom"],7,2.8,12,6,16,8.5]} });
    addRailLayer({ id:"national-lines", type:"line", source:"rail-data", filter:both(isInfra, cls("national")), layout:{"line-cap":"round","line-join":"round"}, paint:{"line-color":NATIONAL_RAIL_COLOR,"line-width":["interpolate",["linear"],["zoom"],7,1.5,12,3.4,16,5.2]} });
    addRailLayer({ id:"national-detail", type:"line", source:"rail-data", filter:both(isInfra, cls("national")), minzoom:8.5, layout:{"line-cap":"round","line-join":"round"}, paint:{"line-color":"rgba(255,255,255,.95)","line-dasharray":[.18,1.25],"line-width":["interpolate",["linear"],["zoom"],8.5,.55,12,.9,16,1.3]} });
    addRailLayer({ id:"highspeed-halo", type:"line", source:"rail-data", filter:both(isInfra, cls("highspeed")), layout:{"line-cap":"round","line-join":"round"}, paint:{"line-color":"#fff","line-opacity":.94,"line-width":["interpolate",["linear"],["zoom"],7,3.2,12,7,16,10]} });
    addRailLayer({ id:"highspeed-lines", type:"line", source:"rail-data", filter:both(isInfra, cls("highspeed")), layout:{"line-cap":"round","line-join":"round"}, paint:{"line-color":NATIONAL_RAIL_COLOR,"line-width":["interpolate",["linear"],["zoom"],7,1.7,12,4.1,16,6.1]} });
    addRailLayer({ id:"highspeed-detail", type:"line", source:"rail-data", filter:both(isInfra, cls("highspeed")), minzoom:8.5, layout:{"line-cap":"round","line-join":"round"}, paint:{"line-color":"rgba(255,255,255,.96)","line-dasharray":[.18,1.25],"line-width":["interpolate",["linear"],["zoom"],8.5,.6,12,1.05,16,1.5]} });
    addRailLayer({ id:"intercity-halo", type:"line", source:"rail-data", filter:["any",both(isInfra,cls("intercity")),both(isService,cls("intercity"))], layout:{"line-cap":"round","line-join":"round"}, paint:{"line-color":"#fff","line-width":["interpolate",["linear"],["zoom"],7,3,12,8,16,11]} });
    addRailLayer({ id:"intercity-lines", type:"line", source:"rail-data", filter:["any",both(isInfra,cls("intercity")),both(isService,cls("intercity"))], layout:{"line-cap":"round","line-join":"round"}, paint:{"line-color":["coalesce",["get","color"],"#147D92"],"line-width":["interpolate",["linear"],["zoom"],7,1.8,12,5,16,7]} });
    addRailLayer({ id:"intercity-center", type:"line", source:"rail-data", filter:["any",both(isInfra,cls("intercity")),both(isService,cls("intercity"))], minzoom:8, layout:{"line-cap":"round","line-join":"round"}, paint:{"line-color":"rgba(255,255,255,.78)","line-width":["interpolate",["linear"],["zoom"],8,.45,12,1.05,16,1.55]} });
    addRailLayer({ id:"metro-infra", type:"line", source:"rail-data", filter:both(isInfra,cls("metro")), minzoom:8, layout:{"line-cap":"round","line-join":"round"}, paint:{"line-color":"#394b58","line-opacity":.45,"line-width":["interpolate",["linear"],["zoom"],8,1,13,3,17,5]} });
    addRailLayer({ id:"tram-infra-halo", type:"line", source:"rail-data", filter:["all",isInfra,["in",["get","railClass"],["literal",["tram","light_rail"]]]], minzoom:8, layout:{"line-cap":"round","line-join":"round"}, paint:{"line-color":"#fff","line-width":["interpolate",["linear"],["zoom"],8,2.5,14,7]} });
    addRailLayer({ id:"tram-infra", type:"line", source:"rail-data", filter:["all",isInfra,["in",["get","railClass"],["literal",["tram","light_rail"]]]], minzoom:8, layout:{"line-cap":"round","line-join":"round"}, paint:{"line-color":["coalesce",["get","color"],"#74818A"],"line-width":["interpolate",["linear"],["zoom"],8,1.2,14,4]} });
    addRailLayer({ id:"service-halo", type:"line", source:"rail-data", filter:["all",isService,cls("metro"),["!=",["get","status"],"construction"]], minzoom:7.2, layout:{"line-cap":"round","line-join":"round"}, paint:{"line-color":"rgba(255,255,255,.92)","line-width":["interpolate",["linear"],["zoom"],7.2,4.5,11,8.5,16,12]} });
    addRailLayer({ id:"service-lines", type:"line", source:"rail-data", filter:["all",isService,cls("metro"),["!=",["get","status"],"construction"]], minzoom:7.2, layout:{"line-cap":"round","line-join":"round"}, paint:{"line-color":["get","color"],"line-width":["interpolate",["linear"],["zoom"],7.2,2.3,11,5.3,16,8]} });
    addRailLayer({ id:"tram-service-halo", type:"line", source:"rail-data", filter:["all",isService,["in",["get","railClass"],["literal",["tram","light_rail"]]],["!=",["get","status"],"construction"]], minzoom:7.8, layout:{"line-cap":"round","line-join":"round"}, paint:{"line-color":"rgba(255,255,255,.9)","line-width":["interpolate",["linear"],["zoom"],7.8,3.4,11,7,16,10]} });
    addRailLayer({ id:"tram-service-lines", type:"line", source:"rail-data", filter:["all",isService,["in",["get","railClass"],["literal",["tram","light_rail"]]],["!=",["get","status"],"construction"]], minzoom:7.8, layout:{"line-cap":"round","line-join":"round"}, paint:{"line-color":["coalesce",["get","color"],"#74818A"],"line-width":["interpolate",["linear"],["zoom"],7.8,1.65,11,3.8,16,5.8]} });
    addRailLayer({ id:"construction-halo", type:"line", source:"rail-data", filter:cls("construction"), minzoom:7.5, layout:{"line-cap":"round","line-join":"round"}, paint:{"line-color":"rgba(255,255,255,.88)","line-dasharray":[.1,1.6],"line-width":["interpolate",["linear"],["zoom"],8,4,14,8]} });
    addRailLayer({ id:"construction-lines", type:"line", source:"rail-data", filter:cls("construction"), minzoom:7.5, layout:{"line-cap":"round","line-join":"round"}, paint:{"line-color":["coalesce",["get","color"],"#8A9298"],"line-dasharray":[.1,1.6],"line-width":["interpolate",["linear"],["zoom"],8,2,14,5]} });
    addRailLayer({ id:"service-labels", type:"symbol", source:"rail-data", filter:["all",isService,cls("metro")], minzoom:9.4, layout:{"symbol-placement":"line","symbol-spacing":430,"text-field":["get","name"],"text-size":["interpolate",["linear"],["zoom"],9,10,14,13],"text-font":["Noto Sans Regular"],"text-keep-upright":true,"text-max-angle":25}, paint:{"text-color":"#1d2b34","text-halo-color":"rgba(255,255,255,.96)","text-halo-width":1.8,"text-opacity":["interpolate",["linear"],["zoom"],9,0,10,1]} });
    addRailLayer({ id:"intercity-labels", type:"symbol", source:"rail-data", filter:["all",isService,cls("intercity")], minzoom:9.4, layout:{"symbol-placement":"line","symbol-spacing":430,"text-field":["get","name"],"text-size":["interpolate",["linear"],["zoom"],9,10,14,13],"text-font":["Noto Sans Regular"],"text-keep-upright":true,"text-max-angle":25}, paint:{"text-color":"#1d2b34","text-halo-color":"rgba(255,255,255,.96)","text-halo-width":1.8,"text-opacity":["interpolate",["linear"],["zoom"],9,0,10,1]} });
    addRailLayer({ id:"tram-labels", type:"symbol", source:"rail-data", filter:["all",isService,["in",["get","railClass"],["literal",["tram","light_rail"]]]], minzoom:11, layout:{"symbol-placement":"line","symbol-spacing":280,"text-field":["get","name"],"text-size":11,"text-font":["Noto Sans Regular"]}, paint:{"text-color":"#392d47","text-halo-color":"#fff","text-halo-width":1.6} });

    const stationClassColor = ["match",["get","railClass"],"tram","#8D5AC7","light_rail","#5E8FC6","metro","#126D75","intercity","#147D92","construction","#8A9298","#465762"];
    addRailLayer({ id:"stations-halo", type:"circle", source:"station-data", paint:{"circle-radius":["interpolate",["linear"],["zoom"],7,2.8,12,5.2,16,7.2],"circle-color":"rgba(255,255,255,.96)","circle-stroke-color":"rgba(255,255,255,.92)","circle-stroke-width":1.4,"circle-opacity":1,"circle-stroke-opacity":1} });
    addRailLayer({ id:"stations", type:"circle", source:"station-data", paint:{"circle-radius":["interpolate",["linear"],["zoom"],7,1.35,12,2.8,16,4.2],"circle-color":"#fff","circle-stroke-color":stationClassColor,"circle-stroke-width":["interpolate",["linear"],["zoom"],7,1.1,14,2.2],"circle-opacity":1,"circle-stroke-opacity":1} });
    addRailLayer({ id:"station-labels", type:"symbol", source:"station-data", minzoom:12, layout:{"text-field":["get","name"],"text-size":["interpolate",["linear"],["zoom"],11,10,15,13],"text-font":["Noto Sans Regular"],"text-offset":[0,1.05],"text-anchor":"top","text-allow-overlap":false,"text-optional":true,"symbol-sort-key":["case",["has","officialLines"],0,1]}, paint:{"text-color":"#1d2b34","text-halo-color":"rgba(255,255,255,.97)","text-halo-width":1.7} });

    applyStationStyle();
    applyRailCategoryStyles();
    applyLabelDetail();
    applyLayerToggles();
    raiseRailLayers();
  }

  function raiseRailLayers() {
    if (!map?.isStyleLoaded()) return;
    for (const id of railLayerOrder) if (map.getLayer(id)) map.moveLayer(id);
  }

  function layerToggleEnabled(group) {
    return Boolean(document.querySelector(`[data-layer-toggle="${group}"]`)?.checked);
  }

  function applyLayerToggles() {
    for (const [group, ids] of Object.entries(layerGroups)) {
      const enabled = layerToggleEnabled(group);
      ids.forEach(id => styleVisibility(id, enabled));
    }
    applyStationVisibility();
    raiseRailLayers();
  }

  function applyStationVisibility() {
    const enabled = layerToggleEnabled("stations");
    stationLayerIds.forEach(id => styleVisibility(id, enabled));
  }

  function expressionContainsZoom(value) {
    return Array.isArray(value) && (value[0] === "zoom" || value.some(expressionContainsZoom));
  }

  function scaledExpression(base, factor) {
    if (typeof base === "number") return base * factor;
    if (!Array.isArray(base)) return deepClone(base);
    const op = base[0];
    if (op === "interpolate") {
      const out = [base[0], deepClone(base[1]), deepClone(base[2])];
      for (let i = 3; i < base.length; i += 2) { out.push(deepClone(base[i]), scaledExpression(base[i + 1], factor)); }
      return out;
    }
    if (op === "step") {
      const out = [base[0], deepClone(base[1]), scaledExpression(base[2], factor)];
      for (let i = 3; i < base.length; i += 2) { out.push(deepClone(base[i]), scaledExpression(base[i + 1], factor)); }
      return out;
    }
    if (op === "case") {
      const out = [base[0]];
      for (let i = 1; i < base.length - 1; i += 2) out.push(deepClone(base[i]), scaledExpression(base[i + 1], factor));
      out.push(scaledExpression(base.at(-1), factor));
      return out;
    }
    if (op === "match") {
      const out = [base[0], deepClone(base[1])];
      for (let i = 2; i < base.length - 1; i += 2) out.push(deepClone(base[i]), scaledExpression(base[i + 1], factor));
      out.push(scaledExpression(base.at(-1), factor));
      return out;
    }
    if (op === "coalesce") return [base[0], ...base.slice(1).map(v => scaledExpression(v, factor))];
    if (expressionContainsZoom(base)) return deepClone(base);
    return ["*", deepClone(base), factor];
  }

  function stationColorExpression() {
    return ["match",["get","railClass"],"tram","#8D5AC7","light_rail","#5E8FC6","metro","#126D75","intercity","#147D92","national",NATIONAL_RAIL_COLOR,"highspeed",NATIONAL_RAIL_COLOR,"construction","#8A9298","#465762"];
  }

  function applyStationStyle() {
    if (!map?.getLayer("stations")) return;
    const color = stationColorExpression();
    const style = settings.stationStyle;
    const halo = { color:"rgba(255,255,255,.96)", stroke:"rgba(255,255,255,.92)", opacity:1, strokeOpacity:1 };
    const station = { color:"#fff", stroke:color, opacity:1, strokeOpacity:1 };
    if (style === "solid") { station.color = color; station.stroke = "rgba(255,255,255,.98)"; }
    else if (style === "ringdot") { halo.color = color; halo.stroke = "rgba(255,255,255,.96)"; station.color = "#fff"; station.stroke = color; }
    else if (style === "minimal") { halo.opacity = 0; halo.strokeOpacity = 0; station.color = color; station.stroke = "rgba(255,255,255,.9)"; }
    map.setPaintProperty("stations-halo", "circle-color", halo.color);
    map.setPaintProperty("stations-halo", "circle-stroke-color", halo.stroke);
    map.setPaintProperty("stations-halo", "circle-opacity", halo.opacity);
    map.setPaintProperty("stations-halo", "circle-stroke-opacity", halo.strokeOpacity);
    map.setPaintProperty("stations", "circle-color", station.color);
    map.setPaintProperty("stations", "circle-stroke-color", station.stroke);
    map.setPaintProperty("stations", "circle-opacity", station.opacity);
    map.setPaintProperty("stations", "circle-stroke-opacity", station.strokeOpacity);
    applyRailAppearance();
  }

  function configureLineLayer(id, { width = 1, opacity = 1, dash } = {}) {
    lineStyleWidthFactors.set(id, width);
    lineStyleOpacityFactors.set(id, opacity);
    if (dash !== undefined && map?.getLayer(id)) {
      try { map.setPaintProperty(id, "line-dasharray", dash); } catch (err) { console.debug("Line dash adjustment skipped", id, err); }
    }
  }

  function applyRailCategoryStyles() {
    if (!map) return;
    lineStyleWidthFactors.clear();
    lineStyleOpacityFactors.clear();

    const solidLayers = ["metro-infra","service-halo","service-lines","tram-infra-halo","tram-infra","tram-service-halo","tram-service-lines","national-halo","national-lines","highspeed-halo","highspeed-lines","intercity-halo","intercity-lines","intercity-center"];
    solidLayers.forEach(id => configureLineLayer(id, { dash:null }));

    if (settings.metroLineStyle === "solid") {
      configureLineLayer("service-halo", { opacity:0 });
      configureLineLayer("service-lines", { width:1.12 });
      configureLineLayer("metro-infra", { opacity:.55, width:.9 });
    } else if (settings.metroLineStyle === "slim") {
      configureLineLayer("service-halo", { opacity:0 });
      configureLineLayer("service-lines", { width:.62 });
      configureLineLayer("metro-infra", { opacity:.42, width:.58 });
    } else {
      configureLineLayer("service-halo", { opacity:1 });
      configureLineLayer("service-lines", { width:1 });
      configureLineLayer("metro-infra", { opacity:1 });
    }

    if (settings.tramLineStyle === "outlined") {
      ["tram-infra-halo","tram-service-halo"].forEach(id => configureLineLayer(id, { opacity:1, width:1 }));
      ["tram-infra","tram-service-lines"].forEach(id => configureLineLayer(id, { width:1 }));
    } else if (settings.tramLineStyle === "solid") {
      ["tram-infra-halo","tram-service-halo"].forEach(id => configureLineLayer(id, { opacity:0 }));
      ["tram-infra","tram-service-lines"].forEach(id => configureLineLayer(id, { width:1.1 }));
    } else {
      configureLineLayer("tram-infra-halo", { opacity:.22, width:.56 });
      configureLineLayer("tram-infra", { opacity:.48, width:.56 });
      configureLineLayer("tram-service-halo", { opacity:.42, width:.86 });
      configureLineLayer("tram-service-lines", { opacity:1, width:.9 });
    }

    const nationalMain = ["national-lines","highspeed-lines"];
    const nationalHalo = ["national-halo","highspeed-halo"];
    const nationalDetail = ["national-detail","highspeed-detail"];
    nationalMain.forEach(id => configureLineLayer(id, { dash:null }));
    nationalHalo.forEach(id => configureLineLayer(id, { dash:null }));
    if (settings.nationalLineStyle === "outlined") {
      nationalHalo.forEach(id => configureLineLayer(id, { opacity:1, width:1 }));
      nationalMain.forEach(id => configureLineLayer(id, { width:1.04 }));
      nationalDetail.forEach(id => configureLineLayer(id, { opacity:0, dash:[.18,1.25] }));
    } else if (settings.nationalLineStyle === "solid") {
      nationalHalo.forEach(id => configureLineLayer(id, { opacity:0 }));
      nationalMain.forEach(id => configureLineLayer(id, { width:1.16 }));
      nationalDetail.forEach(id => configureLineLayer(id, { opacity:0, dash:[.18,1.25] }));
    } else {
      nationalHalo.forEach(id => configureLineLayer(id, { opacity:.92 }));
      nationalMain.forEach(id => configureLineLayer(id, { width:1 }));
      nationalDetail.forEach(id => configureLineLayer(id, { opacity:1, dash:[.18,1.25] }));
    }

    if (settings.intercityLineStyle === "outlined") {
      configureLineLayer("intercity-halo", { opacity:1 });
      configureLineLayer("intercity-lines", { width:1 });
      configureLineLayer("intercity-center", { opacity:0 });
    } else if (settings.intercityLineStyle === "solid") {
      configureLineLayer("intercity-halo", { opacity:0 });
      configureLineLayer("intercity-lines", { width:1.12 });
      configureLineLayer("intercity-center", { opacity:0 });
    } else {
      configureLineLayer("intercity-halo", { opacity:1 });
      configureLineLayer("intercity-lines", { width:1 });
      configureLineLayer("intercity-center", { opacity:1 });
    }

    const dashBase = ({ dots:[.1,1.6], short:[1.4,1.4], long:[4,2], solid:null })[settings.constructionLineStyle] ?? [.1,1.6];
    const constructionDash = dashBase ? [
      Math.max(.05, dashBase[0] * settings.constructionDashLength),
      Math.max(.1, dashBase[1] * settings.constructionDashGap)
    ] : null;
    ["construction-halo","construction-lines"].forEach(id => {
      configureLineLayer(id, { opacity:id === "construction-halo" ? .92 : 1, dash:constructionDash });
      if (map.getLayer(id)) map.setLayoutProperty(id, "line-cap", settings.constructionLineCap);
    });
    applyRailAppearance();
  }

  function applyRailAppearance() {
    if (!map) return;
    for (const [id, defaults] of railLineDefaults) {
      if (!map.getLayer(id)) continue;
      const widthFactor = lineStyleWidthFactors.get(id) ?? 1;
      const opacityFactor = lineStyleOpacityFactors.get(id) ?? 1;
      try { map.setPaintProperty(id, "line-width", scaledExpression(defaults.width, settings.lineWidth * widthFactor)); } catch (err) { console.warn("Line width adjustment skipped", id, err); }
      try { map.setPaintProperty(id, "line-opacity", scaledExpression(defaults.opacity, settings.lineOpacity * opacityFactor)); } catch (err) { console.warn("Line opacity adjustment skipped", id, err); }
    }
    for (const [id, defaults] of stationPaintDefaults) {
      if (!map.getLayer(id)) continue;
      const strokeFactor = settings.stationStyle === "minimal" && id === "stations" ? 0.72 : 1;
      try { map.setPaintProperty(id, "circle-radius", scaledExpression(defaults.radius, settings.stationSize)); } catch (err) { console.warn("Station radius adjustment skipped", id, err); }
      try { map.setPaintProperty(id, "circle-stroke-width", scaledExpression(defaults.strokeWidth, settings.stationSize * strokeFactor)); } catch (err) { console.warn("Station stroke adjustment skipped", id, err); }
    }
  }

  const labelPresets = {
    compact: { serviceMin:10.2, tramMin:11.8, stationMin:12.6, serviceSpacing:560, tramSpacing:370 },
    standard: { serviceMin:9.4, tramMin:11, stationMin:12, serviceSpacing:430, tramSpacing:280 },
    detailed: { serviceMin:8.8, tramMin:10.2, stationMin:10.8, serviceSpacing:330, tramSpacing:220 }
  };

  function applyLabelDetail() {
    if (!map) return;
    const p = labelPresets[settings.labelDetail] || labelPresets.standard;
    if (map.getLayer("service-labels")) { map.setLayerZoomRange("service-labels", p.serviceMin, 24); map.setLayoutProperty("service-labels", "symbol-spacing", p.serviceSpacing); }
    if (map.getLayer("intercity-labels")) { map.setLayerZoomRange("intercity-labels", p.serviceMin, 24); map.setLayoutProperty("intercity-labels", "symbol-spacing", p.serviceSpacing); }
    if (map.getLayer("tram-labels")) { map.setLayerZoomRange("tram-labels", p.tramMin, 24); map.setLayoutProperty("tram-labels", "symbol-spacing", p.tramSpacing); }
    if (map.getLayer("station-labels")) map.setLayerZoomRange("station-labels", p.stationMin, 24);
  }

  const opacityPropertyByLayer = {
    background: ["background-opacity"], fill: ["fill-opacity"], line: ["line-opacity"],
    symbol: ["icon-opacity", "text-opacity"], circle: ["circle-opacity", "circle-stroke-opacity"],
    heatmap: ["heatmap-opacity"], "fill-extrusion": ["fill-extrusion-opacity"], raster: ["raster-opacity"]
  };

  const colorCanvas = document.createElement("canvas");
  const colorContext = colorCanvas.getContext("2d");

  function parseCssColor(value) {
    if (!colorContext || typeof value !== "string" || value.length > 80) return null;
    const text = value.trim();
    if (!text || /^(case|match|get|literal|interpolate|step|zoom|coalesce|format|concat)$/i.test(text)) return null;
    const sentinel = "rgba(1, 2, 3, 0.123)";
    colorContext.fillStyle = sentinel;
    colorContext.fillStyle = text;
    const normalized = colorContext.fillStyle;
    if (normalized === sentinel && text.toLowerCase() !== sentinel) return null;
    if (normalized.startsWith("#")) {
      let h = normalized.slice(1);
      if (h.length === 3) h = h.split("").map(c => c + c).join("");
      if (h.length !== 6) return null;
      return [parseInt(h.slice(0,2),16), parseInt(h.slice(2,4),16), parseInt(h.slice(4,6),16), 1];
    }
    const m = normalized.match(/^rgba?\(([^)]+)\)$/i);
    if (!m) return null;
    const parts = m[1].split(",").map(Number);
    if (parts.length < 3 || parts.some(n => !Number.isFinite(n))) return null;
    return [parts[0], parts[1], parts[2], parts[3] ?? 1];
  }

  function adjustedColor(value) {
    const c = parseCssColor(value);
    if (!c) return value;
    let [r, g, b, a] = c;
    const clamp = n => Math.max(0, Math.min(255, n));
    let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    r = y + (r - y) * settings.baseSaturation;
    g = y + (g - y) * settings.baseSaturation;
    b = y + (b - y) * settings.baseSaturation;
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    r += (y - r) * settings.baseGrayscale;
    g += (y - g) * settings.baseGrayscale;
    b += (y - b) * settings.baseGrayscale;
    r = (r - 128) * settings.baseContrast + 128;
    g = (g - 128) * settings.baseContrast + 128;
    b = (b - 128) * settings.baseContrast + 128;
    return `rgba(${Math.round(clamp(r))},${Math.round(clamp(g))},${Math.round(clamp(b))},${Math.max(0, Math.min(1, a))})`;
  }

  function transformColors(value) {
    if (typeof value === "string") return adjustedColor(value);
    if (Array.isArray(value)) return value.map(transformColors);
    if (value && typeof value === "object") return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, transformColors(v)]));
    return value;
  }

  function captureBaseLayers() {
    baseLayerSnapshots.clear();
    for (const layer of map.getStyle()?.layers || []) {
      if (railLayerIds.has(layer.id)) continue;
      baseLayerSnapshots.set(layer.id, {
        type: layer.type,
        paint: deepClone(layer.paint || {}),
        layout: deepClone(layer.layout || {}),
        sourceLayer: layer["source-layer"] || "",
        minzoom: Number.isFinite(layer.minzoom) ? layer.minzoom : 0,
        maxzoom: Number.isFinite(layer.maxzoom) ? layer.maxzoom : 24
      });
    }
  }

  function baseLayerHasText(snapshot) {
    const textField = snapshot?.layout?.["text-field"];
    return snapshot?.type === "symbol" && textField != null && textField !== "";
  }

  function baseLabelDetailRank(id, snapshot) {
    const text = `${id} ${snapshot.sourceLayer}`.toLowerCase().replace(/[_-]+/g, " ");
    if (/country|continent|state|province|capital|(?:^|\s)(?:city|town)(?:\s|$)/.test(text)) return 0;
    if (/village|district|borough|suburb|neighbou?rhood|quarter|locality|island|ocean|sea|water label|waterway|park|motorway|trunk|primary|road name|road label|street label/.test(text)) return 1;
    if (/poi|house ?number|address|aeroway|airport|transit|bus|ferry|railway|station|shop|amenity|tourism|building|path|track|service|pedestrian|footway|cycleway|minor|residential|living street|landuse/.test(text)) return 2;
    return 1;
  }

  function applyBaseDetail() {
    const mode = settings.baseDetail;
    const maxRank = mode === "none" ? -1 : mode === "minimal" ? 0 : mode === "detailed" ? 2 : 1;
    const spacingFactor = mode === "minimal" ? 1.8 : mode === "standard" ? 1.25 : 1;
    const zoomBoost = mode === "minimal" ? .65 : mode === "standard" ? .2 : 0;

    for (const [id, snapshot] of baseLayerSnapshots) {
      if (!map.getLayer(id)) continue;
      const originallyVisible = snapshot.layout.visibility !== "none";
      const isSymbol = snapshot.type === "symbol";
      const hasText = baseLayerHasText(snapshot);
      const rank = baseLabelDetailRank(id, snapshot);
      const showText = originallyVisible && hasText && rank <= maxRank;

      // Hide the complete symbol layer when its labels are suppressed. Merely
      // clearing text-field leaves road shields and icon backgrounds behind as
      // white boxes, especially in the minimal and no-information modes.
      const showLayer = originallyVisible && (mode === "detailed" || !isSymbol || (mode !== "none" && rank <= maxRank));
      try { map.setLayoutProperty(id, "visibility", showLayer ? "visible" : "none"); } catch (err) { console.debug("Base visibility adjustment skipped", id, err); }
      if (!hasText) continue;

      try { map.setLayoutProperty(id, "text-field", showText ? deepClone(snapshot.layout["text-field"]) : ""); } catch (err) { console.debug("Basemap text adjustment skipped", id, err); }

      const originalSpacing = snapshot.layout["symbol-spacing"];
      if (originalSpacing != null) {
        try { map.setLayoutProperty(id, "symbol-spacing", showText && typeof originalSpacing === "number" ? Math.round(originalSpacing * spacingFactor) : deepClone(originalSpacing)); } catch (err) { console.debug("Basemap label spacing skipped", id, err); }
      }

      const minzoom = Math.min(snapshot.maxzoom - .01, snapshot.minzoom + (showText ? zoomBoost : 0));
      try { map.setLayerZoomRange(id, Math.max(0, minzoom), snapshot.maxzoom); } catch (err) { console.debug("Basemap label zoom adjustment skipped", id, err); }
    }
  }

  function ensureLightMapBackground() {
    const container = map?.getContainer();
    const canvasContainer = map?.getCanvasContainer();
    const canvas = map?.getCanvas();
    if (container) container.style.backgroundColor = "#f4f7f7";
    if (canvasContainer) canvasContainer.style.backgroundColor = "#f4f7f7";
    if (canvas) canvas.style.backgroundColor = "#f4f7f7";
  }

  function applyBasemapAppearance() {
    if (!map?.isStyleLoaded()) return;
    ensureLightMapBackground();
    applyBaseDetail();
    for (const [id, snapshot] of baseLayerSnapshots) {
      if (!map.getLayer(id)) continue;
      for (const [property, original] of Object.entries(snapshot.paint)) {
        if (property.endsWith("-color")) {
          try { map.setPaintProperty(id, property, transformColors(original)); } catch (err) { console.debug("Color adjustment skipped", id, property, err); }
        }
      }
      for (const property of opacityPropertyByLayer[snapshot.type] || []) {
        const original = snapshot.paint[property] ?? 1;
        try { map.setPaintProperty(id, property, scaledExpression(original, settings.baseOpacity)); } catch (err) { console.debug("Opacity adjustment skipped", id, property, err); }
      }
    }
    raiseRailLayers();
  }

  function scheduleBasemapAppearance() {
    cancelAnimationFrame(baseApplyFrame);
    baseApplyFrame = requestAnimationFrame(applyBasemapAppearance);
  }

  function setupMap() {
    const iv = CFG.initialView;
    map = new maplibregl.Map({
      container: "map", style: CFG.basemaps[currentStyle], center: iv.center, zoom: iv.zoom,
      pitch: iv.pitch, bearing: iv.bearing, attributionControl: false, maxZoom: 19
    });
    map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), "top-right");
    map.addControl(new maplibregl.FullscreenControl(), "top-right");
    map.addControl(new maplibregl.ScaleControl({ maxWidth: 130, unit: "metric" }), "bottom-right");
    map.addControl(new maplibregl.AttributionControl({ compact: true, customAttribution: "轨道数据 © OpenStreetMap contributors" }), "bottom-right");
    ensureLightMapBackground();
    const refreshFullscreenAppearance = () => requestAnimationFrame(() => { ensureLightMapBackground(); map.resize(); scheduleBasemapAppearance(); });
    document.addEventListener("fullscreenchange", refreshFullscreenAppearance);
    document.addEventListener("webkitfullscreenchange", refreshFullscreenAppearance);

    map.on("load", async () => {
      captureBaseLayers();
      addRailLayers();
      applyBasemapAppearance();
    });
    map.on("style.load", () => {
      railLayerIds.clear();
      railLineDefaults.clear();
      stationPaintDefaults.clear();
      lineStyleWidthFactors.clear();
      lineStyleOpacityFactors.clear();
      captureBaseLayers();
      addRailLayers();
      applyBasemapAppearance();
    });
    map.on("error", e => { if (e?.error?.message) console.warn("Map error:", e.error.message); });
    setupInteractions();
  }

  function setupInteractions() {
    const clickable = ["stations", "station-labels", "service-lines", "tram-service-lines", "intercity-lines", "highspeed-lines", "national-lines", "tram-infra", "construction-lines"];
    map.on("click", async e => {
      const layers = clickable.filter(id => map.getLayer(id));
      const features = map.queryRenderedFeatures(e.point, { layers });
      if (!features.length) return;
      const f = features[0];
      const p = f.properties || {};
      const title = escapeHtml(p.name || "未命名");
      const meta = [p.ref && `编号：${escapeHtml(p.ref)}`, p.operator && `运营：${escapeHtml(p.operator)}`, p.network && `网络：${escapeHtml(p.network)}`, p.tunnel && `隧道：${escapeHtml(p.tunnel)}`, p.officialLines && `官方线路：${escapeHtml(p.officialLines)}`].filter(Boolean).join("<br>");
      const source = p.osmType && p.osmId ? `<br>OSM ${escapeHtml(p.osmType)} ${escapeHtml(p.osmId)}` : "";
      const typeText = classLabel(p.railClass);
      new maplibregl.Popup({ closeButton: true, maxWidth: "310px" }).setLngLat(e.lngLat).setHTML(`<div class="popup-title">${title}</div><div class="popup-meta">${meta || "暂无更多属性"}${source}</div><span class="popup-chip" style="background:${safeColor(p.color,p.name)}">${escapeHtml(typeText)}</span>`).addTo(map);
    });
    for (const id of clickable) {
      map.on("mouseenter", id, () => { map.getCanvas().style.cursor = "pointer"; });
      map.on("mouseleave", id, () => { map.getCanvas().style.cursor = ""; });
    }
  }

  function classLabel(cls) {
    return ({metro:"地铁",light_rail:"轻轨",tram:"有轨电车",intercity:"城际铁路",highspeed:"高速铁路",national:"国铁干线",minor:"其他铁路",construction:"建设中",station:"车站"})[cls] || "轨道交通";
  }

  function setUpdateStatus(text, state = "") {
    const el = $("#updateStatus");
    el.textContent = text;
    el.dataset.state = state;
  }

  async function fetchJsonWithTimeout(url, options = {}, timeoutMs = 195000) {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), timeoutMs);
    try {
      const res = await fetch(url, { ...options, signal: ctrl.signal });
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      return await res.json();
    } finally { clearTimeout(timer); }
  }

  function beginBundledGeoJSONLoad() {
    if (bundledGeoJSONPromise) return bundledGeoJSONPromise;
    if (!CFG.bundledGeoJSONPath) return Promise.resolve(null);
    bundledGeoJSONPromise = (async () => {
      try {
        let json = await window.GBA_RAIL_SNAPSHOT_PROMISE;
        if (!json) {
          const url = `${CFG.bundledGeoJSONPath}?v=${encodeURIComponent(CFG.version)}`;
          const res = await fetch(url, { cache:"force-cache", priority:"high" });
          if (!res.ok) return null;
          json = await res.json();
        }
        if (json?.type !== "FeatureCollection" || !Array.isArray(json.features) || !json.features.length) return null;
        return json;
      } catch { return null; }
    })();
    return bundledGeoJSONPromise;
  }

  async function tryBundledGeoJSON() {
    return beginBundledGeoJSONLoad();
  }

  async function tryLocalSnapshot(region) {
    const path = CFG.localSnapshotPath.replace("{id}", region.id);
    try {
      const res = await fetch(`${path}?v=${encodeURIComponent(CFG.version)}`, { cache: "force-cache" });
      if (!res.ok) return null;
      const json = await res.json();
      return json?.elements ? json : null;
    } catch { return null; }
  }

  async function queryOverpass(region) {
    const query = makeOverpassQuery(region);
    let lastError;
    for (const endpoint of CFG.overpassEndpoints) {
      try {
        const body = new URLSearchParams({ data: query });
        return await fetchJsonWithTimeout(endpoint, { method:"POST", headers:{"Content-Type":"application/x-www-form-urlencoded;charset=UTF-8"}, body });
      } catch (err) { lastError = err; console.warn(`Overpass failed: ${endpoint}`, err); }
    }
    throw lastError || new Error("所有 Overpass 节点均不可用");
  }

  async function loadRegion(region, force = false) {
    if (!force && regionStates.get(region.id) === "done") return 0;
    regionStates.set(region.id, "loading");
    try {
      let json = !force ? await tryLocalSnapshot(region) : null;
      let origin = "包内快照";
      if (!json && !force) { json = await cacheGet(region.id); origin = "永久缓存"; }
      if (!json) { json = await queryOverpass(region); origin = "在线"; await cachePut(region.id, json); }
      const features = parseOverpass(json, region.id);
      mergeFeatures(features);
      regionStates.set(region.id, "done");
      setUpdateStatus(`${region.name}：${origin} ${features.length} 个要素`, "ok");
      return features.length;
    } catch (err) {
      regionStates.set(region.id, "error");
      console.error(err);
      setUpdateStatus(`${region.name}读取失败`, "error");
      return 0;
    }
  }

  async function loadAll(force = false) {
    if (loadingAll) return 0;
    loadingAll = true;
    let total = 0;
    try {
      for (const region of CFG.regions) total += await loadRegion(region, force) || 0;
      if (featureStore.size) await bundlePut(currentGeoJSON());
      return total;
    } finally { loadingAll = false; }
  }

  async function bootstrapData() {
    setUpdateStatus("正在优先读取 rail_snapshot.geojson…");
    const bundled = await tryBundledGeoJSON();
    if (bundled) {
      mergeFeatures(bundled.features, { trusted:true });
      CFG.regions.forEach(r => regionStates.set(r.id, "done"));
      setUpdateStatus(`rail_snapshot.geojson · ${bundled.features.length} 个要素`, "ok");
      return;
    }
    const cachedBundle = await bundleGet();
    if (cachedBundle?.features?.length) {
      mergeFeatures(cachedBundle.features);
      CFG.regions.forEach(r => regionStates.set(r.id, "done"));
      setUpdateStatus(`永久缓存 · ${cachedBundle.features.length} 个要素`, "ok");
      return;
    }
    const total = await loadAll(false);
    if (total) {
      setUpdateStatus(`已固化到永久缓存 · ${featureStore.size} 个要素`, "ok");
      showToast("轨道数据已保存；以后启动无需重复刷新");
    } else {
      setUpdateStatus("未发现可用几何；可点击在线更新", "error");
    }
  }

  async function refreshAllData() {
    if (loadingAll) return;
    const button = $("#updateDataButton");
    button.disabled = true;
    loadingAll = true;
    setUpdateStatus("正在在线更新…");
    let updatedRegions = 0;
    let updatedFeatures = 0;
    try {
      for (const region of CFG.regions) {
        try {
          setUpdateStatus(`正在更新${region.name}…`);
          const json = await queryOverpass(region);
          const fresh = parseOverpass(json, region.id);
          if (!fresh.length) continue;
          for (const [key, feature] of featureStore) if (feature.properties.sourceRegion === region.id) featureStore.delete(key);
          mergeFeatures(fresh);
          await cachePut(region.id, json);
          regionStates.set(region.id, "done");
          updatedRegions++;
          updatedFeatures += fresh.length;
        } catch (err) {
          console.warn(`${region.name} update failed`, err);
        }
      }
      if (updatedRegions) {
        await bundlePut(currentGeoJSON());
        setUpdateStatus(`已更新 ${updatedRegions} 个区域并永久保存`, "ok");
        showToast(`更新完成：${updatedFeatures} 个要素；未成功区域保留旧数据`, 4600);
      } else {
        setUpdateStatus("更新失败，现有固化数据未被删除", "error");
        showToast("在线更新未取得新数据，已保留当前地图", 4600);
      }
    } finally {
      loadingAll = false;
      button.disabled = false;
    }
  }

  function nearestRegion(center) {
    return CFG.regions.map(r => ({...r, d:(r.center[0]-center.lng)**2 + (r.center[1]-center.lat)**2})).sort((a,b)=>a.d-b.d)[0];
  }

  function flyToCity(region) {
    map.flyTo({ center: region.center, zoom: region.zoom, duration: 900, essential: true });
    closeSidebarMobile();
  }

  function boundsOfGeometry(geometry) {
    const bounds = new maplibregl.LngLatBounds();
    const walk = coords => { if (typeof coords?.[0] === "number") bounds.extend(coords); else if (Array.isArray(coords)) coords.forEach(walk); };
    walk(geometry?.coordinates);
    return bounds.isEmpty() ? null : bounds;
  }

  function rebuildSearch() {
    searchEntries.length = 0;
    for (const r of CFG.regions) searchEntries.push({ key:`city:${r.id}`, title:r.name, sub:"城市定位", type:"城市", icon:"城", action:()=>flyToCity(r), search:`${r.name} ${r.id}` });
    const seen = new Set();
    for (const f of featureStore.values()) {
      const p = f.properties;
      if (!p.name || p.name === "未命名") continue;
      const dedupe = `${p.featureType}:${normalizeText(p.name)}:${p.ref || ""}`;
      if (seen.has(dedupe)) continue;
      seen.add(dedupe);
      searchEntries.push({
        key:p.key, title:p.name, sub:[p.nameEn,p.ref,p.operator].filter(Boolean).join(" · "),
        type:p.featureType === "station" ? "车站" : classLabel(p.railClass), icon:p.featureType === "station" ? "站" : "线",
        search:[p.name,p.nameEn,p.ref,p.operator,p.network,p.officialLines].filter(Boolean).join(" "), feature:f,
        action:()=>focusFeature(f)
      });
    }
  }

  function focusFeature(f) {
    if (f.geometry.type === "Point") map.flyTo({ center:f.geometry.coordinates, zoom:14.2, duration:900, essential:true });
    else {
      const b = boundsOfGeometry(f.geometry);
      if (b) map.fitBounds(b, { padding:{top:115,bottom:65,left:360,right:70}, maxZoom:13.5, duration:1000 });
    }
    hideSearchResults();
  }

  function runSearch(query) {
    const q = normalizeText(query);
    if (!q) { hideSearchResults(); return; }
    const terms = q.split(/\s+/).filter(Boolean);
    const results = searchEntries.map(e => {
      const hay = normalizeText(`${e.title} ${e.sub} ${e.search}`);
      let score = 0;
      if (normalizeText(e.title) === q) score += 100;
      if (normalizeText(e.title).startsWith(q)) score += 45;
      if (hay.includes(q)) score += 25;
      if (terms.every(t => hay.includes(t))) score += 12;
      if (e.type === "车站") score += 2;
      return {e,score};
    }).filter(x=>x.score>0).sort((a,b)=>b.score-a.score || a.e.title.localeCompare(b.e.title,"zh-CN")).slice(0,12).map(x=>x.e);
    renderSearchResults(results, query);
  }

  function renderSearchResults(results, query) {
    const box = $("#searchResults");
    if (!results.length) {
      box.innerHTML = `<div class="search-result"><span class="result-icon">?</span><span><div class="result-title">没有找到“${escapeHtml(query)}”</div><div class="result-sub">可搜索已加载的车站、线路或六座城市</div></span></div>`;
    } else {
      box.innerHTML = results.map((r,i)=>`<button class="search-result" type="button" data-result="${i}" role="option"><span class="result-icon">${escapeHtml(r.icon)}</span><span><div class="result-title">${escapeHtml(r.title)}</div><div class="result-sub">${escapeHtml(r.sub || "已加载地图要素")}</div></span><span class="result-tag">${escapeHtml(r.type)}</span></button>`).join("");
      $$('[data-result]').forEach(btn=>btn.addEventListener("click",()=>results[Number(btn.dataset.result)].action()));
    }
    box.hidden = false;
  }

  function hideSearchResults() { $("#searchResults").hidden = true; }

  function parseCSV(text) {
    const rows = []; let row=[], cell="", quote=false;
    for (let i=0;i<text.length;i++) {
      const ch=text[i], next=text[i+1];
      if (ch==='"' && quote && next==='"') { cell+='"'; i++; }
      else if (ch==='"') quote=!quote;
      else if (ch===',' && !quote) { row.push(cell); cell=""; }
      else if ((ch==='\n' || ch==='\r') && !quote) { if (ch==='\r' && next==='\n') i++; row.push(cell); if (row.some(v=>v!=="")) rows.push(row); row=[]; cell=""; }
      else cell+=ch;
    }
    if (cell || row.length) { row.push(cell); rows.push(row); }
    const header=rows.shift() || [];
    return rows.map(r=>Object.fromEntries(header.map((h,i)=>[h.replace(/^\uFEFF/,""),r[i]||""])));
  }

  async function loadOfficialCatalogs() {
    const getCSV = async url => { try { const r=await fetch(url); return r.ok ? parseCSV(await r.text()) : []; } catch { return []; } };
    officialCatalog.mtr = await getCSV(CFG.officialCatalogs.mtr);
    officialCatalog.lightRail = await getCSV(CFG.officialCatalogs.lightRail);
    enrichStationsFromOfficialCatalog();
  }

  function enrichStationsFromOfficialCatalog() {
    if (!officialCatalog.mtr.length && !officialCatalog.lightRail.length) return;
    const names = new Map();
    for (const row of officialCatalog.mtr) {
      const key = normalizeText(row["Chinese Name"]); if (!key) continue;
      if (!names.has(key)) names.set(key, new Set()); names.get(key).add(row["Line Code"]);
    }
    for (const row of officialCatalog.lightRail) {
      const key = normalizeText(row["Chinese Name"]); if (!key) continue;
      if (!names.has(key)) names.set(key, new Set()); names.get(key).add(`轻铁${row["Line Code"]}`);
    }
    for (const f of featureStore.values()) {
      if (f.properties.featureType !== "station") continue;
      const lines = names.get(normalizeText(f.properties.name));
      if (lines) f.properties.officialLines = [...lines].sort().join("、");
    }
  }

  function openDB() {
    if (!window.indexedDB) return Promise.resolve(null);
    if (dbPromise) return dbPromise;
    dbPromise = new Promise(resolve => {
      let req;
      try { req = indexedDB.open("gba-rail-cache", 2); }
      catch { resolve(null); return; }
      req.onupgradeneeded = () => {
        const db = req.result;
        if (!db.objectStoreNames.contains("regions")) db.createObjectStore("regions", { keyPath:"id" });
        if (!db.objectStoreNames.contains("bundles")) db.createObjectStore("bundles", { keyPath:"id" });
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => resolve(null);
    });
    return dbPromise;
  }

  async function cacheGet(regionId) {
    const db = await openDB(); if (!db) return null;
    return new Promise(resolve => {
      const tx = db.transaction("regions", "readonly");
      const store = tx.objectStore("regions");
      const exact = store.get(`${CFG.version}:${regionId}`);
      exact.onsuccess = () => {
        if (exact.result?.data) return resolve(exact.result.data);
        const all = store.getAll();
        all.onsuccess = () => {
          const matches = (all.result || []).filter(x => x.id === regionId || String(x.id).endsWith(`:${regionId}`)).sort((a,b)=>(b.savedAt||0)-(a.savedAt||0));
          resolve(matches[0]?.data || null);
        };
        all.onerror = () => resolve(null);
      };
      exact.onerror = () => resolve(null);
    });
  }

  async function cachePut(regionId, data) {
    const db = await openDB(); if (!db) return;
    return new Promise(resolve => {
      const tx = db.transaction("regions", "readwrite");
      tx.objectStore("regions").put({ id:`${CFG.version}:${regionId}`, savedAt:Date.now(), data });
      tx.oncomplete = () => resolve(); tx.onerror = () => resolve();
    });
  }

  async function bundleGet() {
    const db = await openDB(); if (!db) return null;
    return new Promise(resolve => {
      const tx = db.transaction("bundles", "readonly");
      const req = tx.objectStore("bundles").get("latest");
      req.onsuccess = () => resolve(req.result?.data || null);
      req.onerror = () => resolve(null);
    });
  }

  async function bundlePut(data) {
    const db = await openDB(); if (!db || !data?.features?.length) return;
    return new Promise(resolve => {
      const tx = db.transaction("bundles", "readwrite");
      tx.objectStore("bundles").put({ id:"latest", savedAt:Date.now(), version:CFG.version, data });
      tx.oncomplete = () => resolve(); tx.onerror = () => resolve();
    });
  }

  function saveSettings() {
    try { localStorage.setItem("gba-rail-display-v2", JSON.stringify({ ...settings, currentStyle, displayVersion:2 })); } catch { /* ignore */ }
  }

  function restoreSettings() {
    try {
      const currentRaw = localStorage.getItem("gba-rail-display-v2");
      const legacyRaw = localStorage.getItem("gba-rail-display-v1") || localStorage.getItem("gba-rail-display-v3");
      const saved = JSON.parse(currentRaw || legacyRaw || "null");
      if (!saved) return;
      for (const key of Object.keys(settings)) if (saved[key] != null) settings[key] = saved[key];
      if (!currentRaw) settings.metroLineStyle = "slim";
      if (saved.currentStyle && CFG.basemaps[saved.currentStyle]) currentStyle = saved.currentStyle;
    } catch { /* ignore */ }
  }

  function syncUIFromSettings() {
    $("#lineWidth").value = Math.round(settings.lineWidth * 100);
    $("#lineOpacity").value = Math.round(settings.lineOpacity * 100);
    $("#stationSize").value = Math.round(settings.stationSize * 100);
    $("#stationStyle").value = settings.stationStyle;
    $("#metroLineStyle").value = settings.metroLineStyle;
    $("#tramLineStyle").value = settings.tramLineStyle;
    $("#nationalLineStyle").value = settings.nationalLineStyle;
    $("#intercityLineStyle").value = settings.intercityLineStyle;
    $("#constructionLineStyle").value = settings.constructionLineStyle;
    $("#constructionDashLength").value = Math.round(settings.constructionDashLength * 100);
    $("#constructionDashGap").value = Math.round(settings.constructionDashGap * 100);
    $("#constructionLineCap").value = settings.constructionLineCap;
    $("#labelDetail").value = settings.labelDetail;
    $("#baseOpacity").value = Math.round(settings.baseOpacity * 100);
    $("#baseGrayscale").value = Math.round(settings.baseGrayscale * 100);
    $("#baseContrast").value = Math.round(settings.baseContrast * 100);
    $("#baseSaturation").value = Math.round(settings.baseSaturation * 100);
    $("#baseDetail").value = settings.baseDetail;
    $("#lineWidthValue").textContent = `${Math.round(settings.lineWidth * 100)}%`;
    $("#lineOpacityValue").textContent = `${Math.round(settings.lineOpacity * 100)}%`;
    $("#stationSizeValue").textContent = `${Math.round(settings.stationSize * 100)}%`;
    $("#constructionDashLengthValue").textContent = `${Math.round(settings.constructionDashLength * 100)}%`;
    $("#constructionDashGapValue").textContent = `${Math.round(settings.constructionDashGap * 100)}%`;
    $("#baseOpacityValue").textContent = `${Math.round(settings.baseOpacity * 100)}%`;
    $("#baseGrayscaleValue").textContent = `${Math.round(settings.baseGrayscale * 100)}%`;
    $("#baseContrastValue").textContent = `${Math.round(settings.baseContrast * 100)}%`;
    $("#baseSaturationValue").textContent = `${Math.round(settings.baseSaturation * 100)}%`;
    $$("[data-style]").forEach(btn => btn.classList.toggle("active", btn.dataset.style === currentStyle));
  }

  function isMobile() { return window.matchMedia("(max-width: 760px)").matches; }

  function setSidebarCollapsed(collapsed) {
    if (isMobile()) {
      $("#sidebar").classList.toggle("open", !collapsed);
      $("#backdrop").hidden = collapsed;
      return;
    }
    document.body.classList.toggle("sidebar-collapsed", collapsed);
    $("#sidebar").classList.toggle("collapsed", collapsed);
    $("#sidebarTab").hidden = !collapsed;
    try { localStorage.setItem("gba-sidebar-collapsed", collapsed ? "1" : "0"); } catch { /* ignore */ }
    setTimeout(() => map?.resize(), 230);
  }

  function closeSidebarMobile() { if (isMobile()) setSidebarCollapsed(true); }

  function buildUI() {
    restoreSettings();
    syncUIFromSettings();
    try { if (!isMobile() && localStorage.getItem("gba-sidebar-collapsed") === "1") setSidebarCollapsed(true); } catch { /* ignore */ }

    $$('[data-layer-toggle]').forEach(input => input.addEventListener("change", applyLayerToggles));
    $("#toggleAllLayers").addEventListener("click", () => { $$('[data-layer-toggle]').forEach(i => i.checked = true); applyLayerToggles(); });
    $$('[data-style]').forEach(btn => btn.addEventListener("click", () => {
      currentStyle = btn.dataset.style;
      $$('[data-style]').forEach(b => b.classList.toggle("active", b === btn));
      saveSettings();
      map.setStyle(CFG.basemaps[currentStyle]);
    }));

    const bindRange = (id, key, divisor, outputId, suffix, callback) => {
      const input = $(id);
      input.addEventListener("input", () => {
        settings[key] = Number(input.value) / divisor;
        $(outputId).textContent = `${input.value}${suffix}`;
        callback(); saveSettings();
      });
    };
    bindRange("#lineWidth", "lineWidth", 100, "#lineWidthValue", "%", applyRailAppearance);
    bindRange("#lineOpacity", "lineOpacity", 100, "#lineOpacityValue", "%", applyRailAppearance);
    bindRange("#stationSize", "stationSize", 100, "#stationSizeValue", "%", applyRailAppearance);
    bindRange("#constructionDashLength", "constructionDashLength", 100, "#constructionDashLengthValue", "%", applyRailCategoryStyles);
    bindRange("#constructionDashGap", "constructionDashGap", 100, "#constructionDashGapValue", "%", applyRailCategoryStyles);
    bindRange("#baseOpacity", "baseOpacity", 100, "#baseOpacityValue", "%", scheduleBasemapAppearance);
    bindRange("#baseGrayscale", "baseGrayscale", 100, "#baseGrayscaleValue", "%", scheduleBasemapAppearance);
    bindRange("#baseContrast", "baseContrast", 100, "#baseContrastValue", "%", scheduleBasemapAppearance);
    bindRange("#baseSaturation", "baseSaturation", 100, "#baseSaturationValue", "%", scheduleBasemapAppearance);

    $("#stationStyle").addEventListener("change", e => { settings.stationStyle = e.target.value; applyStationStyle(); saveSettings(); });
    $("#metroLineStyle").addEventListener("change", e => { settings.metroLineStyle = e.target.value; applyRailCategoryStyles(); saveSettings(); });
    $("#tramLineStyle").addEventListener("change", e => { settings.tramLineStyle = e.target.value; applyRailCategoryStyles(); saveSettings(); });
    $("#nationalLineStyle").addEventListener("change", e => { settings.nationalLineStyle = e.target.value; applyRailCategoryStyles(); saveSettings(); });
    $("#intercityLineStyle").addEventListener("change", e => { settings.intercityLineStyle = e.target.value; applyRailCategoryStyles(); saveSettings(); });
    $("#constructionLineStyle").addEventListener("change", e => { settings.constructionLineStyle = e.target.value; applyRailCategoryStyles(); saveSettings(); });
    $("#constructionLineCap").addEventListener("change", e => { settings.constructionLineCap = e.target.value; applyRailCategoryStyles(); saveSettings(); });
    $("#labelDetail").addEventListener("change", e => { settings.labelDetail = e.target.value; applyLabelDetail(); saveSettings(); });
    $("#baseDetail").addEventListener("change", e => { settings.baseDetail = e.target.value; scheduleBasemapAppearance(); saveSettings(); });
    $("#updateDataButton").addEventListener("click", refreshAllData);
    $("#aboutButton").addEventListener("click", () => $("#aboutDialog").showModal());

    $("#menuButton").addEventListener("click", () => setSidebarCollapsed(false));
    $("#sidebarTab").addEventListener("click", () => setSidebarCollapsed(false));
    $("#collapseSidebar").addEventListener("click", () => setSidebarCollapsed(true));
    $("#backdrop").addEventListener("click", closeSidebarMobile);

    const input = $("#searchInput");
    input.addEventListener("input", () => runSearch(input.value));
    input.addEventListener("focus", () => { if (input.value) runSearch(input.value); });
    input.addEventListener("keydown", e => { if (e.key === "Escape") { input.blur(); hideSearchResults(); } });
    $("#clearSearch").addEventListener("click", () => { input.value = ""; hideSearchResults(); input.focus(); });
    document.addEventListener("click", e => { if (!e.target.closest(".search-wrap")) hideSearchResults(); });
    window.addEventListener("resize", () => { if (isMobile()) $("#sidebarTab").hidden = true; map?.resize(); });
  }

  buildUI();
  rebuildSearch();
  beginBundledGeoJSONLoad();
  bootstrapData();
  loadOfficialCatalogs();
  setupMap();
})();
