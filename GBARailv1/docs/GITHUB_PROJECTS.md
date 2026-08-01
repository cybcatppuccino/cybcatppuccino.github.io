# GitHub 与开源项目调研

以下项目适合参考或用于后续升级；本原型没有直接复制它们的铁路数据。

## 核心技术

### MapLibre GL JS

- 仓库：`https://github.com/maplibre/maplibre-gl-js`
- 用途：浏览器 WebGL 矢量地图渲染、GeoJSON 图层、交互、比例尺、全屏等。
- 本项目：作为前端地图引擎，通过 CDN 固定在 5.24.0。
- 许可：BSD-3-Clause。

### OpenFreeMap

- 仓库：`https://github.com/hyperknot/openfreemap`
- 用途：基于 OpenStreetMap 的开放矢量底图托管和样式。
- 本项目：使用 Liberty、Positron 样式 URL。
- 生产建议：高流量或严格可用性场景应自托管或采用有 SLA 的服务。

### OpenRailwayMap

- 仓库：`https://github.com/OpenRailwayMap/OpenRailwayMap`
- 用途：基于 OSM 的全球铁路基础设施专题地图、铁路标记规则和样式。
- 价值：非常适合核对铁路分类、信号、电气化、速度等标签；其完整网站结构较重，不适合直接嵌入本简洁原型。
- 许可：网站代码 GPL-3.0；地图数据仍来自 OSM/ODbL。

### osmtogeojson

- 仓库：`https://github.com/tyrasd/osmtogeojson`
- 用途：把 Overpass/OSM JSON 或 XML 转成 GeoJSON；Overpass Turbo 也使用它。
- 本项目：为减少 CDN 依赖，当前只针对所需对象实现了轻量解析；复杂关系处理可改用该库。

### Martin

- 仓库：`https://github.com/maplibre/martin`
- 用途：从 PostGIS、MBTiles、PMTiles 等提供高性能矢量瓦片。
- 价值：若以后要把整个广东 PBF、建筑和大量轨道做成正式网站，这是推荐的后端路线之一。
- 许可：MIT / Apache-2.0。

## 数据与边界参考

### ChinaGeoJSON

- 仓库：`https://github.com/longwosion/geojson-map-china`
- 用途：行政区 GeoJSON 示例。
- 注意：许多此类仓库更新慢、边界来源和审图合规不清；不应直接作为面向公众的中国地图边界权威数据。

## 不建议直接依赖的项目

GitHub 上有不少“中国地铁线路 JSON / 全国地铁站点”仓库和 Gist，但常见问题是：

- 停止更新多年，缺少 2024—2026 年新线
- 只有示意线或站序，没有精确坐标
- 坐标来自商业地图平台但未说明转换和许可
- 缺少来源日期、运营状态和许可证

因此本项目把它们仅作为线索，当前线路几何以 OSM 为主，运营信息再由官方资料核对。
