# 资料来源清单

更新日期：2026-07-31。网址写入此文件便于后续下载与核验；正式发布时须重新检查许可、更新时间和使用条款。

## 已直接放入压缩包

| 文件 | 内容 | 原始地址 | 备注 |
|---|---|---|---|
| `data/reference/hk/mtr_lines_and_stations.csv` | 港铁线路、方向、车站代码、中文/英文站名、站序 | `https://opendata.mtr.com.hk/data/mtr_lines_and_stations.csv` | 不含坐标；权利归属与使用条件以港铁 / DATA.GOV.HK 页面为准 |
| `data/reference/hk/light_rail_routes_and_stops.csv` | 香港轻铁路线、方向、站代码、站名、站序 | `https://opendata.mtr.com.hk/data/light_rail_routes_and_stops.csv` | 不含坐标；用于校核 |

## 网页运行时数据

| 来源 | 用途 | 地址 / 接口 | 注意事项 |
|---|---|---|---|
| OpenStreetMap | 铁路、地铁、轻轨、有轨电车、车站、道路、建筑、水域等 | `https://www.openstreetmap.org/` | ODbL；必须署名。完整度和地下线位精度不一 |
| Overpass API | 按六城范围提取 OSM 轨道要素 | `https://overpass-api.de/api/interpreter`、`https://overpass.kumi.systems/api/interpreter` | 公共共享服务，不适合高流量生产站点 |
| OpenFreeMap | Liberty / Positron 矢量底图 | `https://tiles.openfreemap.org/styles/liberty` 等 | 基于 OSM；需保留署名 |
| MapLibre GL JS | 地图渲染 | `https://unpkg.com/maplibre-gl@5.24.0/` | BSD-3-Clause |

## 可下载的统一底图与地形数据

| 来源 | 覆盖 / 格式 | 建议用途 |
|---|---|---|
| Geofabrik Guangdong | 广东并含香港、澳门；`.osm.pbf`、Shape/GPKG 等 | 生产本地 OSM 数据库与可重复快照：`https://download.geofabrik.de/asia/china/guangdong.html` |
| Copernicus DEM GLO-30 | 全球约 30 米高程 | 生成山体阴影和粗略地形：`https://dataspace.copernicus.eu/` |
| 香港地政总署开放空间数据 | iB1000/iB5000、正射影像、地形和建筑等 | 香港官方底图精化：`https://www.landsd.gov.hk/en/spatial-data/open-data.html` |
| 澳门地图绘制暨地籍局 | 数字地形图、网上地图等 | 澳门建筑/道路/水体校核：`https://www.dscc.gov.mo/` |

## 官方轨道资料核验

### 广州与广东城际

- 广州地铁官方 APP、微信公众号和互联互通版线网图。
- 广州市政府关于新版线网图：包含广州地铁、广清/广州东环/广肇/广惠城际、佛山地铁 2/3 号线和南海有轨电车 1 号线。
- 2026-03-15 广东城际运行图：广惠、广肇、广清、广州东环、琶莲城际。
- 广州有轨电车：海珠有轨电车 1 号线、黄埔有轨电车 1 号线。

参考：
- `https://www.gz.gov.cn/zwfw/zxfw/jtfw/content/post_10726817.html`
- `https://www.gz.gov.cn/zwfw/zxfw/jtfw/content/post_10310508.html`

### 深圳

- 深圳市交通运输局轨道线路和网络图页面。
- 龙华现代有轨电车官方资料，包含清湖—新澜、清湖—下围、新澜—下围等服务。

参考：
- `https://jtys.sz.gov.cn/ydmh/jtcx/dtcx_180970/`

### 佛山

- 佛山地铁官网 / APP 用于运营线、站序、首末班车核验。
- 广州地铁互联互通版线网图可交叉核验佛山 2/3 号线与南海有轨电车 1 号线。

参考：`https://www.fmetro.net/`

### 东莞

- 东莞地铁 1 号线于 2025-11-28 开通，应避免使用仅含 2 号线的旧资料。
- 东莞轨道交通局 / 市政府线路与换乘指引。

参考：`https://www.dg.gov.cn/`

### 香港

- DATA.GOV.HK 港铁路线、车站、轻铁路线和车站 CSV；通常按月更新。
- 香港运输署公共交通数据可提供电车路线/站点；电车站 XML：`https://static.data.gov.hk/td/routes-fares-xml/STOP_TRAM.xml`。
- 香港地政总署 iB5000/iB1000 可用于高精度道路、建筑和地形。

### 澳门

- 澳门轻轨官方路线页：氹仔线、石排湾线、横琴线；东线建设中。
- 官方页面：`https://www.mlm.com.mo/sc/route.html`

## 用户上传 ZIP 审查

上传的 `metrobay.zip` 主要是保存下来的百度地图 / 高德地图网页快照及其脚本、图片和广告资源。未发现独立、来源明确、许可清晰的轨道 GeoJSON / Shapefile / CSV 坐标数据。因此：

- 没有把网页画面、瓦片、脚本或坐标复制到成品；
- 仅可在个人研究中作为视觉对照；
- 若其中还有你自行制作但被网页资源淹没的数据，建议单独导出为 GeoJSON、KML、GPX、CSV、SHP 或 GeoPackage 后再导入。

## 许可与合规

- OSM 数据采用 ODbL，需显示“© OpenStreetMap contributors”并遵守数据库许可要求。
- MapLibre GL JS 采用 BSD-3-Clause。
- OpenRailwayMap 网站代码与样式有其自身许可，数据仍受 OSM 许可约束。
- 港铁和其他运营方资料不能因“公开下载”就自动视为可任意再分发；公开部署前须检查各来源页面条款。
- 面向公众提供包含真实地理底图的中国大陆互联网地图，可能涉及地图审核、审图号、测绘资质和服务器等要求；本原型不是合规意见。
