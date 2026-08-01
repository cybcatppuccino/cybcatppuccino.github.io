# 制图方法与数据模型

## 目标

本项目采用地理地图而非拓扑示意图：线路尽量沿 OSM 中的实际轨道、隧道或高架中心线显示。道路、建筑、陆地和水域由开放矢量底图提供。

## OSM 查询范围

每个城市查询以下对象：

- `railway=rail|subway|light_rail|tram|monorail|narrow_gauge|construction` 的线要素
- `route=subway|light_rail|tram` 的路线关系
- `route=train` 且网络名称匹配广东城际的关系
- `railway=station|halt|tram_stop` 的点要素

查询输出使用 `out body geom`，使静态浏览器无需另行解析 OSM 节点引用。

## 显示优先级

1. 彩色运营路线关系
2. 城际基础设施/路线
3. 高铁与国铁干线
4. 地铁、轻轨和有轨电车物理轨道
5. 其他铁路与站场侧线（默认关闭）
6. 建设中线路（橙色虚线）

## 分类规则

- 地铁：`route=subway`、`railway=subway` 或 `station=subway`
- 轻轨：`route/railway=light_rail` 或 `railway=monorail`
- 有轨电车：`route/railway=tram`
- 城际：路线、网络、运营者或名称中包含“城际 / Intercity / 广东城际 / 珠三角城际”
- 高速铁路：`highspeed=yes`、较高 `maxspeed` 或名称含高铁/高速铁路
- 国铁干线：`railway=rail` 且 `usage=main`，以及未标注为站场服务线的主轨道
- 其他铁路：服务线、站场侧线和无法判断等级的轨道

这些是显示规则，不是法律或工程定义。

## 坐标与精度

OSM/网页使用 WGS84 经纬度并由 MapLibre 投影到 Web Mercator。香港和澳门官方工程数据若以后加入，应先从各自本地坐标系正确转换，再与 OSM 对齐。不得直接复制百度、高德等互联网地图的画面或坐标作为开放数据。

## 后续可改进项

- 从 Geofabrik 广东 PBF 制作可版本化的本地数据库
- 逐条核对地下线路环评、保护区和工程图
- 用 PostGIS + Martin/Tippecanoe 生成矢量瓦片，替代浏览器直接加载大 GeoJSON
- 增加基础设施与运营服务分离的数据表
- 对共线、复线和上下行轨道做线位偏移
- 加入正式的开通日期、状态时间轴和来源版本
