# Heap snapshot 提取说明

## 已成功固化的内容

从 `Heap.heapsnapshot` 中定位到最终 `FeatureCollection`，共恢复 17,564 个地图要素：

- 15,937 段轨道基础设施
- 1,401 个车站
- 226 条线路关系

`data/snapshot_feature_catalog.json` 已保存每个要素可恢复的 OSM ID、区域、要素类型、轨道类别、线路/车站名称、英文名、编号、运营方、网络、状态、线路色等属性。统计见 `rail_snapshot_manifest.json`。

## 为什么没有生成可直接使用的完整 rail_snapshot.geojson

该页面在运行时把经纬度坐标放入 JavaScript 数字数组。V8 对这些数组使用 `FixedDoubleArray` 内联保存。Chrome Heap Snapshot 会记录数组对象、长度和引用关系，但不会把 FixedDoubleArray 内部的双精度数值写入 `.heapsnapshot` 文件。

因此，快照里可以完整找到线路/站点对象和属性，也可以确认其几何类型，但无法从该文件本身还原精确经纬度。生成空坐标或伪造线位会让地图显示错误，所以本包没有这样做。

## 新版如何避免反复刷新

v0.3 的加载顺序为：

1. `data/rail_snapshot.geojson`（用户后续放入有效文件时自动启用）
2. `data/osm/*.json` 包内 Overpass 快照
3. 浏览器中的完整 GeoJSON 永久缓存
4. 兼容读取旧版本 IndexedDB 区域缓存
5. 以上均不存在时才在线更新

一次成功加载或在线更新后，完整几何会写入 IndexedDB 的 `bundles/latest`，不再按七天过期，也不因应用版本号变化而丢失。主动点击“在线更新线路”时，更新失败的区域会继续保留旧数据。

## 真正制作完全离线包所需文件

以下任一种即可补全精确几何：

- 原页面 IndexedDB 中 `gba-rail-cache / regions` 的导出数据
- 浏览器控制台执行 `JSON.stringify(currentGeoJSON())` 得到的完整 GeoJSON（需在原页面仍运行时导出）
- 原始 Overpass JSON 响应
- 已生成的 `data/osm/guangzhou.json` 等六个区域文件

将有效的完整 FeatureCollection 保存为 `data/rail_snapshot.geojson`，v0.3 会在启动时自动读取，并保留在线更新能力。
