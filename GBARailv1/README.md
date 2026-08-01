# 湾区轨道地图 v2.1

覆盖广州、佛山、东莞、深圳、香港和澳门的铁路、地铁、轻轨、有轨电车、车站及建设中线路地图。

## 启动

不要直接双击 `index.html`。请在解压后的项目目录启动本地服务器：

- Windows：双击 `start_windows.bat`
- macOS / Linux：运行 `./start_macos_linux.sh`
- 或运行：`python serve.py`

然后打开终端显示的本地地址，通常为 `http://127.0.0.1:8080`。

## v2.1 主要变化

- 有轨电车与轻轨服务线直接使用快照中的具体线路色；少数双向记录颜色不一致时按同一路线修复。
- 有明确线路编号或名称的轨道底层会复用对应服务线颜色；无法归属到单一线路的共线轨道改为中性底层，不再以统一紫色覆盖。
- 地铁默认使用纤细直线，同时保留彩线描边和普通实线选项。
- 底图信息控制改为直接调节底图文字图层，并增加“无信息”选项；精简、标准、详细会逐级增加文字类别和线状标注密度。
- 保留 v1.2 的快照优先加载、并行压缩传输、建设中点线参数和其他既有功能。

## 数据加载顺序

1. `data/rail_snapshot.geojson`（启动即高优先级预取，并始终优先）
2. 完整 GeoJSON 永久缓存（仅在包内快照不可用时作为后备）
3. `data/osm/{region}.json`
4. 旧版本区域缓存
5. 在线 Overpass 更新

在线更新成功后会写入永久缓存，但下次启动仍优先使用包内 `rail_snapshot.geojson`。主动更新按钮保留。

## Heap snapshot

本包从用户提供的 Heap snapshot 恢复了 17,564 个要素的目录属性，保存于：

- `data/snapshot_feature_catalog.json`
- `data/rail_snapshot_manifest.json`
- `data/SNAPSHOT_EXTRACTION.md`

由于 V8 Heap Snapshot 不包含 `FixedDoubleArray` 的经纬度数值，无法仅凭该文件恢复精确线位。详细技术原因和补全方式见 `data/SNAPSHOT_EXTRACTION.md`。

## 完全离线说明

轨道数据可以从包内快照或永久缓存离线读取；当前 MapLibre GL JS 和 OpenFreeMap 底图仍由 CDN/在线瓦片提供。若需要地图引擎和底图也完全离线，需要另行打包 MapLibre 静态文件与离线瓦片。

## 数据与许可

轨道与地图数据 © OpenStreetMap contributors，采用 ODbL。底图由 OpenFreeMap 提供。地图渲染使用 MapLibre GL JS。港铁参考 CSV 的权利和使用条件以原发布方为准。
