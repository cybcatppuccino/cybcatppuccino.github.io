将原始 Overpass JSON 快照放入本目录并按区域命名：
  guangzhou.json
  foshan.json
  dongguan.json
  shenzhen.json
  hongkong.json
  macau.json

网页会优先读取这些包内文件。文件应包含 Overpass 标准 JSON 的 elements 数组。
也可以把完整、有效的 GeoJSON FeatureCollection 保存为 data/rail_snapshot.geojson；该文件优先级更高。
