# GBARail v2 数据合并与分片

本版本以旧包 `rail_snapshot.geojson` 为优先来源，并按 OSM 类型与 ID 合并 `datum` 中缺失的节点、轨道和线路关系。重复键不会覆盖旧记录。

完整合并数据保存在 `rail_snapshot.geojson`；网页默认读取 `chunks/manifest.json`，按当前地图视野选择概览或详细分片。线路关系成员优先用于识别地铁、城际、轻轨、有轨电车及其车辆段、联络线，避免仅因底层 `railway=rail` 标签而误归为国铁。

详细统计见 `merge_audit.json`。
