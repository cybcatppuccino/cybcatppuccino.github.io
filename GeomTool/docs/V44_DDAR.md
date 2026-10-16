# GeomTool v4.4：AlphaGeometry1 / AlphaGeometry2 DDAR 集成说明

## 范围和架构

- `assets/ddar.js`：原有 `GeomDDAR.infer(snapshot, opts)` 接口**参数及结果对象字段**的兼容层；改为返回 Promise，因 Web Worker / Pyodide 无法同步执行。保留 `facts`, `matches`, `goal`, `counts`, `version`, `elapsedMs` 等字段。
- `assets/ddar-compiler.js`：GeomTool 构造节点转换为 AlphaGeometry2 输入点、坐标与形式前提。未能安全转换的构造列入 `limitations`，不会把数值扫描识别出的关系当作证明前提。
- `assets/ddar-ag1.js`：GeomTool 构造转换为 AlphaGeometry1 形式语言，**目前仅覆盖子集**。不支持构造按自由点降级处理，可能遗漏可证目标，不能与完全 AG1 原生输入等同。
- `assets/ddar-py.worker.js`：惰性加载 Pyodide 0.26.4、NumPy、SciPy（AG1），将 `vendor/` 中真实 Python 源码装入 Python 文件系统，调用桥接层。
- `vendor/alphageometry2/`：来自用户提供的 AlphaGeometry2 官方 Python DDAR、消元和解析模块，不改其中的数学算法；`geomtool_bridge.py` 为只做输入/查询转接的自有桥接层。
- `vendor/alphageometry1/`：来自用户提供的 AlphaGeometry 官方 DD、AR、图和 `rules.txt`, `defs.txt`，另有浏览器用的 absl Python 模块；`numericals.py` 仅让 matplotlib 可视化导入变为可选。原版数学规则未改变；`geomtool_ag1_bridge.py` 调用原生求解并返回查询状态。
- **保持不变：** `assets/ddar-goals.js`、`assets/auxiliary.js`、几何作图 Kernel 和关系探索模块的内容。旧 `assets/ddar.worker.js` 已改为独立的目标发掘 Worker，不执行旧 DDAR。

## 使用和依赖

在解压后的 `GeomToolv4.4` 目录里执行：

```bash
python -m http.server 8000
```

访问 `http://localhost:8000/`。首次需要通过 HTTPS CDN 下载 Pyodide 0.26.4 和相关 wasm/npm 包（脚本预置了 jsDelivr 和 Pyodide 官方地址）。首次打开可能有较大下载；每次初始化内核也会有延迟。切换 AG1 会再装载 SciPy。若浏览器禁止外网/CDN，程序会清晰报错，**不会悄悄退回旧 JS 证明器**。压缩包自带引擎 Python 源码，不带约 10MB 以上的 Pyodide 运行时。

## 目标状态

- `proved`：所选官方引擎在形式前提上确认了待证谓词。
- `unknown`：本次闭包中未确认，不代表数学上为假。
- `unsupported`：当前输入形式无法安全转换或检查。

示意坐标可帮助推理器搜索，但数值近似相等本身不成为假设。为了避免未校验关系，“结论发掘”产生的目标被逐一送入 DDAR 验证，再标为已证。

AlphaGeometry2 `elimination.py` 原版不记录完整代数证明回溯；v4.4 不会构造伪“原始证明步骤”。界面中的 `facts` 是被内核检验的关系集合，不是全量闭包的逐条列举。AG1 输出也仅声明官方目标检查的结果，不承诺把完整 proof DAG 转换为 v4.3 的证据对象。

## 本地对照与运行验证

在具备 NumPy、SciPy、absl-py 的 CPython 环境下：

```bash
python tests/v44_native_parity.py
```

已验证：AG2 中点/自由点/垂足/圆构造；AG1 中点/自由点/垂足/圆/垂心；AG2→AG1→AG2 的模块共存；两条官方 AG2 IMO 用例。这只能证明选定测试的**原生 Python 执行**正确性。网页环境的 NumPy、SciPy、Pyodide / WASM 加载情况需要单独的浏览器端对照；当前环境**未取得 Pyodide 运行时**。`tests/v44_browser_shell.py` 在本环境执行超时，因此浏览器端未通过验收，**不声称完整复刻或完全一致**。

## 兼容性边界

1. 原参数/结果字段兼容，但 `GeomDDAR.infer(...)` 必须 `await`。现有 GeomTool 页面已修改；依赖此 API 的外部同步调用需更新。
2. AG1 和 AG2 谓词语言不同，部分 GeomTool 复杂构造转换未覆盖；有限词法及数值退化会返回 unsupported 或 unknown。
3. 官方 AG2 的纯 Python 公开代码不等于论文中未公开的工业版系统。证据追踪与辅助构造策略不在本阶段升级范围内。
4. `GeomToolv4.3-Standalone.html` 是旧版 JS DDAR 构建产物，**不包含**在 v4.4 发行包中；v4.4 不是纯单 HTML 离线版。
5. 外网 CDNs、CORS 与浏览器内存都可能影响运行；不能保证任何环境下与原生 CPython 严格位级一致。
