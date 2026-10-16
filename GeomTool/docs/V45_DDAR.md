# GeomTool v4.5 — DDAR 接入、会话与验收边界

## 使用和界面

解压，在 `GeomToolv4.5/` 目录运行 `python -m http.server 8000`，浏览器打开 `http://localhost:8000/`。首次启动须能够访问 Pyodide 0.26.4 的 CDN，AG2 加载 NumPy，AG1 另加载 SciPy。**不能直接通过 `file://` 运行其推理 Worker**。

「结论探知」区域只保留：AG1 / AG2 选择；“发掘结论”；中文/英文模糊检索与候选形式命题选择；“尝试证明目标”和“尝试推理闭包”；“停止搜索 / 继续搜索”；按秒更新进度；可搜索的已证结论列表；点击结论展开证明详情。辅助构造按钮从页面隐藏，但原来的 `assets/auxiliary.js` 和兼容函数仍在项目中。数值关系探索旧模块源码也仍在项目中，但其冗余页面区域默认隐藏。

列表仅显示由指定内核确认的事实，数值候选**不等于**严格证明。候选数量不再采用 v4.4 的 1200 项固定上限；页面一次最多渲染 180 个匹配候选，继续输入关键词可以查看其他匹配。推理结果列表采用滚动追加、每批 90 行，以减少 DOM 卡顿。这些是**呈现分页**，不代表只向内核传入 180/90 个目标。

## 原始代码与接入层

| 部件 | 文件 | 用途 |
|---|---|---|
| AG2 原始推理 | `vendor/alphageometry2/ddar.py`, `elimination.py`, `numericals.py`, `parse.py` | 保留原有推理和代数消元操作 |
| AG1 原始推理 | `vendor/alphageometry1/dd.py`, `ar.py`, `graph.py`, `rules.txt`, `defs.txt` | 原有定理匹配、DD/AR 与图推导 |
| 双路构造编译 | `assets/ddar-compiler.js`, `assets/ddar-ag1.js` | Kernel 定义 → AG2 原生谓词；再将其形式化为 AG1 图前提 |
| 进度与原生溯源 | 两个 `vendor/alphageometry*/geomtool_v45_session.py` | 按阶段推进；AG1 调用原版 trace_back 输出 DAG |
| Worker | `assets/ddar-py.worker.js`, `assets/ddar-py.engine.worker.js` | 常驻 Pyodide、阶段间暂停和恢复、1 秒监督心跳 |
| 兼容 API | `assets/ddar.js` | `GeomDDAR.startSession(...)`、`GeomDDAR.infer(...)` |

**完整主线程进度与暂停的准确含义：** 新版按 AG2 的原版 search / transfer 阶段、AG1 的原版 BFS 层和 AR 转移推进；在阶段边界响应停止/继续，保留同一个 Python 对象。监督 Worker 每秒发送最近一次确认的计数和已历时，即使 Python 正在运行一个较长的阶段也不会虚报内部增量。因此这不是能在 Python 单次同步步骤**内部**强行抢占的真正实时暂停。

**目标模式**：将所选目标优先检查，每个阶段再检查；一旦原版状态确认该目标成立，退出后续阶段（但不能提前中断正在执行的 BFS 层或 AG2 单阶段）。**闭包模式**：直到原版推理阶段不再有更新，随后完成必要的候选查询。

## 形式前提 / 构造映射

36 种已纳入测试的 Kernel 构造操作包括：中点、垂足、重心、外心、垂心、内外心、平行四边形、正三角形、旋转（90° 和任意输入角）、位似、中心/轴对称、平移、线上/圆上点、三点外接圆点、线圆交点、平行垂直、垂直平分线、角平分线、切线、直径圆、线段半径圆、定半径圆、外接圆、圆心、实轴点以及多边形等；详见 `tests/v45_construction_cases.cjs`。其中若原版 AGDDAR 没有直接术语，编译器使用**内部见证点**与等价初始公理，例如重心添加两个中点，保证不把坐标近似值/雷达猜测塞入形式前提。

所有 36 个测试构造的 75 条生成谓词通过数值几何一致性审计，且对应形式谓词已在原生 AG2 `force_pred` 中成功初始化。数值审计仅用于**排查翻译错误**，不是严格正确性定理或对任意退化输入的证明。

已知构造边界：

- `secondIntersection` 所要求的“不同于第一个交点”需要不等式/分支选择信息，AG1/AG2 的基础正谓词不可精确表示；只能传入正确相交载体，不能宣称无损分支翻译。
- 任意负位似比例中的射线方向/负号、部分圆半径没有已命名圆上点时的点关系，在原生 AG1/AG2 的无向距离及模 180° 语言下可能丢失部分分支语义。
- `circleCenter` 用原版 AG2 的 `overlap` 表示同一中心点（而不是该内核不认识的 `eq`），AG1 使用**已有形式前提**合并别名；未有形式同一证明、却仅数值重合的点不能在 AG1 里静默合并。
- 使用自定义/插件构造、退化点、不可表示的角常量、以及原版内核所不支持的语义，结果会列入 `limitations` 或明确报错。**尚不能声称对一切可能的 GeomTool 图形都完美无损转换。**

## 证明链与原生事实

AG1 采用官方 `trace_back.get_logs()` 展开定理结果，输出节点编号、来源、所依赖的前提编号。已实测两个中点推出 `MN ∥ BC` 的链：4 个作图初始前提 → 2 个 `AR` 导出的 `midp` → 由 `r06` 定理导出平行。AG1 对仅为已知前提的目标使用“作图 / 初始前提”；原版未返回完整来源时**不会伪造**规则。

AG2 的公开 `elimination.py` 未存储充足的代数消元和几何推导前提，所以 v4.5 只能用原版 `check_pred` 确认目标，显示当前阶段来源；不能给出完全可靠的公理/AR 逐项 DAG，界面明确标出这一限制。**用户提出的“双内核每项结论都有完整可回溯证明”目前未全部实现。**

当前“所有已发现结论”指：从图内部直接列举的 AG1 DD 依赖、AG2 已建成的直线共线与圆共圆事实，加上逐项获得内核确认的有效候选。不是将两引擎内部所有代数等价式无限展开，因此不等于理论上**所有可表达关系**的完备枚举。候选发掘仍需要数值筛选，但它永远不进入形式前提。

## 回归测试

在已安装 Node.js 和 CPython（NumPy、SciPy）的机器上运行：

```shell
node --check assets/app.js
node --check assets/ddar.js
node --check assets/ddar-compiler.js
node --check assets/ddar-ag1.js
node --check assets/ddar-py.worker.js
node --check assets/ddar-py.engine.worker.js
python tests/v45_predicate_numeric_check.py
python tests/v45_native_sessions.py
python tests/v44_native_parity.py
```

当前环境已验证：36 个内核构造样本可接受；75 个形式前提数值一致；AG2 分阶段闭包可完结；AG1 目标提前结束及其带编号 DAG；原生 AG1/AG2 老回归通过，JavaScript 语法检查通过。

**尚未验证：** 此次运行环境无法下载 Pyodide CDN，也无法完成真实 HTTP 服务上的浏览器端到端验证，因此 Pyodide 版与本地 CPython 在所有输入上的逐项一致性、1 秒真实终端画面更新时序、长时间运行的 UI 性能尚**未通过实测验收**。本交付并非用户所要求的数学上/行为上完全复刻证书；代码含测试入口，适合继续做有网络环境的浏览器差分回归。
