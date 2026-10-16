# GeomTool v4.6 — 源码修改、回溯证书与验收记录

## 故障复现与根因

用户截图显示 `ValueError: AG1 coincident free points were not aliased`：v4.5 的 AG1 适配层在原生建图前显式拒绝同数值坐标的不同点，场景中只要出现无关虚拟见证的重合就中断。完全移除检查也不正确，原版 AG1 数值建图可能反复重试而无法结束。v4.6 处理策略为：

1. 编译器先去掉形式前提和查询都不需要的虚拟点；复用由同一个构造定义已确定的中点/外心；系统单位圆沿用 `p_1`，不制造 `@radius:c_unit`。
2. `GeomDDAR.prepare(..., {engine:'ag1'})` 对剩余非形式化坐标重合尝试原构造程序的**通用数值实现**：只在 Kernel 副本中小幅修改自由点，重新计算全部从属点/线/圆，直到构造前提数值有效且自由点彼此分离。原图未修改。任何派生节点失效的示意被拒绝。
3. 如果不能取得非退化示意，明确报错；从未用坐标重合推断公理级点身份，也不无限进入原生建图。

## 构造映射

- 原版 `assets/ddar-compiler.js` 仍将 Kernel 约束变成 `coll`, `cong`, `para`, `perp`, `eqangle`, `rconst`, `aconst`, `overlap`, `cyclic` 等原生前提；`assets/ddar-ag1.js` 进行 AG1 点名、别名、证明映射。
- 重心使用最多两个中点作为内部见证；如果玩家已经构造其中中点则直接复用。对应构造为 `mid`（共线与两半等长）以及两条中线共线和 2:1 比例约束。
- 虚拟点只在生成必要原生前提或载线时保留；不是额外显示给玩家的构造。
- 数值验证用来审计构造翻译，**不能**代替 AG 内核确认。第二交点的排除分支、负位似的射线定向等目前无法无损编码到两个核心的正谓词语言。

## 证明来源格式

点击结论或目标搜到时，前端请求对应 Python 引擎的 `proof_json(id)`。

- **AG1**：调用官方 `trace_back.get_logs`，将返回的前提与中间结论编号；来源为 `构造 / <操作名>`、`AR` 或官方 DD 定理编号（例如 `r06`），每步的 `premises: [id...]` 可反向追索。若原生回溯缺失，不会虚构证明。
- **AG2**：官方 `ddar.py` 的消元组件不跟踪所有 DD 规则前提。v4.6 在用户查看某已证结论时，按需使用输入的原始 AG2 形式构造前提重放原生闭包；从最初全体输入开始消减不必要输入，每轮都用**原生引擎**确认目标。最后记录留下的前提编号、结论和首先得到结论的实际运算阶段（例如 `search_similar` 或 `transfer_dist_arc_mul`）；AR 不展示内部消元细节。

注意：**AG2 证书是有效的“前提集 → 原版 DDAR 运算 → 目标”的宏步骤回放，不是逐项可检查的命名 DD 公理链**。若未来修改原版 AG2 的全部消元状态并为每条中间谓词存储父结论，才能得到与 AG1 严格等价的逐公理 DAG。`minimize` 最多执行 32 次试排除以保护界面，最终保留前提集合不保证数学上的最小集，但会再次原生复验。

`steps` 格式：

```json
{
  "available": true,
  "verified": true,
  "stage": "search_similar",
  "steps": [
    {"id": 1, "predicate": "coll M A B", "rule": "构造 / midpoint", "premises": []},
    {"id": 2, "predicate": "cong A M M B", "rule": "构造 / midpoint", "premises": []},
    {"id": 3, "predicate": "coll N A C", "rule": "构造 / midpoint", "premises": []},
    {"id": 4, "predicate": "cong A N N C", "rule": "构造 / midpoint", "premises": []},
    {"id": 5, "predicate": "para M N B C", "rule": "DD · AG2 相似三角形搜索", "premises": [1,2,3,4]}
  ]
}
```

## 验收方法

从解压目录运行：

```bash
node --check assets/ddar-compiler.js
node --check assets/ddar-ag1.js
node --check assets/ddar.js
node --check assets/app.js
node --check assets/ddar-py.engine.worker.js
python tests/v45_predicate_numeric_check.py
python tests/v45_native_sessions.py
python tests/v46_native_regression.py
```

本地核验的 ABC 中位线：A/B/C 为自由点，`M=midpoint(A,B)`、`N=midpoint(A,C)`，目标 `MN∥BC`。AG1 回溯为 4 个中点前提→2 个 `AR midp`→`r06`；AG2 为 4 个中点前提→原生 `search_similar` 重放确认。另复测 `reflectLine` 恰好生成与无关自由点同坐标的案例，通用数值示意可解除这种偶然退化。

## 尚未通过的验收

此次运行环境中 Chromium 请求本地 HTTP 页面被管理策略阻止，同时缺少访问 Pyodide CDN 的网络，故尚无实际网页操作 AG1/AG2 Python Worker 的完整端到端测试结果。原生 CPython 回归不能等同于 Pyodide/WebAssembly 浏览器验收。
