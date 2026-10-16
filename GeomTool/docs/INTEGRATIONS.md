# 推理器与符号服务扩展协议（规划边界）

当前文件记录计划中的服务契约；**静态 v1 不启动、也不内置** GeomTool / Newclid Python 进程。

## 1. 证明桥接载荷

网页「专家证明 → 复制 DD+AR 对接数据」得到以下 JSON（`geomtool.proof-bridge/v1`）：

```json
{
  "schema":"geomtool.proof-bridge/v1",
  "source":"GeomToolv1 static",
  "graph":{"schema":"geomtool.ir/v1","protocol":"geomtool-kernel/1.0","nodes":[]},
  "goal":null,
  "computedFacts":[],
  "engine":"unconfigured",
  "note":"This is a bridge payload, not native Newclid/GeomTool problem input."
}
```

本地 `computedFacts` 来源是**构造的定义性事实**（例如 midpoint 的「M 是 AB 的中点」）；`proof-goal` 仅列出有定义来源的目标。无需后台即可显示这些已知事实，但**不包含从任意前提推导新结论的搜索**。

下一阶段建议：

- `POST /api/v1/proof/check` 接收上述 payload；服务端检查导入的构造、映射的假设、构造规则和非退化条件；无法映射的 op 明确给出 unsupported，不可静默忽略。
- 在适配器中将构造 DAG 转换为 GeomTool graph / Newclid JGEX 可处理的形式。需要逐种构造做映射，**当前 JSON 并非原生可直接提交到 Newclid 的输入**。
- `POST /api/v1/proof/search` 返回 `status: proved | unknown | disproved | unsupported`、`proof_dag`、`conditions`、`engine`, `time_ms`, `graph_revision`。对于 `unknown` 不能暗示目标为假；数值反例应带完整可重现的构造参数及容差。
- 辅助点搜索器单独注册 `candidateConstructors[]`，维护目标影响评分与搜索预算；生成的辅助点也是 IR 构造对象，必须满足存在性和非退化条件。
- 返回结果必须绑定 graph revision / 内容哈希。若画板变化，页面应放弃对应旧结果。

## 2. 复数法服务

建议 `POST /api/v1/complex/evaluate` 输入 IR 和 `targets:[ID]`，返回：

```json
{
  "schema":"geomtool.complex-results/v1",
  "status":"ok",
  "points":[{
    "id":"M", "domain":"rational", "expression":"(a+b)/2",
    "latex":"\\frac{a+b}{2}", "constraints":["A != B"],
    "certificate":null
  }]
}
```

注意：这只是建议的数据结构，不代表客户端现在产生了精确的符号表达式。未来计算内核可以使用 SymPy/独立 CAS，针对实轴点和单位圆点采用专门的共轭化简约束；一般直线与圆、两圆求交不保证产生有理函数。

## 3. 保持计算安全

- 服务端不相信来自浏览器的 `valid`、数值关系、`computedFacts`、代数公式或可视属性。所有推理假设应从输入的构造 DAG 和可信规则中重新构建。
- 显示数据不等于证明。只有完成形式化/可复核步骤的结论才可标为 `proved`。
- 显式限制输入文件体积、构造数量、表达式规模、证明节点数、CPU 时间；证明器使用独立 worker 进程并实现硬超时。
- 避免用可变全局状态承载多个用户问题；隔离每个证明任务与回放步骤。

## 4. 可复用资源映射

| 来源 | v1 | 服务端下一阶段 |
| --- | --- | --- |
| Geommunity Blueprint | `vendor/bs-relations.js` 数值关系雷达已集成 | 使用其 BS 画图搜索器作为可选服务或 Worker |
| GeomTool-main | 核心几何构造概念用来扩展注册表 | 适配 `geomtool/core/dd`、`core/ar`、自然语言模块 |
| Newclid-main | 核对 JGEX 常用构造清单 | 接入 DD+AR、heuristics，需要兼容 JGEX 模型与环境 |
| Mathematica 复数笔记本 | 本版未依赖笔记本运行时 | 提取恒等式/案例，建立 SymPy 代数测试题 |
