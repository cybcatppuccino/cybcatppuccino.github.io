# GeomTool v4.3 — DDAR 构造编译、目标发掘与形式证明查询

## 立即使用

直接打开 `index.html`（纯离线，无 Python/Numba/SciPy），或打开同次交付的 `GeomToolv4.3-Standalone.html`。展开右栏 **结论探知 · DDAR**，点击 **发掘可设置的待证目标**，筛选和选择目标，再点击 **全力搜索选中目标**。支持 `AB=CD`、`AB ⟂ CD`、`AB ∥ CD`、`∠ABC=∠DEF`、`∠(AB,CD)=∠(EF,GH)`（这里紧邻字母只适用于单字母点名），也支持不受数值筛选限制的空格命令，例如 `cong A B C D`、`perp A B C D`、`angle A B C D E F G H`、`eqangle A B C D E F`、`eqratio A B C D 2`、`angconst O P O Q 37`、`midp M A B`、`coll A B C`、`cyclic A B C D`。命令 `ddar goals`、`ddar prove [序号]` 可在控制台使用。

**重要：**目标发现以图形的当前数值配置为筛选器，列表中的条目一律是**待证猜想**，并非公理。即使数值满足，也只有独立 DDAR 规则查询返回 `proved` 才显示「已证」；`unknown` 不等于假。可以手动输入不在发现列表里的命题。自由点的数值巧合不会变成形式前提。

## v4.3 相比 v4.2 的实际升级

- 新的 **构造编译步骤**：以同一套内部谓词 `onl/onc/coll/midp/cong/eqratio/para/perp/eqangleLines/eqangle/cyclic` 描述更多 Kernel 构造；仅在内部引入临时定义对象，不落盘、不污染撤销历史。重心最多补 2 个中点；中垂线或直径圆最多补一个中点；轴对称、外点切线可补一个垂足；三点圆需要符号圆心，三点圆上点可补一个逻辑圆及符号圆心。辅助对象以 `@` 开头，不加入画板。简单双点、圆、交点等无需新增辅助点。
- 增加平移、中心对称、轴对称、位似、旋转（等半径、任意已知有向旋转角 `angconst`，以及 90°、60° 的长度/垂直特例）、平行四边形两组等边、内/旁心角平分线、三点圆圆心、切线接触点、同三点唯一圆、等腰底角、欧拉线与 2:1 比、内心与 A 旁心的轴关系等形式构造公理/几何定理。
- **角度表达**：`eqangleLines` 四个有向直线支撑，角度取模 180°。支持圆周角、角平分线产生的角度等式；通过方向奇偶类和角度等价类传播，可检测两对线夹角相等。
- **勾股型代数规则**：两个已证直角三角形中的两对对应边相等时，推得第三对边相等。仅使用已证垂直和长度约束，不读取几何测量值作证明。
- **目标系统**：`assets/ddar-goals.js` 的候选枚举与 `GeomDDAR.infer(snapshot,{goal})` 的独立形式证明查询分离；Worker 异步执行，失败回退主线程；移动或编辑作图立即清空旧目标证明。
- 更新 `assets/ddar.worker.js`、`assets/app.js`、`assets/app.css`、`index.html`，并保留原有 Radar、Kernel、Pretty、题库、辅助构造、导入导出与可撤销操作。导出的 `geomtool.ddar/v4.3` 数据包含目标、证明状态、形式事实和辅助构造。

### 当前 Kernel 方法映射审计

| 方法类别 | 内部表达 / 辅助 |
|---|---|
| free/referencePoint/real、line/segment/ray、polygon | 自由对象不引入伪约束；实轴点带实轴 incidence；多边形边由 Kernel 创建真实 segment |
| onLine/onCircle/onCircumcircle、intersection/secondIntersection | 有效载体的 incidence；三点圆辅助逻辑圆 |
| midpoint/centroid、foot/foot3、perpBisector | 中点、共线、垂直、长度比例；需要时临时中点或垂足 |
| parallel/parallel3、perpendicular/perpendicular3、angleBisector/outerAngleBisector | 平行/垂直/有向角等式 |
| circle/circleRadius/circleBySegment/circumcircle/diameterCircle/circleCenter | 圆 incidence、同圆等半径、半径转移、已定义圆心或直径中点 |
| circumcenter/orthocenter/incenter/excenter | 等半径、三高、内/外角平分关系和中心定理 |
| tangent/tangentFrom | 切线-半径垂直；外点切线用一个虚拟接触垂足 |
| parallelogram/equilateral/reflectPoint/reflectLine/translate | 等长、平行、共线与中点约束 |
| rotate/rotateLeft/rotateRight/homothety | 等半径、给定旋转角的已实现特殊情形、比例和共线 |

**限制必须区分：**这是更强的纯 JavaScript **DDAR 子集**，尚未达到上传 Python `GeomTool-main/geomtool/alphageometry` 或 Newclid 的全部规则与代数推理能力；例如完整的有向角线性模 180° 方程组、所有相似/全等三角形的组合匹配、一般距离二次式与比值消元、任意给定角度参与一般线性角方程的完整消元、全部多步自动辅助搜寻与形式证明最小化仍未等价移植。因此不能声称与其“完美复刻”或证明完备。归并等价类时，数字坐标只用于排除退化；形式事实的来源是构造语义或已经成立的关系。

### 性能与不完备标志

- 默认 DDAR `maxFacts=16000`、`maxRounds=11`（模块 API 默认为 12000/10）；可设上限约束。若达到安全上限显示截断，不把「尚未证明」判断成「假」。方向类用奇偶并查集，长度用带权并查集，角度用等价类，候选线段距离用排序窗口。
- 数值命题枚举缺省展示上限为 **2200 条**，最多扫描 24 个命名点和 48 个候选支撑线；界面会标注到达上限。该列表不保证覆盖所有可能的命题——通过手动输入可查询更多。直角三角形匹配最多使用 72 个已证直角三角形；同圆四点显式枚举每圆最多 14 个命名点。这些上限是性能保护，不是数学定理。
- `eqangleLines` 是有向角 **mod 180°**，不同于普通非负内角度数；带方向的等式必须用正确的直线顺序。
- 证据链在少数经过并查集传递的查询中显示相关支持事实，而非最短、可在独立证明检查器逐行复演的完全演绎路径。JS 的工程规则仍需专业复核。

## 测试与 Python 基准说明

- `node tests/v43_randomized_soundness.cjs`：13 组随机组合构造，对 4187 条已证几何结论实施独立的数值一致性审计（不把测量结果反馈给推理器）。
- `node tests/v43_ddar.test.cjs`：构造编译、临时辅助点、角度、目标、欧拉线与内/旁心关系、独立形式证明、性能上限。
- `python tests/v43_browser_smoke.py`：真实 Chromium 网页目标发掘、输入、选择、证明、修改图形失效、未证不误报。
- `node tests/v42_ddar.test.cjs`、`node tests/v41_dummy_50.cjs`、`node tests/bootstrap.test.cjs`：兼容性回归。
- 上传的 Python 参考源码已在临时解压目录运行了 `orthocenter` 内置 DDAR 小题；由于无图形界面的环境不能加载原文件硬编码的 `TkAgg`，**仅在临时测试副本**把 matplotlib 后端替换为 `Agg`。运行结果是题目条件直接支持结论。`euler` 例题在受限时间窗口未完成全程，故没有基于该案例宣称网页与 Python 相同的题库成功率。源码及 benchmark 不作为网页运行时依赖。

## APIs

```js
const theorem = GeomDDAR.infer(GeomTool.kernel.snapshot(), {
  goal: {kind:'eqangleLines', args:['P:A|B','P:C|D','P:E|F','P:G|H']},
  maxFacts:16000, maxRounds:11
});
console.log(theorem.goal.status, theorem.goal.evidence);
const candidates = GeomDDARGoals.discover(GeomTool.kernel.snapshot());
const manual = GeomDDARGoals.parseGoal('AB ⟂ CD', GeomTool.kernel.snapshot());
```

此版的主要收益是更统一、可扩展的构造语言和真正可选的目标查询，而非以数值测量或静态预置答案假冒定理证明。
