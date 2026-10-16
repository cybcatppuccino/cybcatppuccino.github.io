# GeomTool Kernel Protocol v1.0

目标：全部绘制工具、控制台、关系发现和未来证明器**共享同一图形语义**，不依赖 Canvas 或 DOM。

## 1. 分层与边界

```text
UI / CLI / File import
       ↓ execute(Command)
Geometry Kernel (geometry IR, unique IDs, construction DAG)
       ↓ rebuild()           ↘ relationSnapshot()
Numeric geometry snapshots    Blueprint bs-relations.js
       ↓                        ↓
Canvas renderer             numeric conjectures
       └─────────→ future Symbolic / DD+AR adapter
```

所有用户编辑最终都进入 `new GeomKernel.Kernel().execute(command)`。新增操作通过 `registerMethod()` 扩展。`kernel.snapshot()` 返回不可依赖 DOM 的显示快照。持久化必须使用 `kernel.serialize()`，不能把 `geom` 缓存写进文件。

## 2. 数据结构

```json
{
  "schema": "geomtool.ir/v1",
  "protocol": "geomtool-kernel/1.0",
  "nodes": [
    {"id":"A", "label":"A", "op":"free", "inputs":[], "params":{"x":0,"y":0},"style":{"visible":true,"dashed":false}},
    {"id":"B", "label":"B", "op":"free", "inputs":[], "params":{"x":4,"y":0},"style":{"visible":true,"dashed":false}},
    {"id":"M", "label":"M", "op":"midpoint", "inputs":["A","B"], "params":{},"style":{"visible":true,"dashed":false}}
  ]
}
```

对象字段：

| 字段 | 含义与约束 |
| --- | --- |
| `id` | 永久内部 ID，ASCII 字母开头，只含英文字母/数字/下划线。父子关系只存此 ID。 |
| `label` | 可重命名的显示名称。不能与其他对象的 ID 或名称冲突。 |
| `op` | 对应注册构造方法的名称。每种方法定义输入类型及计算规则。 |
| `inputs` | 已建立对象 ID 数组；顺序由构造规则规定。只允许引用当前节点之前已有的对象，避免依赖环。 |
| `params` | 纯 JSON 数值参数：如自由点 x/y、圆上的 `theta` (弧度)、线上的 `t`、交点分支 `branch`。 |
| `style` | `visible`、`dashed`、可选 `color`；不影响对象是否具有几何意义。 |
| `geom` | **运行时只读数值缓存**，不序列化；根据 `inputs` 拓扑重算得到。 |
| `valid` | 运行时布尔状态；拖动产生退化时节点及其依赖项显示失效，拓扑顺序重新计算。 |

## 3. 核心命令

```javascript
const graph = new GeomKernel.Kernel();
const a = graph.execute({ op: 'free', id: 'A', params: {x:0, y:0} });
graph.execute({ op: 'free', id: 'B', params: {x:4, y:0} });
graph.execute({ op: 'midpoint', id: 'M', inputs: ['A','B'] });
console.log(graph.geom('M'));          // {kind:'point',x:2,y:0}
graph.execute({op:'move',id:'B',x:6,y:0});
console.log(graph.geom('M').x);       // 3 (自动重算)
graph.execute({op:'undo'});
graph.execute({op:'style',id:'M',style:{visible:false}});
graph.execute({op:'rename',id:'M',label:'中点 M'});
graph.execute({op:'delete',id:'A'});  // 删除 A 及全部依赖对象
graph.execute({op:'clear'});          // 清空所有构造
```

全部修改命令写入有上限的 Undo 历史；`previewMove(id,x,y)` 是为指针拖动提供的低成本中间状态，不直接建立历史，拖动开始先取得 `before=serialize()`，完成时 `checkpoint(before)`。

`graph.subscribe(({type,revision}) => ...)` 在状态变化时触发；拖动预览会发 `preview`。当图形更新，应使基于先前 revision 的雷达或证明结果失效，避免显示过时结论。

`graph.list()` 返回构造节点，`graph.node(id)` 返回节点，`graph.snapshot()` 返回渲染快照，`graph.relationSnapshot()` 返回 Blueprint 原型所需对象。`load(doc)` 执行完整导入验证并原子替换图结构，出错时原图保留。`serialize()` 返回可直接存盘的纯 JSON。

## 4. 几何数据 / 坐标与分支约定

- 数学坐标：x 轴向右，y 轴向上。画布坐标在渲染层翻转 y。单位为“逻辑长度”，不是像素。
- `point`: `{kind:'point',x,y}`。
- `line`: `{kind:'line',p:{x,y},d:{x,y},extent:'line'|'segment'|'ray'}`；`p+t*d` 给出支撑线，射线要求 t≥0，线段要求 0≤t≤1。
- `circle`: `{kind:'circle',c:{x,y},r}`，半径必须严格大于 0。
- `polygon`: `{kind:'polygon',points:[{x,y},...]}`。
- `intersection`: 0/1 分支；所有有效交点首先按 `x,y` 升序排列；`branch=0` 取第一个，`branch=1` 取第二个。**分支是离散选择**，参数变化可能造成交点合并、消失或更换排序，在证明器中必须显式处理非退化、存在性与定向条件。
- `onLine` 参数 t 是沿父载线方向的标量；`onCircle` 参数 theta 以弧度计算。
- 带几何限制的点在用户拖动时会投影/钳制回合法位置。一般法线投影允许投向被选线段的延长线，属于几何构造而不等同于“在线段内”。
- 若底层运算检测到零长度、共线导致三点圆无法定义、无实交点、内点求实切线等情况，创建会失败，移动后的受影响派生对象被标记为 invalid。

**准确性限定**：v1 使用 JavaScript 双精度浮点数与容差，目标是动态画板。没有精确谓词、符号复数和完整可认证的定理推导。不可仅凭数值相等宣称证明完成。

## 5. 构造方法注册表

`GeomKernel.METHODS` 是当前注册表，包括 43 个操作。每个项包含 `kind`、`inputs`、`group`、`label`、`syntax`，以及部分操作的 `rational` 分类标签。分类是后续符号化引擎的开发提示，不构成对任意参数/交点分支的数学保证。

可通过 `GeomKernel.registerMethod(name,spec,evaluator)` 扩展：

```javascript
GeomKernel.registerMethod(
  'third',
  {kind:'point',inputs:['point','point'],group:'point',label:'三分点'},
  ([a,b], params, h) => ({kind:'point', ...h.add(a,h.mul(h.sub(b,a),1/3))})
);
const graph = new GeomKernel.Kernel();
graph.execute({op:'free',id:'A',params:{x:0,y:0}});
graph.execute({op:'free',id:'B',params:{x:3,y:0}});
graph.execute({op:'third',id:'T',inputs:['A','B']});
```

UI 中可再通过 `GeomTool.registerTool('point',{id:'third',name:'三分点'})` 添加入口。独立工程若要添加较复杂构造，可注册多个普通方法作为宏，或者在新的 `op` evaluator 内计算相应几何。

运行时注册扩展不自动生成严格证明公理。每条新的演绎规则应在后端声明其正确性及依赖条件。

## 6. 命令控制台

`GeomKernel.runText(graph, 'midpoint M A B')` 是人类可读命令的解释器，调用同一套 `execute()`。`help` 返回内置命令语法。部分坐标/角度命令使用度数输入并转换为存储的弧度：

```text
point A 0 0
point B 2 0
circle c A B
oncircle P c 90
online M l 0.5
intersect X l c 1
secondintersect Y X l c
rotate R B A 60
```

命令与 GUI 共享数值模型，但不提供任意 JavaScript `eval`。对后来运行环境中的扩展命令，默认可用 `op id input1 input2 ...` 格式（数值扩展参数需自行扩展解释器）。

## 7. 版本演进建议

IR 中下一阶段可增加但**不应冒充 v1 已实现**的字段：`assumptions` (非退化与构造约束)、`symbolic` (数学表达式 DAG)、`proofRef` (可检验证明来源)、`domain` (complex-real / unit / general)、`source` (自由点或由原始题意给出)。未来迁移应有显式 `schema` 版本，避免直接改变当前读写语义。
