# GeomTool v3.5 · 控制台命令、输入行为与边界

## 与 v3.4 的兼容

直接修改用户提供的 v3.4 静态网页。图形文件继续使用 `geomtool.ir/v1` 内核协议以及 `geomtool.document/v1` 外层格式；参考对象仍使用固定 `GTREF_...` IDs，用户数据可以沿用。仅改变前端呈现和可操作入口，原 v3.4 关系推导与数值扫描模块未改写。网页离线可用。

## 角度与位似

左侧“几何变换”选择 **旋转点** 或 **位似点** 后，激活工具卡片内出现输入框。旋转有 `deg` 与 `rad` 两种真实单位，不再以“π 倍数”代替弧度制。度数输入如 `60`、`180/3`；弧度输入如 `π`、`pi`、`2π/3`、`2pi/3`、`π/2`、`-sqrt(2)`，以及 `2*pi/3`、`2×π÷3`。通过输入框旁的 π 按钮在光标处插入 `π`。单位切换保留角度的数学含义（例如 `120°` 转换成 `2π/3`；复杂非有理角度用等价表达式展示）。当公式无效时拒绝单位切换并保留原始输入，创建前再次校验参数。

位似倍数支持 `2`、`-sqrt(2)`、`5/7` 等表达式。算式只允许明确的数字、括号、π、e、平方根、幂和基本四则运算，拒绝脚本执行与无限数字；不保证能解释任意高级 CAS 符号表达式。

## 对象与参考轴

右侧“作图关系 → 对象”保留所有系统/普通对象，隐藏对象标记 `◌`，可点击进入“属性”。“属性”可操作每个对象的显示/隐藏，以及普通对象的更名与删除；系统节点依旧不能改名或删除。虚线和十六进制颜色可通过控制台 `style` / `color` 命令单独修改。六个系统对象名称分别是：

| 内核 ID | 标签 | 参考系前置开关 |
| --- | --- | --- |
| `GTREF_origin` | `0` | 坐标轴 或 复平面 |
| `GTREF_real` | `Re` | 坐标轴 或 复平面 |
| `GTREF_imag` | `Im` | 坐标轴 或 复平面 |
| `GTREF_one` | `1` | 复平面（同时显示坐标轴） |
| `GTREF_i` | `i` | 复平面（同时显示坐标轴） |
| `GTREF_unit` | `单位圆` | 复平面（同时显示坐标轴） |

对象实际出现在画布需同时满足 **内核有效 + 对象独立 visible + 参考系前置开关允许**。关闭坐标轴时旧行为仍然成立：如果复平面同时开启，关闭坐标轴会自动关闭复平面；各参考对象不会显示。重开后各对象之前独立保存的 visible 状态仍生效。

## 控制台示例

浏览器左侧终端图标可打开控制台；输入 `help` 打开详细、可折叠的独立手册（含全部内核构造命令），在搜索框中输入 `旋转`、`关系`、`隐藏`、`export` 等可逐项过滤。按帮助条目的「填入」可把示例复制到命令框，输入时也有实时提示。上下箭头浏览执行历史。控制台命令不区分英文关键字大小写（几何对象 ID/标签保持原样），对象名含空格时可用引号。

```text
point O 0 0
point A 2 0
rotate R A O 2π/3 rad
homothety H A O -sqrt(2)
hide A
A
show A
style A dashed on
settings axes on
hide GTREF_real
select GTREF_origin
settings precision 10
settings palette emerald
settings grid on
settings theme dark
settings leftWidth 290
zoom 135%
center 0 0
fit
tool rotate
unit rad
param π/3
step A
step O
beautify 75
pretty progress 50
relations kinds all
relations hidden on
relations perturb off
relations scan
relations search A B
relations sort kind
relations select 1
proof list
proof check
library search Euler
library list
library select 1
export file
export json
import file
copy graph
console history
console clear
```

其他可用类别和命令包括：

- **对象 / 内核**：`list hidden`、`list visible`、`select 对象`、`toggle 对象`、`move A x y`、`rename A 新标签`、`delete A`、`undo`、`redo`、`clear`、`demo`、`methods`、`guide 旋转工具ID`。几何构造覆盖 Kernel 注册的所有非系统构造（中点、圆心、交点、旁心、足点、切线、多边形等），详情可通过 `help` 搜索。
- **完整工作区设置**：`settings grid|labels|axes|complex|snap|wheel on|off`；`settings precision 0..15`、`settings palette blue|graphite|emerald`、`settings theme light|dark`、`settings leftWidth|rightWidth 数值`、`settings leftCollapsed|rightCollapsed on|off`，`settings list` 查看设置。
- **画布工具输入**：`group point|line|circle|construct|transform`、`tool rotate|homothety|line|...`、`step 对象`、`finish`、`cancel`、`smart x y`、`click x y`。这里的 `x y` 是世界坐标，而 `pan dx dy` 是像素偏移。
- **关系系统**：`relations kinds [类别ID on|off]`、`relations kinds none|all`、`relations scan`、`relations hidden on|off`、`relations perturb on|off`、`relations search 关键字`、`relations select 序号`、`relations more|clear|list`。
- **专家证明与资料库**：`proof list|select|check|export`、`library search|filter|list|select|load|more`（载入覆盖时仍执行原程序确认）。
- **文件与通用界面**：`import file` 打开本地文件选择器；`import json {单行 JSON}` 将控制台数据导入；`export file|json|proof`、`doc 名称`、`copy graph|proof`、`ui click 控件ID`、`ui set 控件ID 值`、`shortcuts`、`panel details|objects|pretty|relations|complex|proof|library|left|right|settings`。

`ui click` / `ui set` 是执行带 DOM ID 控件的通用补充入口；`import file` 受浏览器文件沙盒限制，不能通过输入任意本地路径读取用户文件，必须由浏览器原生文件选择器确认。关系发掘仍为数值候选，证明区仍是本地构造定义校验，控制台不会将它们伪装为完整形式证明。

## 测试与已知边界

- 新增 `node tests/v35_console_expression.test.cjs`，验证 Unicode π、隐式乘法、角度单位等值转换、两种构造、隐藏对象样式及 Help 命令覆盖。
- 可选 `python tests/v35_browser_smoke.py`，在 Chromium + Playwright 下对真实 DOM 进行交互测试，包括左侧旋转输入框、π 按钮、隐藏对象属性、独立系统轴状态、命令手册、设置、关系扫描选项、题库、导入导出。
- 原附带 11 个旧测试文件可全部复跑通过；另有 `tests/v33_relations.test.cjs` 的 `correspondence metadata preserved` 断言在**用户上传的原版 v3.4** 上就已失败。v3.5 未改关系计算模块，也未宣称修复这个旧失败。详细情况请参阅原有 v3.3 测试与关系引擎说明。
- 浏览器功能测试是操作烟雾验证，不是所有 UI 组合路径、全部题库或形式推理的穷举证明。
