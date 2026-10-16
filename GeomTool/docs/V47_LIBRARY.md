# GeomTool v4.7 — 统一几何证明资料库

## 变化范围

本次仅替换内置几何资料库的数据、离线编译脚本、转换解释器和该资料库的少量现有 UI 文案。DDAR、原有作图工具、证明系统及辅助构造模块不改。没有将题库目标自动提交给 DDAR。v4.6 旧题库、分类和内容从运行数据中整体替换，不保留 Blueprint/GeoGebra/Mathematica 等旧条目。

## 来源和转换规则

| 原始来源 | 题目数 | 在 ZIP 中的位置 | 数据类型 |
|---|---:|---|---|
| AlphaGeometry 1 | 265 | `examples.txt`, `jgex_ag_231.txt`, `imo_ag_30.txt` | 高级构造→目标 |
| AlphaGeometry 2 | 26 | `test.py` 的 `problems_without_aux`、`problems_with_aux` | 逐点原始坐标 + 几何谓词→目标 |
| Newclid | 879 | `newclid/problems_datasets/*.txt`（11 份） | 高级构造→目标 |
| **总计** | **1,170** | | |

每题独立保留原始程序串、原始文件路径、标题、原始点名、坐标（若有）、有序构造步骤、辅助构造段（若有）、多个目标（若有）。**源数据没有提供的绝对坐标不会写进题库**。低级 AG2 格式只含点样本坐标和原始谓词，没有可唯一恢复的尺规构造顺序；因此加载为坐标实例，不伪称为依赖式作图。

`tools/build_library.py` 使用三个上传的原始 ZIP 直接机械解析生成 `assets/library-data.js`，不调用网络。示例：

```sh
python tools/build_library.py \
  --ag1 '/your/path/alphageometry-main(1).zip' \
  --ag2 '/your/path/alphageometry2-main(1).zip' \
  --newclid '/your/path/Newclid-main.zip'
```

AG1/Newclid 的高层步骤：自由点使用源位置（存在时）或确定性采样坐标，依赖点按有序约束调用既有 Kernel 运算。需要时临时创建线/圆作为载体，记录依赖关系；额外实现定角轨迹、定长圆、阿波罗尼斯圆、双圆公切线等操作，仍由 Kernel 注册、序列化和重算。一个带多个约束的构造语句只在**全部约束可保真实现**时才成功；否则原子回滚并记录“未转换”。在没有源坐标的题目加载后调用已有 Pretty 工具美化布局；有原坐标时禁止 Pretty 改写。

## 目标格式

每个条目在原界面详细面板中显示独立的 `AlphaGeometry 1` 和 `AlphaGeometry 2` 文本，保持原点名：例如 `para e f b c` 或 `cong a b c d`。两个内核共用的大部分谓词名字相同，所以文本可能一致。对 AG2 原始内核没有对应单谓词目标类型的条目（如 `simtri`、`midp`、`sameclock`），明确标为“AG2 原生 DDAR 不支持”；文本保留而不捏造一种有效的 AG2 原生定理。多目标问题逐项显示。

## 回归验收（代码级）

- 1,170 条题目逐项读取，无解析失败；26 条 AG2 坐标记录保持原始数值。
- 原生 Node Kernel 逐题建图：**1,019 条完整步骤转换，151 条包含尚不能完全保真映射的步骤**；其中 12 条尚无可画图对象。记录全部保留，**不把部分转换误报为完整转换**。
- 7,616/8,063 条原始高级构造条款可以直接转换或作为坐标记录载入（含 26 份 AG2 题目的点，口径参考 `tests/v47_library.test.cjs` 输出）。
- 5,307 个已转换构造关系的额外数值检验（共线、中点、圆、定角、平行、垂直、定距等）通过，**并不意味着剩余未支持的复杂构造也已成功**。
- Newclid `r06` (`a b c` 三个自由点，`e=midpoint a b`, `f=midpoint a c`, `para e f b c`) 已通过节点类型及目标文本检查。
- Chromium/Playwright 的 `http://127.0.0.1` 和 `file://` 访问在当前测试沙盒被管理员策略阻断，因此**浏览器端到端测试未通过/未执行**。在用户正常浏览器中仍应执行现场回归。

验证：

```sh
node tests/v47_library.test.cjs
node tests/v47_numeric.test.cjs
python -m http.server 8000
```

浏览器打开 `http://localhost:8000`。打开「几何资料库」，按来源筛选或搜索 `r06` / `2000_p1`，选中查看目标，按「载入构造并自动 Pretty」（AG2 为按原坐标载入）。已知不保真题目的加载提示明确列出未能翻译的步骤，不会临时生成虚假自由点。
