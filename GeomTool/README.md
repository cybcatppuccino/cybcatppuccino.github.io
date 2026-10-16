# GeomTool v5.4 — 高精度约束 Pretty / AG1 / AG2 / Newclid 统一几何工作台

在 v4.6 基础上完全替换旧的内置「几何资料库」，其余作图、关系探知和 AlphaGeometry DDAR 模块保持原功能，不对接新增题库目标。前端仍是现有的几何资料库面板。

## 运行

在本目录执行 `python -m http.server 8000`，浏览器打开 `http://localhost:8000/`。不使用 `file://` 测试 DDAR Python Worker。资料库是纯离线编译的 JS 数据，浏览题目/作图不需要 DDAR 或互联网。

内置 1,170 份源码题目：AlphaGeometry 1（265）、AlphaGeometry 2（26）、Newclid（879）。每题展示原始构造/谓词以及两个 AlphaGeometry 版本的目标描述。保留源数据提供的点坐标；无坐标构造通过 GeomTool Kernel 构造并使用 Pretty 整理。

本版在 Node Kernel 审计下 **1,019 份全部作图步骤可转换；151 份只能部分转换，其中 12 份无可执行图形**。全部 1,170 份题目文字、坐标和目标均可检索，不等同于所有 1,170 份都已完成语义作图。详见 `docs/V47_LIBRARY.md`。

复现数据：`python tools/build_library.py --ag1 <AG1.zip> --ag2 <AG2.zip> --newclid <Newclid.zip>`。
测试：`node tests/v47_library.test.cjs` 和 `node tests/v47_numeric.test.cjs`。


**v5 更新：** 关系发掘恢复；题库作图增加纯显示线、圆和角标记；已证命题同步标记并支持点击单条证明步骤高亮。详见 `docs/V5_CHANGES.md`。本次按要求未运行额外测试。

**v5.1 更新：** 将构造依赖转换为 AG2 原生坐标和形式前提，新增原样提交查看器与 JSONL 推理日志；对原生 DD/AR 插入可选审计钩子；移除旧交点控件并修正外角平分线示意。详见 `docs/V51_AG2_ARCHITECTURE.md`。

**v5.2 更新：** 在点/几何构造中新增“相似点”，在“线与射线”中新增“等角线”；编译对应的 AG2 有向角与三组非退化边比例；新增网页端完整推理日志查看器；AG2 消元方程记录父编号、相似定理应用记录直接符号前提，完成可读目标溯源。其他 DD 结构的不能严格定位的条件采用明确标注的宏级来源。详见 `docs/V52_CONSTRUCTIONS_AND_PROVENANCE.md`。

**v5.3 更新：** Pretty 现支持带提示的条件输入框（回车 / 确认、语法检查、列表管理），优先满足几何条件的阻尼数值下降，满足约束后进行近距离美观优化。支持共线、共圆、距离、夹角、点在线/圆上、圆线相切/正交、全等/相似、面积与代数等式及若干 AG1/AG2 原生谓词。数值满足并可原生表达的用户条件会以 `USER-PRETTY` 来源写入 AG2 假设，其余项明确报告不可编译。条件随工作区和 `.geom.json` 保存。详细说明见 `docs/V53_PRETTY_CONSTRAINTS.md`；回归运行 `node tests/v53_constraints.test.cjs`。


**v5.4 更新：** 条件输入框右侧改为可展开的 `?` 规则手册（7 条完整规则及例子），去除实时语法建议；加强量纲检查和角度单位写法。Pretty 把约束精度目标提高为 `5e-13` 归一化残差，并在合适构型中再进行高精度 Newton 抛光（部分实测约 `1e-15`）；共圆的数值残差更加稳定。关系检索微扰仍默认关闭；AG1 遇到重合的数值点不再默认生成扰动替代图，需显式启用。新测试 `tests/v54_precision.test.cjs`，详细规则、数值数据与扰动审计见 `docs/V54_PRETTY_RULES_PRECISION_AND_PERTURBATION_AUDIT.md`。
