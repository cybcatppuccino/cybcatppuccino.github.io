# RIES v11.9.2 hypergeom 数据库合并、排除与去重报告

## 输入
- 旧库：v11.5.2 三段 hypergeom chunk，原始行数 109738。
- 新数据：`data.zip` / `hyper2f1_grid_v2_blocks_v2_json`，JSON block 610 个，记录 30442 条，其中 ok 29322 条，失败/跳过 1120 条。

## 有理数排除
- 方法：所有数先四舍五入保留 20 位小数，再用连分数展开寻找分母 ≤ 1000000 的有理数；若与 20 位小数值的绝对误差 ≤ 5E-21，判定为有理数并排除。
- 行级 H 值排除：593 行；其中旧库 45 行，新 data.zip 548 行。
- 实数搜索投影排除：5452 个 scalar 投影（component 0=H，1=Re(H)，2=Im(H)；分布 {'1': 5409, '2': 43}）。

## 去重
- 行级复数 H 去重：发现重复组 673 个，移除重复行 2297 条；保留规则为参数/系数绝对值求和较小者优先，其次 complexity、stage。
- 实数搜索投影去重：发现重复组 3916 个，移除重复 scalar 投影 3991 个。

## 输出
- 最终 H 行数：136170。
- 最终实数搜索 scalar 数量（含 H、Re(H)、Im(H)）：205890。
- 最终复数搜索 H 数量：136170。
- tierCounts：{'1': 29631, '2': 36403, '3': 70136}。
- pFqCounts：{'0F1': 72, '1F0': 21, '1F1': 96, '1F2': 72, '2F1': 27767, '2F2': 72, '2F3': 72, '3F2': 1851, '3F3': 72, '3F4': 24, '4F3': 17472, '4F4': 24, '5F4': 18931, '5F5': 24, '5F6': 24, '6F5': 6173, '7F6': 6737, '8F7': 56666}。

## 资产
- assets/ries-hypdata-v11_9_2-level4.js：level 4 / stage 1，H rows 29631，real scalar rows 36169，complex rows 29631，bytes 3722500。
- assets/ries-hypdata-v11_9_2-level5.js：level 5 / stage 2，H rows 36403，real scalar rows 54493，complex rows 36403，bytes 5916649。
- assets/ries-hypdata-v11_9_2-level6.js：level 6 / stage 3，H rows 70136，real scalar rows 115228，complex rows 70136，bytes 13860985。

## 搜索与 LaTeX
- 新 data.zip 的 2F1 记录放入 level4 chunk，因此 level4/5/6 的累计加载都会使用。
- 实数搜索新增 `realCompB64`：0 表示 H 本身，1 表示 `\operatorname{Re}(H)`，2 表示 `\operatorname{Im}(H)`；前端显示时会生成 `\operatorname{Re}` / `\operatorname{Im}` 的 LaTeX。
- 复数目标搜索继续使用完整 H = Re(H)+i Im(H) 表。
