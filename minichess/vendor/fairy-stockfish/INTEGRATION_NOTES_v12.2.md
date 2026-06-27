# fairy-stockfish-nnue.wasm 1.1.11 接入 v12.2 备注

## 已下载文件

本目录来自 npm 包 `fairy-stockfish-nnue.wasm@1.1.11`，包含：

- `package.json`
- `stockfish.js`
- `stockfish.wasm`
- `stockfish.worker.js`
- `uci.js`

`SHA256SUMS.txt` 是本次下载后的校验值。

## v12.2 当前状态

v12.2 已经有空目录 `vendor/fairy-stockfish/`，并且棋盘、SAN、UCI、FEN 已经统一为 5×5 Gardner/Minichess 的标准 A1-E5 坐标。Fairy-Stockfish 的内置 `gardner` 变体可直接接受 v12.2 的 compact FEN，例如：

```text
rnbqk/ppppp/5/PPPPP/RNBQK w - - 0 1
```

本地验证命令流：

```text
uci
setoption name UCI_Variant value gardner
isready
position fen rnbqk/ppppp/5/PPPPP/RNBQK w - - 0 1
go depth 3
```

实测可返回合法 UCI，例如 `bestmove e2e3`。

## 推荐接入方案

1. 把这 5 个文件放到站点同源静态目录：

```text
vendor/fairy-stockfish/package.json
vendor/fairy-stockfish/stockfish.js
vendor/fairy-stockfish/stockfish.wasm
vendor/fairy-stockfish/stockfish.worker.js
vendor/fairy-stockfish/uci.js
```

2. 浏览器端不要直接在主线程跑。使用单独 worker 包装 UCI。可先参考本目录附带的 `fairy-uci-worker.example.js`，它输出的 `result.lines[]` 已尽量对齐 v12.2 的分析结果结构。

3. 初始化顺序建议：

```text
uci
setoption name UCI_Variant value gardner
setoption name Hash value 32
setoption name Threads value 1
setoption name Use NNUE value true
isready
```

4. 每次搜索：

```text
setoption name MultiPV value 3
position fen <v12.2 compact FEN>
go movetime 1000
```

或调试时用：

```text
go depth 5
```

5. 解析 `info ... score cp/mate ... pv ...` 和 `bestmove ...`，并转成 v12.2 UI 期望的结构：

```js
{
  engine: 'Fairy-Stockfish wasm 1.1.11',
  depth,
  nodes,
  nps,
  elapsed,
  lines: [{ move, score, scoreText, pv, mateVerified: false }]
}
```

6. 分数方向要注意：Stockfish/Fairy-Stockfish 的 root score 是“当前走棋方视角”；v12.2 的 UI 使用“白方视角”。所以黑方走棋时需要把 cp/mate 符号取反。

7. 应用前必须用 v12.2 的 `uciToMove(position, uci)` 或 `findMoveByUci` 校验。外部引擎给出的 PV 中任何非法步，都不要直接落子；作为备用引擎时应 fallback 到 Orion JS 12.2。

## 部署注意

- 必须通过 HTTP/HTTPS 服务访问，不能用 `file://` 直接打开。
- `.wasm` 必须以 `application/wasm` 返回。
- 此 wasm 构建使用 pthread/SharedArrayBuffer 相关机制。正式部署建议配置跨源隔离响应头：

```text
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

并尽量同源加载 `stockfish.js`、`stockfish.wasm`、`stockfish.worker.js`。

## 与 v12.2 的具体改动点

最小改动路线：

- 新增一个 `js/engine/fairy-stockfish-client.js`，负责和 `vendor/fairy-stockfish/fairy-uci-worker.example.js` 通信。
- 在 `js/engine/play-worker.js` 或 `PlayEngineClient` 外层增加“备用引擎”开关：Orion 正常可用时仍走现有 `GardnerSearcher`；Orion 超时、失败或用户手动选择时调用 Fairy-Stockfish。
- 分析面板也可同理在 `AnalysisClient` 外层加一个 provider 选择，不建议直接替换现有 `worker.js`，因为 v12.2 的 tablebase、缓存、mateVerified、风格化难度都在现有 JS 引擎里。
- 缓存 key 要区分来源，例如 `engine=Fairy-Stockfish wasm 1.1.11`，不要把外部引擎结果混入 `Orion JS 12.2` 的已验证 mate/tablebase 缓存。

## 许可证提醒

该包 `package.json` 标注为 GPL-3.0。将其打包进网站分发时，应保留版权/许可证信息，并按 GPL-3.0 的要求提供对应源代码与修改说明。
