from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import os
import tempfile

HOST = "127.0.0.1"
PORT = 8765

PROJECT_DIR = Path.cwd()
OUTPUT_FILE = PROJECT_DIR / "data" / "rail_snapshot.geojson"


class ExportHandler(BaseHTTPRequestHandler):
    def add_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type, Content-Length"
        )
        self.send_header("Access-Control-Allow-Private-Network", "true")

    def do_OPTIONS(self):
        self.send_response(204)
        self.add_cors_headers()
        self.end_headers()

    def do_POST(self):
        if self.path != "/export":
            self.send_error(404, "Not found")
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))

            if content_length <= 0:
                raise ValueError("收到的数据长度为 0")

            raw_data = self.rfile.read(content_length)

            if not raw_data:
                raise ValueError("没有收到任何内容")

            text = raw_data.decode("utf-8")
            geojson = json.loads(text)

            if geojson.get("type") != "FeatureCollection":
                raise ValueError("数据不是 GeoJSON FeatureCollection")

            features = geojson.get("features")

            if not isinstance(features, list) or len(features) == 0:
                raise ValueError("GeoJSON 的 features 为空")

            OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

            # 先写临时文件，成功后再替换，避免产生 0 字节目标文件。
            fd, temp_name = tempfile.mkstemp(
                prefix="rail_snapshot_",
                suffix=".tmp",
                dir=OUTPUT_FILE.parent
            )

            try:
                with os.fdopen(fd, "wb") as temp_file:
                    temp_file.write(raw_data)
                    temp_file.flush()
                    os.fsync(temp_file.fileno())

                os.replace(temp_name, OUTPUT_FILE)
            except Exception:
                try:
                    os.unlink(temp_name)
                except OSError:
                    pass
                raise

            result = {
                "ok": True,
                "path": str(OUTPUT_FILE.resolve()),
                "bytes": OUTPUT_FILE.stat().st_size,
                "features": len(features)
            }

            response = json.dumps(
                result,
                ensure_ascii=False
            ).encode("utf-8")

            self.send_response(200)
            self.add_cors_headers()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

            print()
            print("导出成功")
            print(f"文件：{OUTPUT_FILE.resolve()}")
            print(f"大小：{OUTPUT_FILE.stat().st_size:,} 字节")
            print(f"要素：{len(features):,}")

        except Exception as exc:
            error = json.dumps(
                {
                    "ok": False,
                    "error": str(exc)
                },
                ensure_ascii=False
            ).encode("utf-8")

            self.send_response(400)
            self.add_cors_headers()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(error)))
            self.end_headers()
            self.wfile.write(error)

            print(f"导出失败：{exc}")

    def log_message(self, format, *args):
        print(f"[HTTP] {format % args}")


if __name__ == "__main__":
    print(f"项目目录：{PROJECT_DIR}")
    print(f"输出文件：{OUTPUT_FILE.resolve()}")
    print(f"正在监听：http://{HOST}:{PORT}/export")
    print("保持本终端开启，然后在已加载地图的页面控制台执行导出代码。")

    server = ThreadingHTTPServer((HOST, PORT), ExportHandler)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n接收程序已停止。")
    finally:
        server.server_close()