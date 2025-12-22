import http.server
import socketserver
import webbrowser
import os
from datetime import datetime

# 切换到当前目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

class NoCacheHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # 检查是否是需要禁用缓存的文件
        no_cache_paths = [
            '/cpp/minesweeper.js',
            '/cpp/minesweeper.wasm',
            '/index.html'
        ]
        
        should_no_cache = any(self.path.startswith(path) for path in no_cache_paths)
        
        if should_no_cache:
            # 禁用缓存
            self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Expires', '0')
        else:
            # 保留其他文件的缓存（如 Pyodide）
            if '/pyodide/' in self.path:
                # Pyodide 文件可以缓存较长时间
                self.send_header('Cache-Control', 'public, max-age=3600')
            else:
                # 其他文件使用默认缓存策略
                pass
        
        super().end_headers()

# 启动服务器
PORT = 8000

with socketserver.TCPServer(("", PORT), NoCacheHTTPRequestHandler) as httpd:
    print(f"服务器启动: http://localhost:{PORT}/index.html")
    webbrowser.open(f"http://localhost:{PORT}/index.html")
    print("按 Ctrl+C 停止")
    print("注意：只有 minesweeper 相关文件禁用缓存，Pyodide 仍可缓存")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n服务器已停止")
