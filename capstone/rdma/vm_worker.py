#!/usr/bin/env python3
"""
Lightweight Worker Service running on QEMU Virtual Nodes.
Listens on port 18000 for pipeline stage execution requests from the Host Orchestrator.
"""
import os
import sys
import json
import argparse
import subprocess
from http.server import HTTPServer, BaseHTTPRequestHandler

current_process = None

class WorkerHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global current_process
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok"}).encode())
        elif self.path == "/status":
            is_running = current_process is not None and current_process.poll() is None
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "running" if is_running else "idle",
                "returncode": current_process.poll() if current_process else None
            }).encode())
        elif self.path == "/stop":
            if current_process and current_process.poll() is None:
                try: current_process.kill()
                except: pass
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "stopped"}')
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        global current_process
        if self.path == "/start":
            content_len = int(self.headers.get('Content-Length', 0))
            post_body = self.rfile.read(content_len)
            params = json.loads(post_body.decode())

            # Stop existing process if still running
            if current_process and current_process.poll() is None:
                try:
                    current_process.kill()
                    current_process.wait(timeout=1)
                except Exception:
                    pass

            pipeline_script = "/mnt/weights/distributed_pipeline.py"
            if not os.path.exists(pipeline_script):
                pipeline_script = os.path.expanduser("~/qwen_weights/distributed_pipeline.py")

            cmd = [
                sys.executable, "-u", pipeline_script,
                "--rank", str(params["rank"]),
                "--total-nodes", str(params["total_nodes"]),
                "--node-ips", params["node_ips"],
                "--backend", params["backend"],
                "--port-base", str(params.get("port_base", 19000)),
                "--seq-len", str(params.get("seq_len", 32)),
                "--max-tokens", str(params.get("max_tokens", 20))
            ]
            if params.get("benchmark", True):
                cmd.append("--benchmark")
            if params.get("model_path"):
                cmd.extend(["--model-path", params["model_path"]])

            log_file = open("/tmp/pipeline_stage.log", "w")
            current_process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "started",
                "pid": current_process.pid,
                "rank": params["rank"]
            }).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vm-id", type=int, default=1)
    parser.add_argument("--port", type=int, default=18000)
    args = parser.parse_args()

    server = HTTPServer(("0.0.0.0", args.port), WorkerHandler)
    print(f"[VM Worker {args.vm_id}] Service listening on 0.0.0.0:{args.port}...")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    server.server_close()

if __name__ == "__main__":
    main()
