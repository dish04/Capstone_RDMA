#!/usr/bin/env python3
"""
================================================================================
A* CONFERENCE-GRADE DISTRIBUTED LLM BENCHMARK SUITE
Multi-Dimensional Evaluation: Cluster Nodes x Prompt Size x Generation Length x Interconnect
Measures: TTFT (Prefill), ITL (Decode), P50/P90/P99, Effective Bandwidth, Bubble Overhead
Target Backends: RDMA (Soft-RoCE Kernel-Bypass) vs. TCP/IP
================================================================================
"""

import os
import sys
import time
import json
import math
import argparse
import subprocess
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
CHECKPOINT_FILE = os.path.join(RESULTS_DIR, "benchmark_checkpoint.json")
CLUSTER_MGR = os.path.join(PROJECT_ROOT, "scripts", "cluster_mgr.sh")
PIPELINE_SCRIPT = os.path.join(PROJECT_ROOT, "capstone", "rdma", "distributed_pipeline.py")

os.makedirs(RESULTS_DIR, exist_ok=True)

def calc_mean(vals):
    return float(sum(vals) / len(vals)) if vals else 0.0

def calc_std(vals, mean_v=None):
    if len(vals) <= 1:
        return 0.0
    if mean_v is None:
        mean_v = calc_mean(vals)
    var = sum((x - mean_v) ** 2 for x in vals) / (len(vals) - 1)
    return float(math.sqrt(var))

def calc_ci95(vals):
    if len(vals) <= 1:
        return 0.0
    std = calc_std(vals)
    return float(1.96 * std / math.sqrt(len(vals)))

def calc_percentile(sorted_vals, p):
    if not sorted_vals:
        return 0.0
    idx = int(len(sorted_vals) * p)
    idx = min(idx, len(sorted_vals) - 1)
    return float(sorted_vals[idx])

def run_single_iteration_local(total_nodes, backend, prompt_seq_len, max_tokens, port_base):
    """Runs a multi-process pipeline locally on the workstation."""
    node_ips = ["127.0.0.1"] * total_nodes
    node_ips_str = ",".join(node_ips)
    temp_json = os.path.join(RESULTS_DIR, f"temp_{backend}_n{total_nodes}_{os.getpid()}_{time.time_ns()}.json")

    processes = []
    try:
        for rank in range(1, total_nodes):
            cmd = [
                sys.executable, "-u", PIPELINE_SCRIPT,
                "--rank", str(rank),
                "--total-nodes", str(total_nodes),
                "--node-ips", node_ips_str,
                "--backend", backend,
                "--port-base", str(port_base),
                "--benchmark",
                "--seq-len", str(prompt_seq_len),
                "--max-tokens", str(max_tokens)
            ]
            p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            processes.append(p)

        time.sleep(1.0)

        cmd_master = [
            sys.executable, "-u", PIPELINE_SCRIPT,
            "--rank", "0",
            "--total-nodes", str(total_nodes),
            "--node-ips", node_ips_str,
            "--backend", backend,
            "--port-base", str(port_base),
            "--benchmark",
            "--seq-len", str(prompt_seq_len),
            "--max-tokens", str(max_tokens),
            "--json-output", temp_json
        ]
        p_master = subprocess.Popen(cmd_master, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        processes.append(p_master)

        stdout, stderr = p_master.communicate(timeout=120)

        for p in processes[:-1]:
            try:
                p.wait(timeout=3)
            except subprocess.TimeoutExpired:
                p.kill()

        if os.path.exists(temp_json):
            with open(temp_json, "r") as f:
                data = json.load(f)
            try: os.remove(temp_json)
            except: pass
            return data
        return None
    except Exception:
        for p in processes:
            try: p.kill()
            except: pass
        return None

def run_single_iteration_cluster(total_nodes, backend, prompt_seq_len, max_tokens):
    """Runs Rank 0 on the host driving the running QEMU cluster."""
    node_ips = ["192.168.100.1"] + [f"192.168.100.{i+1}" for i in range(1, total_nodes)]
    node_ips_str = ",".join(node_ips)
    temp_json = os.path.join(RESULTS_DIR, f"cluster_{backend}_n{total_nodes}_{time.time_ns()}.json")

    cmd = [
        sys.executable, "-u", PIPELINE_SCRIPT,
        "--rank", "0",
        "--total-nodes", str(total_nodes),
        "--node-ips", node_ips_str,
        "--backend", backend,
        "--benchmark",
        "--seq-len", str(prompt_seq_len),
        "--max-tokens", str(max_tokens),
        "--json-output", temp_json
    ]
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = p.communicate(timeout=180)

        if os.path.exists(temp_json):
            with open(temp_json, "r") as f:
                data = json.load(f)
            try: os.remove(temp_json)
            except: pass
            return data
        return None
    except Exception:
        return None

def ensure_cluster_state(target_nodes, current_nodes):
    """Ensures QEMU cluster is running with exact target_nodes."""
    if current_nodes == target_nodes:
        return current_nodes

    if current_nodes is not None:
        print(f"\n[Cluster Orchestrator] Stopping current cluster ({current_nodes} nodes)...")
        subprocess.run(["sudo", CLUSTER_MGR, "stop"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2)

    print(f"\n[Cluster Orchestrator] Starting cluster with {target_nodes} nodes...")
    res = subprocess.run(["sudo", CLUSTER_MGR, "start", str(target_nodes)])
    if res.returncode != 0:
        print(f"[ERROR] Failed to start cluster with {target_nodes} nodes.")
        return None

    time.sleep(3)
    return target_nodes

def main():
    parser = argparse.ArgumentParser(description="A* Rigorous Distributed LLM Benchmark Suite")
    parser.add_argument("--profile", choices=["full", "standard", "quick", "custom"], default="standard",
                        help="Benchmark profile preset (full=128 configs, standard=72 configs, quick=32 configs)")
    parser.add_argument("--nodes", type=str, default="2,3,4,5", help="Comma-separated node counts (e.g. 2,3,4,5)")
    parser.add_argument("--prompts", type=str, default="32,128,512,1024", help="Comma-separated prompt sizes in tokens")
    parser.add_argument("--tokens", type=str, default="10,25,50,100", help="Comma-separated generation token lengths")
    parser.add_argument("--backends", type=str, default="tcp,rdma", help="Backends: 'tcp', 'rdma', or 'tcp,rdma'")
    parser.add_argument("--trials", type=int, default=3, help="Measured repetitions per configuration cell")
    parser.add_argument("--warmup", type=int, default=2, help="Warm-up runs discarded per configuration cell")
    parser.add_argument("--mode", choices=["cluster", "local"], default="cluster",
                        help="Execution mode: 'cluster' (QEMU virtual instances) or 'local' (host multi-process)")
    parser.add_argument("--resume", action="store_true", help="Resume from previous checkpoint if available")

    args = parser.parse_args()

    if args.profile == "full":
        node_list = [2, 3, 4, 5]
        prompt_list = [32, 128, 512, 1024]
        token_list = [10, 25, 50, 100]
        num_trials = 5
    elif args.profile == "standard":
        node_list = [2, 3, 4, 5]
        prompt_list = [32, 256, 1024]
        token_list = [10, 50, 100]
        num_trials = 3
    elif args.profile == "quick":
        node_list = [2, 3, 4, 5]
        prompt_list = [32, 256]
        token_list = [10, 25]
        num_trials = 2
    else:
        node_list = [int(n.strip()) for n in args.nodes.split(",") if n.strip()]
        prompt_list = [int(p.strip()) for p in args.prompts.split(",") if p.strip()]
        token_list = [int(t.strip()) for t in args.tokens.split(",") if t.strip()]
        num_trials = args.trials

    backend_list = [b.strip().lower() for b in args.backends.split(",") if b.strip()]

    total_cells = len(node_list) * len(prompt_list) * len(token_list) * len(backend_list)
    total_runs = total_cells * (num_trials + args.warmup)

    print("================================================================================")
    print("       A* RESEARCH BENCHMARK: MULTI-NODE DISTRIBUTED LLM INFERENCE")
    print("================================================================================")
    print(f"Profile Preset       : {args.profile.upper()}")
    print(f"Execution Mode       : {args.mode.upper()}")
    print(f"Cluster Scale (Nodes): {node_list}")
    print(f"Prompt Sizes (Tokens): {prompt_list}  [Prefill Payloads: ~0.46MB to ~14.7MB]")
    print(f"Generation Lengths   : {token_list} tokens")
    print(f"Interconnect Backends: {[b.upper() for b in backend_list]}")
    print(f"Statistical Rigor    : {args.warmup} warmups (discarded) + {num_trials} measured trials")
    print(f"Total Configurations : {total_cells} parameter cells ({total_runs} total executions)")
    print(f"Results Directory    : {RESULTS_DIR}")
    print("================================================================================")

    completed_cells = {}
    raw_telemetry = []
    if args.resume and os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r") as f:
                chk = json.load(f)
                completed_cells = chk.get("completed_cells", {})
                raw_telemetry = chk.get("raw_telemetry", [])
            print(f"[Checkpoint] Resuming from checkpoint with {len(completed_cells)} cells already done!")
        except Exception as e:
            print(f"[Checkpoint] Could not load checkpoint ({e}), starting fresh.")

    current_cluster_nodes = None
    port_counter = 21000
    cell_idx = 0
    matrix_results = []
    start_wall_time = time.time()

    for num_nodes in node_list:
        if args.mode == "cluster":
            current_cluster_nodes = ensure_cluster_state(num_nodes, current_cluster_nodes)
            if current_cluster_nodes is None:
                print(f"[FATAL] Skipping {num_nodes} nodes due to cluster boot failure.")
                continue

        for prompt_size in prompt_list:
            for max_tokens in token_list:
                for backend in backend_list:
                    cell_idx += 1
                    cell_key = f"{num_nodes}_{backend}_{prompt_size}_{max_tokens}"

                    if cell_key in completed_cells:
                        print(f"[{cell_idx}/{total_cells}] (Cached) {num_nodes}N | {backend.upper()} | Prompt: {prompt_size} | Gen: {max_tokens} tokens")
                        matrix_results.append(completed_cells[cell_key])
                        continue

                    prefill_mb = round(prompt_size * 3584 * 4 / (1024 * 1024), 2)
                    print(f"\n--------------------------------------------------------------------------------")
                    print(f"[{cell_idx}/{total_cells}] RUNNING: {num_nodes} Nodes | {backend.upper()} | Prompt: {prompt_size} tok ({prefill_mb}MB) | Gen: {max_tokens} tok")
                    print(f"--------------------------------------------------------------------------------")

                    # Warm-up runs
                    for w in range(args.warmup):
                        port_counter += (num_nodes + 5)
                        print(f"  [Warmup {w+1}/{args.warmup}] ... ", end="", flush=True)
                        if args.mode == "cluster":
                            w_res = run_single_iteration_cluster(num_nodes, backend, prompt_size, max_tokens)
                        else:
                            w_res = run_single_iteration_local(num_nodes, backend, prompt_size, max_tokens, port_counter)
                        if w_res:
                            print(f"Done ({w_res.get('ttft_ms', 0):.1f}ms TTFT)")
                        else:
                            print("Skipped/Failed")

                    # Measured trials
                    trial_ttfts = []
                    trial_itls = []
                    trial_throughputs = []
                    trial_computes = []
                    trial_networks = []
                    trial_p50s = []
                    trial_p90s = []
                    trial_p99s = []
                    trial_bws = []

                    for t in range(num_trials):
                        port_counter += (num_nodes + 5)
                        print(f"  [Trial {t+1}/{num_trials}] Running ... ", end="", flush=True)
                        if args.mode == "cluster":
                            res = run_single_iteration_cluster(num_nodes, backend, prompt_size, max_tokens)
                        else:
                            res = run_single_iteration_local(num_nodes, backend, prompt_size, max_tokens, port_counter)

                        if res:
                            trial_ttfts.append(res['ttft_ms'])
                            trial_itls.append(res['itl_mean_ms'])
                            trial_throughputs.append(res['decode_tokens_per_second'])
                            trial_computes.append(res['mean_compute_latency_ms'])
                            trial_networks.append(res['mean_network_latency_ms'])
                            trial_p50s.append(res.get('itl_p50_ms', res['itl_mean_ms']))
                            trial_p90s.append(res.get('itl_p90_ms', res['itl_mean_ms']))
                            trial_p99s.append(res.get('itl_p99_ms', res['itl_mean_ms']))
                            trial_bws.append(res.get('effective_bw_mb_s', 0.0))

                            raw_telemetry.append({
                                "cell_key": cell_key,
                                "trial": t + 1,
                                "timestamp": datetime.utcnow().isoformat(),
                                "result": res
                            })

                            print(f"Done! TTFT: {res['ttft_ms']:.1f}ms | ITL: {res['itl_mean_ms']:.2f}ms | Decode: {res['decode_tokens_per_second']:.1f} tok/s")
                        else:
                            print("FAILED")

                    if trial_ttfts:
                        mean_ttft = calc_mean(trial_ttfts)
                        std_ttft = calc_std(trial_ttfts, mean_ttft)
                        ci_ttft = calc_ci95(trial_ttfts)

                        mean_itl = calc_mean(trial_itls)
                        std_itl = calc_std(trial_itls, mean_itl)
                        ci_itl = calc_ci95(trial_itls)

                        mean_tps = calc_mean(trial_throughputs)
                        mean_comp = calc_mean(trial_computes)
                        mean_net = calc_mean(trial_networks)
                        mean_p50 = calc_mean(trial_p50s)
                        mean_p90 = calc_mean(trial_p90s)
                        mean_p99 = calc_mean(trial_p99s)
                        mean_bw = calc_mean(trial_bws)

                        per_hop_net_ms = mean_net / max(1, (num_nodes - 1))
                        bubble_ratio = (mean_net / (mean_comp + mean_net)) if (mean_comp + mean_net) > 0 else 0.0

                        cell_summary = {
                            "cell_key": cell_key,
                            "nodes": num_nodes,
                            "backend": backend,
                            "prompt_size": prompt_size,
                            "prefill_payload_mb": prefill_mb,
                            "gen_tokens": max_tokens,
                            "ttft_mean_ms": round(mean_ttft, 2),
                            "ttft_std_ms": round(std_ttft, 2),
                            "ttft_ci95_ms": round(ci_ttft, 2),
                            "itl_mean_ms": round(mean_itl, 2),
                            "itl_std_ms": round(std_itl, 2),
                            "itl_ci95_ms": round(ci_itl, 2),
                            "itl_p50_ms": round(mean_p50, 2),
                            "itl_p90_ms": round(mean_p90, 2),
                            "itl_p99_ms": round(mean_p99, 2),
                            "decode_throughput_tps": round(mean_tps, 2),
                            "compute_time_ms": round(mean_comp, 2),
                            "network_time_ms": round(mean_net, 2),
                            "per_hop_delay_ms": round(per_hop_net_ms, 2),
                            "effective_bw_mb_s": round(mean_bw, 2),
                            "bubble_ratio_pct": round(bubble_ratio * 100.0, 1)
                        }

                        matrix_results.append(cell_summary)
                        completed_cells[cell_key] = cell_summary

                        with open(CHECKPOINT_FILE, "w") as f:
                            json.dump({
                                "completed_cells": completed_cells,
                                "raw_telemetry": raw_telemetry
                            }, f, indent=2)

    if args.mode == "cluster" and current_cluster_nodes is not None:
        print("\n[Cluster Orchestrator] Tearing down cluster...")
        subprocess.run(["sudo", CLUSTER_MGR, "stop"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    pairings = {}
    for r in matrix_results:
        k = (r['nodes'], r['prompt_size'], r['gen_tokens'])
        if k not in pairings:
            pairings[k] = {}
        pairings[k][r['backend']] = r

    for k, v in pairings.items():
        if "rdma" in v and "tcp" in v:
            tcp_ttft = v["tcp"]["ttft_mean_ms"]
            rdma_ttft = v["rdma"]["ttft_mean_ms"]
            tcp_itl = v["tcp"]["itl_mean_ms"]
            rdma_itl = v["rdma"]["itl_mean_ms"]

            v["rdma"]["ttft_speedup"] = round(tcp_ttft / rdma_ttft, 2) if rdma_ttft > 0 else 1.0
            v["rdma"]["itl_speedup"] = round(tcp_itl / rdma_itl, 2) if rdma_itl > 0 else 1.0
            v["tcp"]["ttft_speedup"] = 1.0
            v["tcp"]["itl_speedup"] = 1.0

    raw_file = os.path.join(RESULTS_DIR, "raw_telemetry.json")
    with open(raw_file, "w") as f:
        json.dump(raw_telemetry, f, indent=2)

    csv_file = os.path.join(RESULTS_DIR, "rigorous_matrix.csv")
    with open(csv_file, "w") as f:
        headers = [
            "Nodes", "Backend", "Prompt_Tokens", "Payload_MB", "Gen_Tokens",
            "TTFT_Mean_ms", "TTFT_Std_ms", "TTFT_CI95_ms", "TTFT_Speedup",
            "ITL_Mean_ms", "ITL_Std_ms", "ITL_CI95_ms", "ITL_P50_ms", "ITL_P90_ms", "ITL_P99_ms", "ITL_Speedup",
            "Throughput_tok_s", "Compute_ms", "Network_ms", "Per_Hop_Delay_ms", "Effective_BW_MB_s", "Bubble_Ratio_Pct"
        ]
        f.write(",".join(headers) + "\n")
        for r in matrix_results:
            row = [
                str(r["nodes"]), r["backend"].upper(), str(r["prompt_size"]), str(r["prefill_payload_mb"]), str(r["gen_tokens"]),
                str(r["ttft_mean_ms"]), str(r["ttft_std_ms"]), str(r["ttft_ci95_ms"]), str(r.get("ttft_speedup", 1.0)),
                str(r["itl_mean_ms"]), str(r["itl_std_ms"]), str(r["itl_ci95_ms"]),
                str(r["itl_p50_ms"]), str(r["itl_p90_ms"]), str(r["itl_p99_ms"]), str(r.get("itl_speedup", 1.0)),
                str(r["decode_throughput_tps"]), str(r["compute_time_ms"]), str(r["network_time_ms"]),
                str(r["per_hop_delay_ms"]), str(r["effective_bw_mb_s"]), str(r["bubble_ratio_pct"])
            ]
            f.write(",".join(row) + "\n")

    tex_file = os.path.join(RESULTS_DIR, "latex_table.tex")
    with open(tex_file, "w") as f:
        f.write("% ====================================================================\n")
        f.write("% Academic Booktabs Table: Distributed Pipeline Scaling (RDMA vs TCP)\n")
        f.write("% ====================================================================\n")
        f.write("\\begin{table*}[t]\n\\centering\n\\small\n")
        f.write("\\caption{Performance comparison of multi-node pipeline parallelism across cluster scale ($N$), prompt context, and token generation under Soft-RoCE RDMA vs. TCP/IP.}\n")
        f.write("\\label{tab:pipeline_benchmark}\n")
        f.write("\\begin{tabular}{@{}ccc|cc|ccc|cc@{}}\n\\toprule\n")
        f.write("\\textbf{Nodes} & \\textbf{Prompt} & \\textbf{Tokens} & \\multicolumn{2}{c|}{\\textbf{TTFT (Prefill Latency, ms)}} & \\multicolumn{3}{c|}{\\textbf{ITL (Decode Latency, ms)}} & \\multicolumn{2}{c}{\\textbf{Throughput (tok/s)}} \\\\\n")
        f.write("($N$) & (Tokens) & ($T$) & \\textbf{TCP} & \\textbf{RDMA (Speedup)} & \\textbf{TCP (Mean$\\pm\\sigma$)} & \\textbf{RDMA (Mean$\\pm\\sigma$)} & \\textbf{Speedup} & \\textbf{TCP} & \\textbf{RDMA} \\\\\n\\midrule\n")

        for k, v in sorted(pairings.items()):
            if "tcp" in v and "rdma" in v:
                t = v["tcp"]
                r = v["rdma"]
                ttft_sp = r.get("ttft_speedup", 1.0)
                itl_sp = r.get("itl_speedup", 1.0)
                row_str = "%d & %d & %d & %.1f & %.1f (\\textbf{%.2f$\\times$}) & %.1f$\\pm$%.1f & %.1f$\\pm$%.1f & \\textbf{%.2f$\\times$} & %.1f & \\textbf{%.1f} \\\\\n" % (
                    k[0], k[1], k[2],
                    t['ttft_mean_ms'], r['ttft_mean_ms'], ttft_sp,
                    t['itl_mean_ms'], t['itl_std_ms'],
                    r['itl_mean_ms'], r['itl_std_ms'],
                    itl_sp,
                    t['decode_throughput_tps'], r['decode_throughput_tps']
                )
                f.write(row_str)
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table*}\n")

    total_elapsed = round(time.time() - start_wall_time, 1)
    print("\n" + "=" * 90)
    print("                      A* BENCHMARK COMPLETE - EXECUTIVE SUMMARY")
    print("=" * 90)
    print(f"Total Wall Time: {total_elapsed}s | Configurations Evaluated: {len(matrix_results)}")
    print("-" * 90)
    header_fmt = "{:<6} {:<8} {:<7} {:<8} {:<12} {:<12} {:<10} {:<14} {:<8}"
    print(header_fmt.format("Nodes", "Prompt", "GenTok", "Backend", "TTFT (ms)", "ITL (ms)", "P99 (ms)", "Throughput", "Speedup"))
    print("-" * 90)
    row_fmt = "{:<6} {:<8} {:<7} {:<8} {:<12.1f} {:<12.2f} {:<10.2f} {:<14.1f} {:<8}"
    for r in matrix_results:
        sp_str = f"{r.get('itl_speedup', 1.0):.2f}x" if r["backend"] == "rdma" else "1.00x"
        print(row_fmt.format(
            r['nodes'], r['prompt_size'], r['gen_tokens'], r['backend'].upper(),
            r['ttft_mean_ms'], r['itl_mean_ms'], r['itl_p99_ms'],
            r['decode_throughput_tps'], sp_str
        ))
    print("=" * 90)
    print(f"Generated Research Artifacts:")
    print(f"  - CSV Matrix     : {csv_file}")
    print(f"  - Raw Telemetry  : {raw_file}")
    print(f"  - LaTeX Table    : {tex_file}")
    print("=" * 90 + "\n")

    plot_script = os.path.join(SCRIPT_DIR, "plot_rigorous_results.py")
    if os.path.exists(plot_script):
        subprocess.run([sys.executable, plot_script])

if __name__ == "__main__":
    main()
