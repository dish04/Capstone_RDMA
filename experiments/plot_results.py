#!/usr/bin/env python3
"""
Visualization and Results Analysis for RDMA vs TCP LLM Inference Experiments.
Generates terminal comparison tables and publication-quality comparison charts.
"""

import os
import sys
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
JSON_PATH = os.path.join(RESULTS_DIR, "benchmark_data.json")

def print_markdown_table(data):
    print("\n" + "=" * 95)
    print("                 DISTRIBUTED LLM INFERENCE: RDMA vs. TCP BENCHMARK RESULTS")
    print("=" * 95)

    # Group data by node count
    by_nodes = {}
    for entry in data:
        n = entry['num_nodes']
        if n not in by_nodes:
            by_nodes[n] = {}
        by_nodes[n][entry['backend'].lower()] = entry

    header = f"| {'Nodes':<5} | {'Protocol':<8} | {'Total Latency':<14} | {'Compute Time':<13} | {'Network Delay':<14} | {'Per-Hop Delay':<13} | {'Throughput':<12} |"
    sep = f"|{'-'*7}|{'-'*10}|{'-'*16}|{'-'*15}|{'-'*16}|{'-'*15}|{'-'*14}|"
    print(header)
    print(sep)

    for n in sorted(by_nodes.keys()):
        node_group = by_nodes[n]
        for backend in ['rdma', 'tcp']:
            if backend in node_group:
                e = node_group[backend]
                line = (f"| {e['num_nodes']:<5} | {e['backend'].upper():<8} | "
                        f"{e['mean_latency_ms']:>8.2f} ms     | "
                        f"{e['compute_time_ms']:>7.2f} ms     | "
                        f"{e['network_time_ms']:>8.2f} ms     | "
                        f"{e['per_hop_delay_ms']:>7.2f} ms     | "
                        f"{e['throughput_tps']:>6.2f} tok/s  |")
                print(line)

        # Print speedup / comparison row if both exist
        if 'rdma' in node_group and 'tcp' in node_group:
            r = node_group['rdma']
            t = node_group['tcp']
            lat_diff = t['mean_latency_ms'] - r['mean_latency_ms']
            lat_pct = (lat_diff / t['mean_latency_ms']) * 100 if t['mean_latency_ms'] > 0 else 0
            net_diff = t['network_time_ms'] - r['network_time_ms']
            net_pct = (net_diff / t['network_time_ms']) * 100 if t['network_time_ms'] > 0 else 0
            speedup = t['mean_latency_ms'] / r['mean_latency_ms'] if r['mean_latency_ms'] > 0 else 1.0

            summary_note = f"--> [{n} Nodes] RDMA provides {speedup:.2f}x speedup ({lat_pct:+.1f}% total latency, {net_pct:+.1f}% network delay reduction)"
            print(f"| {summary_note:<89} |")
            print(sep)

    print("=" * 95 + "\n")

def generate_plots(data):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[Notice] matplotlib is not installed in the current environment.")
        print("         To generate graphical PNG charts, run: pip install matplotlib")
        return

    # Extract unique node counts and backends
    nodes = sorted(list(set(d['num_nodes'] for d in data)))
    
    rdma_lat = []
    tcp_lat = []
    rdma_net = []
    tcp_net = []
    rdma_tps = []
    tcp_tps = []

    for n in nodes:
        r_entry = next((d for d in data if d['num_nodes'] == n and d['backend'].lower() == 'rdma'), None)
        t_entry = next((d for d in data if d['num_nodes'] == n and d['backend'].lower() == 'tcp'), None)

        rdma_lat.append(r_entry['mean_latency_ms'] if r_entry else 0)
        tcp_lat.append(t_entry['mean_latency_ms'] if t_entry else 0)
        rdma_net.append(r_entry['network_time_ms'] if r_entry else 0)
        tcp_net.append(t_entry['network_time_ms'] if t_entry else 0)
        rdma_tps.append(r_entry['throughput_tps'] if r_entry else 0)
        tcp_tps.append(t_entry['throughput_tps'] if t_entry else 0)

    x = np.arange(len(nodes))
    width = 0.35

    # 1. Total Token Latency Comparison
    plt.figure(figsize=(9, 5))
    plt.bar(x - width/2, rdma_lat, width, label='RDMA (Soft-RoCE)', color='#1f77b4')
    plt.bar(x + width/2, tcp_lat, width, label='TCP/IP', color='#ff7f0e')
    plt.xlabel('Number of Pipeline Nodes', fontsize=12, fontweight='bold')
    plt.ylabel('Mean Token Latency (ms)', fontsize=12, fontweight='bold')
    plt.title('Distributed LLM Inference Latency: RDMA vs. TCP', fontsize=14, fontweight='bold')
    plt.xticks(x, [f'{n} Nodes' for n in nodes], fontsize=11)
    plt.legend(fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    chart1 = os.path.join(RESULTS_DIR, "latency_vs_nodes.png")
    plt.savefig(chart1, dpi=300)
    plt.close()
    print(f"[Plot] Generated: {chart1}")

    # 2. Network Transmission Delay Breakdown
    plt.figure(figsize=(9, 5))
    plt.plot(nodes, rdma_net, marker='o', linewidth=2.5, label='RDMA Network Delay', color='#2ca02c')
    plt.plot(nodes, tcp_net, marker='s', linewidth=2.5, label='TCP Network Delay', color='#d62728')
    plt.xlabel('Number of Pipeline Nodes', fontsize=12, fontweight='bold')
    plt.ylabel('Network Transmission Delay (ms)', fontsize=12, fontweight='bold')
    plt.title('Network Delay Scaling Across Pipeline Stages', fontsize=14, fontweight='bold')
    plt.xticks(nodes, [f'{n} Nodes' for n in nodes], fontsize=11)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    chart2 = os.path.join(RESULTS_DIR, "network_delay_vs_nodes.png")
    plt.savefig(chart2, dpi=300)
    plt.close()
    print(f"[Plot] Generated: {chart2}")

    # 3. Throughput Scaling
    plt.figure(figsize=(9, 5))
    plt.bar(x - width/2, rdma_tps, width, label='RDMA', color='#2ca02c')
    plt.bar(x + width/2, tcp_tps, width, label='TCP', color='#1f77b4')
    plt.xlabel('Number of Pipeline Nodes', fontsize=12, fontweight='bold')
    plt.ylabel('Throughput (Tokens / Second)', fontsize=12, fontweight='bold')
    plt.title('Token Generation Throughput: RDMA vs. TCP', fontsize=14, fontweight='bold')
    plt.xticks(x, [f'{n} Nodes' for n in nodes], fontsize=11)
    plt.legend(fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    chart3 = os.path.join(RESULTS_DIR, "throughput_vs_nodes.png")
    plt.savefig(chart3, dpi=300)
    plt.close()
    print(f"[Plot] Generated: {chart3}")

def main():
    if not os.path.exists(JSON_PATH):
        print(f"Error: Results file not found at {JSON_PATH}")
        print("Please run experiments/run_experiment.py first to generate benchmark data.")
        sys.exit(1)

    with open(JSON_PATH, "r") as f:
        data = json.load(f)

    print_markdown_table(data)
    generate_plots(data)

if __name__ == "__main__":
    main()
