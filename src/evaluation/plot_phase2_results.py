"""
Phase 2 Results Visualization Script for SDTP
Generates comprehensive, publication-quality figures for Phase 2 evaluation results.

This script creates multiple figures:
1. End2End latency comparison (multi-config)
2. KV Cache compression effect
3. Performance-Speed trade-off
4. LongBench performance retention
5. Comprehensive heatmap comparison
"""

import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# Color scheme for different configurations
COLORS = {
    'baseline': '#808080',  # Gray
    'keep09': '#2E86AB',    # Blue
    'keep08': '#A23B72',    # Purple
    'keep07': '#F18F01',    # Orange
}

# Configuration names mapping
CONFIG_NAMES = {
    'keep09': 'SDTP (keep=0.9)',
    'keep08': 'SDTP (keep=0.8)',
    'keep07': 'SDTP (keep=0.7)',
}


def setup_plot_style():
    """Setup publication-quality plot style."""
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
    })
    sns.set_style("whitegrid")
    sns.set_palette("husl")


def load_latency_data(results_dir: str = "results") -> Dict:
    """
    Load latency data from all configuration JSON files.
    
    Returns:
        dict: {
            'keep09': {length: {baseline: {...}, sdtp: {...}, speedup: {...}}},
            'keep08': {...},
            'keep07': {...}
        }
    """
    configs = ['keep09', 'keep08', 'keep07']
    data = {}
    
    for config in configs:
        json_path = os.path.join(results_dir, f"latency_results_{config}.json")
        if not os.path.exists(json_path):
            print(f"[Warning] File not found: {json_path}")
            continue
        
        with open(json_path, 'r') as f:
            file_data = json.load(f)
        
        if 'results' not in file_data:
            print(f"[Warning] Invalid format in {json_path}")
            continue
        
        data[config] = {}
        for length_str, result in file_data['results'].items():
            length = int(length_str)
            data[config][length] = result
    
    return data


def load_longbench_data(results_dir: str = "results") -> Dict:
    """
    Load LongBench evaluation results.
    
    Returns:
        dict: {
            'baseline': {'hit_rate': 0.345, ...},
            'keep09': {'hit_rate': 0.320, ...},
            'keep08': {'hit_rate': 0.328, ...},
            'keep07': {'hit_rate': 0.332, ...}
        }
    """
    data = {}
    
    # Load baseline
    baseline_path = os.path.join(results_dir, "hotpotqa_baseline.json")
    if os.path.exists(baseline_path):
        with open(baseline_path, 'r') as f:
            baseline_data = json.load(f)
            data['baseline'] = {
                'hit_rate': baseline_data.get('hit_rate', 0),
                'num_eval': baseline_data.get('num_eval', 0),
            }
    
    # Load SDTP configurations
    # Map config names to file suffixes: keep09 -> 0.9, keep08 -> 0.8, keep07 -> 0.7
    config_to_suffix = {
        'keep09': '0.9',
        'keep08': '0.8',
        'keep07': '0.7'
    }
    for config in ['keep09', 'keep08', 'keep07']:
        suffix = config_to_suffix[config]
        json_path = os.path.join(results_dir, f"hotpotqa_sdtp_{suffix}.json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                sdtp_data = json.load(f)
                data[config] = {
                    'hit_rate': sdtp_data.get('hit_rate', 0),
                    'num_eval': sdtp_data.get('num_eval', 0),
                }
    
    return data


def plot_end2end_latency_comparison(latency_data: Dict, out_dir: str = "results/fig"):
    """
    Plot 1: End2End latency comparison across configurations.
    Creates a 3x2 subplot layout showing prefill, decode, total latency and speedups.
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    
    configs = ['keep09', 'keep08', 'keep07']
    lengths = sorted(set(length for config_data in latency_data.values() 
                         for length in config_data.keys()))
    
    # Extract baseline data (use keep09 as reference for baseline)
    baseline_data = {}
    if 'keep09' in latency_data:
        for length in lengths:
            if length in latency_data['keep09']:
                baseline_data[length] = latency_data['keep09'][length].get('baseline', {})
    
    # Plot 1a: Prefill Latency
    ax = axes[0, 0]
    if baseline_data:
        baseline_prefill = [baseline_data.get(l, {}).get('prefill_latency_seconds', 0) for l in lengths]
        ax.plot(lengths, baseline_prefill, 'o-', color=COLORS['baseline'], 
                linewidth=2, markersize=6, label='Baseline', zorder=3)
    
    for config in configs:
        if config not in latency_data:
            continue
        sdtp_prefill = []
        for length in lengths:
            if length in latency_data[config]:
                sdtp_data = latency_data[config][length].get('sdtp', {})
                sdtp_prefill.append(sdtp_data.get('prefill_latency_seconds', 0))
            else:
                sdtp_prefill.append(0)
        ax.plot(lengths, sdtp_prefill, 's--', color=COLORS[config], 
                linewidth=2, markersize=5, label=CONFIG_NAMES[config], alpha=0.8)
    
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Prefill Latency (seconds)')
    ax.set_title('(a) Prefill Latency')
    ax.set_xscale('log', base=2)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    # Plot 1b: Decode Latency
    ax = axes[0, 1]
    if baseline_data:
        baseline_decode = [baseline_data.get(l, {}).get('decode_latency_seconds', 0) for l in lengths]
        ax.plot(lengths, baseline_decode, 'o-', color=COLORS['baseline'], 
                linewidth=2, markersize=6, label='Baseline', zorder=3)
    
    for config in configs:
        if config not in latency_data:
            continue
        sdtp_decode = []
        for length in lengths:
            if length in latency_data[config]:
                sdtp_data = latency_data[config][length].get('sdtp', {})
                sdtp_decode.append(sdtp_data.get('decode_latency_seconds', 0))
            else:
                sdtp_decode.append(0)
        ax.plot(lengths, sdtp_decode, 's--', color=COLORS[config], 
                linewidth=2, markersize=5, label=CONFIG_NAMES[config], alpha=0.8)
    
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Decode Latency (seconds)')
    ax.set_title('(b) Decode Latency')
    ax.set_xscale('log', base=2)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    # Plot 1c: Total Latency
    ax = axes[1, 0]
    if baseline_data:
        baseline_total = [baseline_data.get(l, {}).get('total_latency_seconds', 0) for l in lengths]
        ax.plot(lengths, baseline_total, 'o-', color=COLORS['baseline'], 
                linewidth=2, markersize=6, label='Baseline', zorder=3)
    
    for config in configs:
        if config not in latency_data:
            continue
        sdtp_total = []
        for length in lengths:
            if length in latency_data[config]:
                sdtp_data = latency_data[config][length].get('sdtp', {})
                sdtp_total.append(sdtp_data.get('total_latency_seconds', 0))
            else:
                sdtp_total.append(0)
        ax.plot(lengths, sdtp_total, 's--', color=COLORS[config], 
                linewidth=2, markersize=5, label=CONFIG_NAMES[config], alpha=0.8)
    
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Total Latency (seconds)')
    ax.set_title('(c) Total Latency')
    ax.set_xscale('log', base=2)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    # Plot 1d: Prefill Speedup
    ax = axes[1, 1]
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Baseline (1.0x)')
    for config in configs:
        if config not in latency_data:
            continue
        speedups = []
        for length in lengths:
            if length in latency_data[config]:
                speedup_data = latency_data[config][length].get('speedup', {})
                speedups.append(speedup_data.get('prefill', 1.0))
            else:
                speedups.append(1.0)
        ax.plot(lengths, speedups, 's-', color=COLORS[config], 
                linewidth=2, markersize=5, label=CONFIG_NAMES[config], alpha=0.8)
    
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Prefill Speedup (×)')
    ax.set_title('(d) Prefill Speedup')
    ax.set_xscale('log', base=2)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    # Plot 1e: Decode Speedup
    ax = axes[2, 0]
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Baseline (1.0x)')
    for config in configs:
        if config not in latency_data:
            continue
        speedups = []
        for length in lengths:
            if length in latency_data[config]:
                speedup_data = latency_data[config][length].get('speedup', {})
                speedups.append(speedup_data.get('decode', 1.0))
            else:
                speedups.append(1.0)
        ax.plot(lengths, speedups, 's-', color=COLORS[config], 
                linewidth=2, markersize=5, label=CONFIG_NAMES[config], alpha=0.8)
    
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Decode Speedup (×)')
    ax.set_title('(e) Decode Speedup')
    ax.set_xscale('log', base=2)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    # Plot 1f: Total Speedup
    ax = axes[2, 1]
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Baseline (1.0x)')
    for config in configs:
        if config not in latency_data:
            continue
        speedups = []
        for length in lengths:
            if length in latency_data[config]:
                speedup_data = latency_data[config][length].get('speedup', {})
                speedups.append(speedup_data.get('total', 1.0))
            else:
                speedups.append(1.0)
        ax.plot(lengths, speedups, 's-', color=COLORS[config], 
                linewidth=2, markersize=5, label=CONFIG_NAMES[config], alpha=0.8)
    
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Total Speedup (×)')
    ax.set_title('(f) Total Speedup')
    ax.set_xscale('log', base=2)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    plt.tight_layout()
    out_path = os.path.join(out_dir, "phase2_end2end_latency_comparison.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {out_path}")
    plt.close()


def plot_kv_cache_reduction(latency_data: Dict, out_dir: str = "results/fig"):
    """
    Plot 2: KV Cache compression effect.
    Shows KV cache length and reduction rate across sequence lengths.
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    configs = ['keep09', 'keep08', 'keep07']
    lengths = sorted(set(length for config_data in latency_data.values() 
                         for length in config_data.keys()))
    
    # Extract baseline KV cache lengths
    baseline_kv = {}
    if 'keep09' in latency_data:
        for length in lengths:
            if length in latency_data['keep09']:
                baseline_data = latency_data['keep09'][length].get('baseline', {})
                kv_lens = baseline_data.get('kv_lens_after_prefill', [])
                if kv_lens:
                    baseline_kv[length] = kv_lens[0]  # Use first layer as representative
    
    # Left Y-axis: KV Cache Length
    if baseline_kv:
        baseline_values = [baseline_kv.get(l, 0) for l in lengths]
        ax1.plot(lengths, baseline_values, 'o-', color=COLORS['baseline'], 
                linewidth=2.5, markersize=7, label='Baseline', zorder=3)
    
    for config in configs:
        if config not in latency_data:
            continue
        sdtp_kv = []
        for length in lengths:
            if length in latency_data[config]:
                sdtp_data = latency_data[config][length].get('sdtp', {})
                tokens_kept = sdtp_data.get('tokens_kept', 0)
                sdtp_kv.append(tokens_kept)
            else:
                sdtp_kv.append(0)
        ax1.plot(lengths, sdtp_kv, 's--', color=COLORS[config], 
                linewidth=2, markersize=6, label=CONFIG_NAMES[config], alpha=0.8)
    
    ax1.set_xlabel('Sequence Length', fontsize=12)
    ax1.set_ylabel('KV Cache Length (tokens)', fontsize=12, color='black')
    ax1.set_xscale('log', base=2)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    # Right Y-axis: KV Cache Reduction Rate
    ax2 = ax1.twinx()
    for config in configs:
        if config not in latency_data:
            continue
        reductions = []
        for length in lengths:
            if length in latency_data[config]:
                reduction = latency_data[config][length].get('kv_reduction', 0)
                reductions.append(reduction * 100)  # Convert to percentage
            else:
                reductions.append(0)
        ax2.plot(lengths, reductions, '^:', color=COLORS[config], 
                linewidth=2, markersize=5, label=f'{CONFIG_NAMES[config]} Reduction', alpha=0.7)
    
    ax2.set_ylabel('KV Cache Reduction Rate (%)', fontsize=12, color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    out_path = os.path.join(out_dir, "phase2_kv_cache_reduction.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {out_path}")
    plt.close()


def plot_performance_speed_tradeoff(latency_data: Dict, longbench_data: Dict, 
                                   out_dir: str = "results/fig", ref_length: int = 16384):
    """
    Plot 3: Performance-Speed trade-off.
    Shows Hit Rate vs Speedup for different configurations.
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    
    # Extract data points
    points = []
    
    # Baseline point
    if 'baseline' in longbench_data:
        baseline_hit_rate = longbench_data['baseline']['hit_rate']
        points.append({
            'config': 'Baseline',
            'hit_rate': baseline_hit_rate,
            'speedup': 1.0,
            'color': COLORS['baseline'],
            'marker': 'o',
            'size': 150
        })
    
    # SDTP configurations
    configs = ['keep09', 'keep08', 'keep07']
    for config in configs:
        if config not in latency_data or config not in longbench_data:
            continue
        
        # Get speedup at reference length
        speedup = 1.0
        if ref_length in latency_data[config]:
            speedup_data = latency_data[config][ref_length].get('speedup', {})
            speedup = speedup_data.get('total', 1.0)
        
        # Get hit rate
        hit_rate = longbench_data[config]['hit_rate']
        
        points.append({
            'config': CONFIG_NAMES[config],
            'hit_rate': hit_rate,
            'speedup': speedup,
            'color': COLORS[config],
            'marker': 's',
            'size': 120
        })
    
    # Plot points
    for point in points:
        ax.scatter(point['hit_rate'], point['speedup'], 
                  c=point['color'], marker=point['marker'], 
                  s=point['size'], alpha=0.7, edgecolors='black', 
                  linewidths=1.5, zorder=3, label=point['config'])
    
    # Add annotations
    for point in points:
        if point['config'] != 'Baseline':
            # Create shorter label
            label = point['config'].replace('SDTP (keep=', 'k=').replace(')', '')
            ax.annotate(label, 
                       (point['hit_rate'], point['speedup']),
                       xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    # Add reference lines
    if 'baseline' in longbench_data:
        baseline_hit = longbench_data['baseline']['hit_rate']
        ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(x=baseline_hit, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # Set axis limits to include all points with some padding
    if points:
        hit_rates = [p['hit_rate'] for p in points]
        speedups = [p['speedup'] for p in points]
        ax.set_xlim([min(hit_rates) - 0.01, max(hit_rates) + 0.01])
        ax.set_ylim([max(0.9, min(speedups) - 0.1), max(speedups) + 0.2])
    
    ax.set_xlabel('Hit Rate (LongBench HotpotQA)', fontsize=12)
    ax.set_ylabel(f'End2End Speedup (×) at {ref_length} tokens', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    plt.tight_layout()
    out_path = os.path.join(out_dir, "phase2_performance_speed_tradeoff.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {out_path}")
    plt.close()


def plot_longbench_performance(longbench_data: Dict, out_dir: str = "results/fig"):
    """
    Plot 4: LongBench performance retention.
    Bar chart showing Hit Rate for different configurations.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    configs = ['baseline', 'keep09', 'keep08', 'keep07']
    config_labels = ['Baseline', 'SDTP\n(keep=0.9)', 'SDTP\n(keep=0.8)', 'SDTP\n(keep=0.7)']
    hit_rates = []
    colors_list = []
    
    for config in configs:
        if config in longbench_data:
            hit_rate = longbench_data[config]['hit_rate']
            hit_rates.append(hit_rate)
            colors_list.append(COLORS[config])
        else:
            hit_rates.append(0)
            colors_list.append('#CCCCCC')
    
    bars = ax.bar(config_labels, hit_rates, color=colors_list, alpha=0.8, 
                  edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for i, (bar, rate) in enumerate(zip(bars, hit_rates)):
        if rate > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{rate:.3f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            # Add percentage change from baseline
            if i > 0 and hit_rates[0] > 0:
                change = ((rate - hit_rates[0]) / hit_rates[0]) * 100
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'({change:+.1f}%)',
                       ha='center', va='bottom', fontsize=9, style='italic', color='gray')
    
    # Add baseline reference line
    if hit_rates[0] > 0:
        ax.axhline(y=hit_rates[0], color=COLORS['baseline'], linestyle='--', 
                  linewidth=2, alpha=0.5, label='Baseline Reference')
    
    ax.set_ylabel('Hit Rate', fontsize=12)
    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylim([0, max(hit_rates) * 1.15 if hit_rates else 0.4])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    out_path = os.path.join(out_dir, "phase2_longbench_performance.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {out_path}")
    plt.close()


def plot_comprehensive_heatmap(latency_data: Dict, out_dir: str = "results/fig"):
    """
    Plot 5: Comprehensive heatmap showing speedup across configurations and sequence lengths.
    """
    configs = ['keep09', 'keep08', 'keep07']
    lengths = sorted(set(length for config_data in latency_data.values() 
                         for length in config_data.keys()))
    
    # Prepare data matrix
    data_matrix = []
    row_labels = ['Baseline'] + [CONFIG_NAMES[c] for c in configs]
    
    # Baseline row (all 1.0x)
    baseline_row = [1.0] * len(lengths)
    data_matrix.append(baseline_row)
    
    # SDTP rows
    for config in configs:
        if config not in latency_data:
            data_matrix.append([1.0] * len(lengths))
            continue
        
        row = []
        for length in lengths:
            if length in latency_data[config]:
                speedup_data = latency_data[config][length].get('speedup', {})
                speedup = speedup_data.get('total', 1.0)
                row.append(speedup)
            else:
                row.append(1.0)
        data_matrix.append(row)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Convert to numpy array
    data_array = np.array(data_matrix)
    
    # Create custom colormap (green for speedup > 1, red for < 1)
    from matplotlib.colors import LinearSegmentedColormap
    colors = ['#d62728', '#ffffff', '#2ca02c']  # Red, White, Green
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('speedup', colors, N=n_bins)
    
    im = ax.imshow(data_array, cmap=cmap, aspect='auto', vmin=0.5, vmax=2.0)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(lengths)))
    ax.set_xticklabels([f'{l//1024}K' if l >= 1024 else str(l) for l in lengths])
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    
    # Add text annotations
    for i in range(len(row_labels)):
        for j in range(len(lengths)):
            text = ax.text(j, i, f'{data_array[i, j]:.2f}×',
                          ha="center", va="center", color="black", fontsize=9,
                          fontweight='bold' if data_array[i, j] != 1.0 else 'normal')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Speedup (×)', rotation=270, labelpad=20)
    
    ax.set_xlabel('Sequence Length', fontsize=12)
    ax.set_ylabel('Configuration', fontsize=12)
    
    plt.tight_layout()
    out_path = os.path.join(out_dir, "phase2_speedup_heatmap.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {out_path}")
    plt.close()


def plot_comprehensive_config_comparison(latency_data: Dict, longbench_data: Dict,
                                       out_dir: str = "results/fig", ref_length: int = 16384):
    """
    Plot: Comprehensive configuration comparison.
    Combines multiple metrics in a 2x2 subplot layout:
    - Hit Rate (LongBench performance)
    - End2End Speedup
    - KV Cache Reduction
    - Performance-Efficiency Trade-off (bubble chart)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    configs = ['baseline', 'keep09', 'keep08', 'keep07']
    config_labels = ['Baseline', 'SDTP\n(keep=0.9)', 'SDTP\n(keep=0.8)', 'SDTP\n(keep=0.7)']
    
    # Prepare data for all subplots
    hit_rates = []
    speedups = []
    kv_reductions = []
    colors_list = []
    
    for config in configs:
        if config == 'baseline':
            hit_rates.append(longbench_data.get(config, {}).get('hit_rate', 0))
            speedups.append(1.0)
            kv_reductions.append(0.0)
            colors_list.append(COLORS['baseline'])
        else:
            hit_rates.append(longbench_data.get(config, {}).get('hit_rate', 0))
            # Get speedup at reference length
            speedup = 1.0
            kv_reduction = 0.0
            if config in latency_data and ref_length in latency_data[config]:
                speedup_data = latency_data[config][ref_length].get('speedup', {})
                speedup = speedup_data.get('total', 1.0)
                kv_reduction = latency_data[config][ref_length].get('kv_reduction', 0.0) * 100
            speedups.append(speedup)
            kv_reductions.append(kv_reduction)
            colors_list.append(COLORS[config])
    
    # Subplot 1: Hit Rate (LongBench Performance)
    ax = axes[0, 0]
    bars = ax.bar(config_labels, hit_rates, color=colors_list, alpha=0.8, 
                  edgecolor='black', linewidth=1.5)
    for i, (bar, rate) in enumerate(zip(bars, hit_rates)):
        if rate > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{rate:.3f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
            if i > 0 and hit_rates[0] > 0:
                change = ((rate - hit_rates[0]) / hit_rates[0]) * 100
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'({change:+.1f}%)',
                       ha='center', va='bottom', fontsize=8, style='italic', color='gray')
    if hit_rates[0] > 0:
        ax.axhline(y=hit_rates[0], color=COLORS['baseline'], linestyle='--', 
                  linewidth=2, alpha=0.5)
    ax.set_ylabel('Hit Rate', fontsize=11)
    ax.set_title('(a) LongBench Performance (HotpotQA)', fontsize=12, fontweight='bold')
    ax.set_ylim([0, max(hit_rates) * 1.15 if hit_rates else 0.4])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Subplot 2: End2End Speedup
    ax = axes[0, 1]
    bars = ax.bar(config_labels, speedups, color=colors_list, alpha=0.8, 
                  edgecolor='black', linewidth=1.5)
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    for i, (bar, speedup) in enumerate(zip(bars, speedups)):
        if speedup > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{speedup:.2f}×',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylabel(f'Speedup (×) at {ref_length//1024}K tokens', fontsize=11)
    ax.set_title('(b) End2End Speedup', fontsize=12, fontweight='bold')
    ax.set_ylim([0, max(speedups) * 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Subplot 3: KV Cache Reduction
    ax = axes[1, 0]
    bars = ax.bar(config_labels, kv_reductions, color=colors_list, alpha=0.8, 
                  edgecolor='black', linewidth=1.5)
    for i, (bar, reduction) in enumerate(zip(bars, kv_reductions)):
        if reduction > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{reduction:.1f}%',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylabel('KV Cache Reduction (%)', fontsize=11)
    ax.set_title('(c) KV Cache Compression', fontsize=12, fontweight='bold')
    ax.set_ylim([0, max(kv_reductions) * 1.1 if kv_reductions else 100])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Subplot 4: Performance-Efficiency Trade-off (Bubble Chart)
    ax = axes[1, 1]
    # Plot baseline point
    if hit_rates[0] > 0:
        ax.scatter(hit_rates[0], speedups[0], s=200, c=COLORS['baseline'], 
                  marker='o', alpha=0.7, edgecolors='black', linewidths=2, 
                  zorder=3, label='Baseline')
        ax.annotate('Baseline', (hit_rates[0], speedups[0]),
                   xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    # Plot SDTP points with bubble size = KV Cache reduction
    for i, config in enumerate(['keep09', 'keep08', 'keep07']):
        if i+1 < len(hit_rates) and hit_rates[i+1] > 0:
            # Bubble size proportional to KV cache reduction (scaled)
            bubble_size = kv_reductions[i+1] * 10 if kv_reductions[i+1] > 0 else 100
            ax.scatter(hit_rates[i+1], speedups[i+1], s=bubble_size, 
                      c=COLORS[config], marker='s', alpha=0.6, 
                      edgecolors='black', linewidths=1.5, zorder=3,
                      label=CONFIG_NAMES[config])
            # Annotation
            label = f'k={config[-2:]}'
            ax.annotate(label, (hit_rates[i+1], speedups[i+1]),
                       xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    # Add reference lines
    if hit_rates[0] > 0:
        ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(x=hit_rates[0], color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # Set axis limits
    if hit_rates and speedups:
        ax.set_xlim([min(hit_rates) - 0.01, max(hit_rates) + 0.01])
        ax.set_ylim([max(0.9, min(speedups) - 0.1), max(speedups) + 0.2])
    
    ax.set_xlabel('Hit Rate (LongBench HotpotQA)', fontsize=11)
    ax.set_ylabel(f'End2End Speedup (×) at {ref_length//1024}K tokens', fontsize=11)
    ax.set_title('(d) Performance-Efficiency Trade-off\n(Bubble size = KV Cache Reduction)', 
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    # No legend needed - points are already annotated with labels
    
    plt.tight_layout()
    out_path = os.path.join(out_dir, "phase2_comprehensive_config_comparison.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {out_path}")
    plt.close()


def plot_sample_level_analysis(results_dir: str = "results", out_dir: str = "results/fig"):
    """
    Plot: Sample-level analysis for hotpotqa_sdtp_0.9.json.
    Visualizes pruning statistics across 200 samples with multiple subplots.
    """
    import re
    
    json_path = os.path.join(results_dir, "hotpotqa_sdtp_0.9.json")
    if not os.path.exists(json_path):
        print(f"[Warning] File not found: {json_path}")
        return
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract PRUNE logs
    prune_logs = [log for log in data.get('stdout_logs', []) if '[PRUNE]' in log]
    
    # Parse pruning data
    pruning_records = []
    for log in prune_logs:
        match = re.search(r'Layer (\d+): seq_len=(\d+), keep_k=(\d+), kept=(\d+), pruned=(\d+), ratio=([\d.]+)%', log)
        if match:
            layer = int(match.group(1))
            seq_len = int(match.group(2))
            keep_k = int(match.group(3))
            kept = int(match.group(4))
            pruned = int(match.group(5))
            ratio = float(match.group(6))
            pruning_records.append({
                'layer': layer,
                'seq_len': seq_len,
                'keep_k': keep_k,
                'kept': kept,
                'pruned': pruned,
                'ratio': ratio
            })
    
    if not pruning_records:
        print("[Warning] No pruning records found in logs")
        return
    
    # Group by sample: each sample starts with Layer 4
    # We identify new samples when we see Layer 4 after a complete set of layers
    samples = []
    current_sample = []
    seen_layers = set()
    
    for record in pruning_records:
        layer = record['layer']
        
        # If we see Layer 4 and we already have a complete sample, start a new one
        if layer == 4 and len(current_sample) > 0 and len(seen_layers) == 8:
            samples.append(current_sample)
            current_sample = [record]
            seen_layers = {4}
        elif layer == 4 and len(current_sample) == 0:
            # First sample
            current_sample = [record]
            seen_layers = {4}
        else:
            current_sample.append(record)
            seen_layers.add(layer)
    
    # Add last sample
    if current_sample:
        samples.append(current_sample)
    
    # Filter to samples with all 8 layers and limit to 200
    complete_samples = [s for s in samples if len(s) >= 8]
    samples = complete_samples[:200]
    
    # Calculate per-sample statistics
    sample_stats = []
    for i, sample in enumerate(samples):
        if not sample:
            continue
        # Get initial sequence length (from layer 4)
        initial_seq_len = sample[0]['seq_len'] if sample else 0
        # Get final kept tokens (from last layer)
        final_kept = sample[-1]['kept'] if sample else 0
        # Calculate average pruning ratio
        avg_prune_ratio = sum(r['ratio'] for r in sample) / len(sample) if sample else 0
        # Count layers with actual pruning
        layers_pruned = sum(1 for r in sample if r['pruned'] > 0)
        
        sample_stats.append({
            'sample_id': i,
            'initial_seq_len': initial_seq_len,
            'final_kept': final_kept,
            'avg_prune_ratio': avg_prune_ratio,
            'layers_pruned': layers_pruned,
            'total_reduction': (initial_seq_len - final_kept) / initial_seq_len * 100 if initial_seq_len > 0 else 0
        })
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35, 
                         left=0.06, right=0.98, top=0.98, bottom=0.08)
    
    # Subplot 1: Initial Sequence Length Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    initial_lengths = [s['initial_seq_len'] for s in sample_stats]
    ax1.hist(initial_lengths, bins=20, color=COLORS['keep09'], alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Initial Sequence Length', fontsize=10)
    ax1.set_ylabel('Frequency', fontsize=10)
    ax1.set_title('(a) Input Length Distribution', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axvline(np.mean(initial_lengths), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(initial_lengths):.1f}')
    ax1.legend(fontsize=9)
    
    # Subplot 2: Final Kept Tokens Distribution
    ax2 = fig.add_subplot(gs[0, 1])
    final_kept = [s['final_kept'] for s in sample_stats]
    ax2.hist(final_kept, bins=20, color=COLORS['keep09'], alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Final Kept Tokens', fontsize=10)
    ax2.set_ylabel('Frequency', fontsize=10)
    ax2.set_title('(b) Final KV Cache Size Distribution', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axvline(np.mean(final_kept), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(final_kept):.1f}')
    ax2.legend(fontsize=9)
    
    # Subplot 3: Total Reduction Rate Distribution
    ax3 = fig.add_subplot(gs[0, 2])
    reductions = [s['total_reduction'] for s in sample_stats]
    ax3.hist(reductions, bins=20, color=COLORS['keep09'], alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Total Reduction Rate (%)', fontsize=10)
    ax3.set_ylabel('Frequency', fontsize=10)
    ax3.set_title('(c) Overall Compression Rate Distribution', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.axvline(np.mean(reductions), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(reductions):.1f}%')
    ax3.legend(fontsize=9)
    
    # Subplot 4: Average Pruning Ratio per Sample
    ax4 = fig.add_subplot(gs[1, 0])
    avg_ratios = [s['avg_prune_ratio'] for s in sample_stats]
    ax4.scatter(range(len(sample_stats)), avg_ratios, 
               c=COLORS['keep09'], alpha=0.6, s=20, edgecolors='black', linewidths=0.5)
    ax4.set_xlabel('Sample ID', fontsize=10)
    ax4.set_ylabel('Average Pruning Ratio (%)', fontsize=10)
    ax4.set_title('(d) Pruning Ratio per Sample', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(np.mean(avg_ratios), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(avg_ratios):.2f}%')
    ax4.legend(fontsize=9)
    
    # Subplot 5: Initial Length vs Final Kept (Scatter)
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.scatter(initial_lengths, final_kept, 
               c=COLORS['keep09'], alpha=0.6, s=30, edgecolors='black', linewidths=0.5)
    # Add diagonal line (no compression)
    max_len = max(max(initial_lengths), max(final_kept))
    ax5.plot([0, max_len], [0, max_len], 'r--', linewidth=1, alpha=0.5, label='No compression')
    ax5.set_xlabel('Initial Sequence Length', fontsize=10)
    ax5.set_ylabel('Final Kept Tokens', fontsize=10)
    ax5.set_title('(e) Compression Effect', fontsize=11, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend(fontsize=9)
    
    # Subplot 6: Layers with Pruning per Sample
    ax6 = fig.add_subplot(gs[1, 2])
    layers_pruned = [s['layers_pruned'] for s in sample_stats]
    ax6.hist(layers_pruned, bins=range(0, 10), color=COLORS['keep09'], 
            alpha=0.7, edgecolor='black', align='left')
    ax6.set_xlabel('Number of Layers Pruned', fontsize=10)
    ax6.set_ylabel('Frequency', fontsize=10)
    ax6.set_title('(f) Active Pruning Layers Distribution', fontsize=11, fontweight='bold')
    ax6.set_xticks(range(0, 9))
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Subplot 7: Pruning Ratio by Layer (Box plot)
    ax7 = fig.add_subplot(gs[2, :])
    layer_ratios = {}
    for record in pruning_records:
        layer = record['layer']
        if layer not in layer_ratios:
            layer_ratios[layer] = []
        layer_ratios[layer].append(record['ratio'])
    
    layers = sorted(layer_ratios.keys())
    data_to_plot = [layer_ratios[l] for l in layers]
    bp = ax7.boxplot(data_to_plot, labels=[f'Layer {l}' for l in layers], 
                     patch_artist=True, widths=0.6)
    for patch in bp['boxes']:
        patch.set_facecolor(COLORS['keep09'])
        patch.set_alpha(0.7)
    
    ax7.set_xlabel('Pruning Layer', fontsize=11)
    ax7.set_ylabel('Pruning Ratio (%)', fontsize=11)
    ax7.set_title('(g) Pruning Ratio Distribution by Layer', fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3, axis='y')
    
    # Don't use tight_layout with gridspec, use manual spacing instead
    out_path = os.path.join(out_dir, "phase2_sample_level_analysis_keep09.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {out_path}")
    plt.close()


def main():
    """Main function to generate all Phase 2 visualization figures."""
    parser = argparse.ArgumentParser(description='Generate Phase 2 evaluation visualization figures')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Directory containing JSON result files')
    parser.add_argument('--out_dir', type=str, default='results/fig',
                       help='Output directory for figures')
    parser.add_argument('--ref_length', type=int, default=16384,
                       help='Reference sequence length for trade-off plot')
    
    args = parser.parse_args()
    
    # Setup plot style
    setup_plot_style()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("[Info] Loading latency data...")
    latency_data = load_latency_data(args.results_dir)
    if not latency_data:
        print("[Error] No latency data found. Please check results directory.")
        return
    
    print("[Info] Loading LongBench data...")
    longbench_data = load_longbench_data(args.results_dir)
    if not longbench_data:
        print("[Warning] No LongBench data found. Some plots will be skipped.")
    
    print("\n[Info] Generating figures...")
    
    # Generate all figures
    print("\n1. Generating End2End latency comparison...")
    plot_end2end_latency_comparison(latency_data, args.out_dir)
    
    print("\n2. Generating KV Cache reduction plot...")
    plot_kv_cache_reduction(latency_data, args.out_dir)
    
    if longbench_data:
        print("\n3. Generating comprehensive config comparison...")
        plot_comprehensive_config_comparison(latency_data, longbench_data,
                                           args.out_dir, args.ref_length)
        
        print("\n4. Generating sample-level analysis for hotpotqa_sdtp_0.9...")
        plot_sample_level_analysis(args.results_dir, args.out_dir)
    
    print("\n5. Generating comprehensive heatmap...")
    plot_comprehensive_heatmap(latency_data, args.out_dir)
    
    print("\n[OK] All figures generated successfully!")
    print(f"[Info] Figures saved to: {args.out_dir}")


if __name__ == "__main__":
    main()

