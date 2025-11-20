"""
Latency curve plotting script for SDTP
Generates latency, speedup, and FLOPs reduction curves
"""
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os


def load_json(path):
    """
    Load JSON file and extract latency data.
    Supports both old format ({length: latency}) and new format (with metadata/results).
    
    Returns:
        dict: {length: latency} mapping
        dict: Metadata if available (new format), None otherwise
    """
    if not os.path.exists(path):
        print(f"[Warning] File not found: {path}")
        return {}, None
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    # Check if it's new format (has 'results' or 'metadata' key)
    if "results" in data:
        # New format: extract latency from results
        latency_data = {}
        for length_str, result in data["results"].items():
            if isinstance(result, dict) and "baseline" in result:
                # Combined format: extract baseline latency
                latency_data[int(length_str)] = result["baseline"]["latency_seconds"]
            elif isinstance(result, dict) and "latency_seconds" in result:
                # Separate baseline/sdtp format
                latency_data[int(length_str)] = result["latency_seconds"]
            else:
                # Fallback: assume it's a number
                latency_data[int(length_str)] = float(result)
        metadata = data.get("metadata", None)
        return latency_data, metadata
    elif "metadata" in data:
        # New format but only metadata (shouldn't happen, but handle it)
        return {}, data.get("metadata", None)
    else:
        # Old format: simple {length: latency}
        return {int(k): float(v) for k, v in data.items()}, None


def plot_latency(baseline, sdtp, out_path):
    """
    Plot prefill latency vs sequence length
    
    Args:
        baseline: Dict mapping sequence length to latency (seconds)
        sdtp: Dict mapping sequence length to latency (seconds)
        out_path: Output file path
    """
    # Get common lengths
    lengths = sorted(set(baseline.keys()) & set(sdtp.keys()))
    if not lengths:
        print("[Error] No common sequence lengths found")
        return
    
    base_vals = [baseline[L] for L in lengths]
    sdtp_vals = [sdtp[L] for L in lengths]
    
    plt.figure(figsize=(10, 6))
    plt.plot(lengths, base_vals, marker='o', linewidth=2, label="Baseline", markersize=8)
    plt.plot(lengths, sdtp_vals, marker='s', linewidth=2, label="SDTP", markersize=8)
    plt.xlabel("Sequence Length (tokens)", fontsize=12)
    plt.ylabel("Prefill Latency (seconds)", fontsize=12)
    plt.title("Prefill Latency vs Sequence Length", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Saved {out_path}")


def plot_speedup(baseline, sdtp, out_path):
    """
    Plot speedup vs sequence length
    
    Args:
        baseline: Dict mapping sequence length to latency (seconds)
        sdtp: Dict mapping sequence length to latency (seconds)
        out_path: Output file path
    """
    # Get common lengths
    lengths = sorted(set(baseline.keys()) & set(sdtp.keys()))
    if not lengths:
        print("[Error] No common sequence lengths found")
        return
    
    speedups = [baseline[L] / sdtp[L] if sdtp[L] > 0 else 0 for L in lengths]
    
    plt.figure(figsize=(10, 6))
    plt.plot(lengths, speedups, marker='o', linewidth=2, label="Speedup", 
             markersize=8, color='green')
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Baseline (1x)')
    plt.xlabel("Sequence Length (tokens)", fontsize=12)
    plt.ylabel("Speedup (Baseline / SDTP)", fontsize=12)
    plt.title("Speedup vs Sequence Length", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Saved {out_path}")
    print(f"[Info] Average speedup: {np.mean(speedups):.2f}x")


def estimate_flops(length, keep_ratio=0.7, hidden_size=3584, num_layers=28, num_heads=32):
    """
    Estimate FLOPs for Transformer forward pass
    
    Args:
        length: Sequence length
        keep_ratio: Token keep ratio (for SDTP)
        hidden_size: Model hidden size
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        head_dim: Attention head dimension
    
    Returns:
        Estimated FLOPs (normalized)
    """
    head_dim = hidden_size // num_heads
    
    # Attention FLOPs: 4 * L^2 * d (QK^T, softmax, AV)
    # MLP FLOPs: 8 * L * d^2 (two linear layers, expansion ratio ~4)
    # Per layer FLOPs
    attn_flops = 4 * length * length * hidden_size
    mlp_flops = 8 * length * hidden_size * hidden_size
    
    # Total FLOPs for all layers
    total_flops = num_layers * (attn_flops + mlp_flops)
    
    return total_flops


def plot_flops(baseline, sdtp, out_path, keep_ratio=0.7):
    """
    Plot estimated FLOPs reduction
    
    Args:
        baseline: Dict mapping sequence length to latency (for reference)
        sdtp: Dict mapping sequence length to latency (for reference)
        out_path: Output file path
        keep_ratio: Average token keep ratio for SDTP
    """
    # Get common lengths
    lengths = sorted(set(baseline.keys()) & set(sdtp.keys()))
    if not lengths:
        print("[Error] No common sequence lengths found")
        return
    
    # Estimate FLOPs
    base_flops = [estimate_flops(L) for L in lengths]
    
    # SDTP FLOPs: tokens are pruned progressively, use average keep ratio
    # For simplicity, use keep_ratio for all layers (in reality it's progressive)
    sdtp_flops = [estimate_flops(int(L * keep_ratio)) for L in lengths]
    
    # Normalize to first value for better visualization
    if base_flops[0] > 0:
        base_flops_norm = [f / base_flops[0] for f in base_flops]
        sdtp_flops_norm = [f / base_flops[0] for f in sdtp_flops]
    else:
        base_flops_norm = base_flops
        sdtp_flops_norm = sdtp_flops
    
    reduction = [(1 - sdtp_flops[i] / base_flops[i]) * 100 
                 for i in range(len(lengths))]
    
    plt.figure(figsize=(10, 6))
    plt.plot(lengths, base_flops_norm, marker='o', linewidth=2, 
             label="Baseline FLOPs", markersize=8)
    plt.plot(lengths, sdtp_flops_norm, marker='s', linewidth=2, 
             label=f"SDTP FLOPs (keep_ratio={keep_ratio})", markersize=8)
    plt.xlabel("Sequence Length (tokens)", fontsize=12)
    plt.ylabel("Relative FLOPs (normalized)", fontsize=12)
    plt.title("Estimated FLOPs Reduction", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Saved {out_path}")
    print(f"[Info] Average FLOPs reduction: {np.mean(reduction):.1f}%")


def load_combined_json(path):
    """
    Load combined JSON file (new format with both baseline and SDTP).
    
    Returns:
        tuple: (baseline_dict, sdtp_dict, metadata)
    """
    if not os.path.exists(path):
        print(f"[Warning] File not found: {path}")
        return {}, {}, None
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    if "results" not in data:
        print(f"[Warning] Invalid format in {path}, expected 'results' key")
        return {}, {}, None
    
    baseline = {}
    sdtp = {}
    
    for length_str, result in data["results"].items():
        length = int(length_str)
        if isinstance(result, dict):
            if "baseline" in result:
                baseline[length] = result["baseline"]["latency_seconds"]
            if "sdtp" in result:
                sdtp[length] = result["sdtp"]["latency_seconds"]
    
    metadata = data.get("metadata", None)
    return baseline, sdtp, metadata


def main():
    parser = argparse.ArgumentParser(
        description="Generate latency curves for SDTP evaluation"
    )
    parser.add_argument(
        "--baseline", 
        type=str, 
        default=None,
        help="Path to baseline latency JSON file (old format or separate file)"
    )
    parser.add_argument(
        "--sdtp", 
        type=str, 
        default=None,
        help="Path to SDTP latency JSON file (old format or separate file)"
    )
    parser.add_argument(
        "--combined",
        type=str,
        default=None,
        help="Path to combined results JSON file (new format with both baseline and SDTP)"
    )
    parser.add_argument(
        "--out_dir", 
        type=str, 
        default="results/fig",
        help="Output directory for figures"
    )
    parser.add_argument(
        "--keep_ratio",
        type=float,
        default=None,
        help="Token keep ratio for FLOPs estimation (auto-detect from metadata if not set)"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Prefix for output filenames (e.g., 'singlegpu_' or 'multigpu_')"
    )
    
    args = parser.parse_args()
    
    # Load data - support both old and new formats
    baseline = {}
    sdtp = {}
    metadata = None
    
    if args.combined:
        # New format: load from combined file
        print(f"[Loading] Combined results: {args.combined}")
        baseline, sdtp, metadata = load_combined_json(args.combined)
        if metadata and args.keep_ratio is None:
            # Auto-detect keep_ratio from metadata
            args.keep_ratio = metadata.get("pruning_config", {}).get("keep_ratio", 0.7)
    else:
        # Old format: load separate files
        if args.baseline is None:
            args.baseline = "results/latency_baseline.json"
        if args.sdtp is None:
            args.sdtp = "results/latency_sdtp.json"
        
        print(f"[Loading] Baseline: {args.baseline}")
        baseline, baseline_meta = load_json(args.baseline)
        
        print(f"[Loading] SDTP: {args.sdtp}")
        sdtp, sdtp_meta = load_json(args.sdtp)
        
        metadata = baseline_meta or sdtp_meta
        if metadata and args.keep_ratio is None:
            args.keep_ratio = metadata.get("pruning_config", {}).get("keep_ratio", 0.7)
    
    if not baseline:
        print("[Error] Baseline data is empty")
        return
    
    if not sdtp:
        print("[Error] SDTP data is empty")
        return
    
    if args.keep_ratio is None:
        args.keep_ratio = 0.7  # Default fallback
    
    print(f"[Info] Found {len(baseline)} baseline points, {len(sdtp)} SDTP points")
    if metadata:
        config_name = metadata.get("config_name", "unknown")
        print(f"[Info] Configuration: {config_name}")
    
    # Generate plots with optional prefix
    prefix = f"{args.prefix}_" if args.prefix else ""
    plot_latency(baseline, sdtp, os.path.join(args.out_dir, f"{prefix}latency_curve.png"))
    plot_speedup(baseline, sdtp, os.path.join(args.out_dir, f"{prefix}speedup_curve.png"))
    plot_flops(baseline, sdtp, os.path.join(args.out_dir, f"{prefix}flops_curve.png"), 
               keep_ratio=args.keep_ratio)
    
    print("[OK] All plots generated successfully!")


if __name__ == "__main__":
    main()

