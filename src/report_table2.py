# Generate Table 2 style report from benchmark results
# Input: JSON results from baseline and SDTP runs
# Output: Markdown table with Prefill Speedup, End2End Speedup, KV Reduction

import json
import argparse
from typing import Dict, List, Optional


def load_results(json_path: str) -> Dict:
    """Load results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def extract_metrics(results: Dict, length: int, mode: str = "end2end") -> Dict:
    """
    Extract metrics for a given length.
    
    Args:
        results: Results dictionary from JSON
        length: Sequence length to extract
        mode: "prefill" or "end2end"
        
    Returns:
        Dictionary with extracted metrics
    """
    length_str = str(length)
    if length_str not in results.get("results", {}):
        return None
    
    result = results["results"][length_str]
    
    if mode == "end2end":
        # End2End mode
        baseline = result.get("baseline", {})
        sdtp = result.get("sdtp", {})
        speedup = result.get("speedup", {})
        
        baseline_prefill = baseline.get("prefill_latency_seconds", 0)
        baseline_total = baseline.get("total_latency_seconds", 0)
        sdtp_prefill = sdtp.get("prefill_latency_seconds", 0)
        sdtp_total = sdtp.get("total_latency_seconds", 0)
        
        prefill_speedup = speedup.get("prefill", 0)
        total_speedup = speedup.get("total", 0)
        
        # KV cache reduction
        baseline_kv = baseline.get("kv_lens_after_prefill", [])
        sdtp_kv = sdtp.get("kv_lens_after_prefill", [])
        
        if baseline_kv and sdtp_kv:
            baseline_kv_avg = sum(baseline_kv) / len(baseline_kv) if baseline_kv else length
            sdtp_kv_avg = sum(sdtp_kv) / len(sdtp_kv) if sdtp_kv else length
            kv_reduction = (baseline_kv_avg - sdtp_kv_avg) / baseline_kv_avg * 100 if baseline_kv_avg > 0 else 0
        else:
            kv_reduction = result.get("kv_reduction", 0) * 100
        
        return {
            "prefill_speedup": prefill_speedup,
            "end2end_speedup": total_speedup,
            "kv_reduction": kv_reduction,
            "baseline_prefill": baseline_prefill,
            "baseline_total": baseline_total,
            "sdtp_prefill": sdtp_prefill,
            "sdtp_total": sdtp_total,
        }
    else:
        # Prefill-only mode
        baseline = result.get("baseline", {})
        sdtp = result.get("sdtp", {})
        speedup = result.get("speedup", 0)
        
        baseline_latency = baseline.get("latency_seconds", 0)
        sdtp_latency = sdtp.get("latency_seconds", 0)
        
        return {
            "prefill_speedup": speedup,
            "end2end_speedup": None,  # Not available in prefill-only mode
            "kv_reduction": None,  # Not available in prefill-only mode
            "baseline_prefill": baseline_latency,
            "baseline_total": None,
            "sdtp_prefill": sdtp_latency,
            "sdtp_total": None,
        }


def format_length(length: int) -> str:
    """Format length for display."""
    if length >= 1024:
        return f"{length // 1024}K"
    return str(length)


def generate_table2_markdown(
    baseline_results: Dict,
    sdtp09_results: Optional[Dict],
    sdtp08_results: Optional[Dict],
    sdtp07_results: Optional[Dict],
    lengths: List[int],
    mode: str = "end2end",
) -> str:
    """
    Generate Table 2 style markdown table.
    
    Args:
        baseline_results: Baseline results JSON
        sdtp09_results: SDTP keep09 results JSON (optional)
        sdtp08_results: SDTP keep08 results JSON (optional)
        sdtp07_results: SDTP keep07 results JSON (optional)
        lengths: List of sequence lengths to include
        mode: "prefill" or "end2end"
        
    Returns:
        Markdown formatted table string
    """
    lines = []
    
    # Table header
    if mode == "end2end":
        lines.append("| Length | Method | Prefill (s) | Decode (s) | Total (s) | Prefill Speedup (×) | End2End Speedup (×) | KV Reduction (%) |")
        lines.append("|--------|--------|-------------|------------|----------|---------------------|---------------------|------------------|")
    else:
        lines.append("| Length | Method | Prefill (s) | Speedup (×) |")
        lines.append("|--------|--------|-------------|-------------|")
    
    # Process each length
    for length in lengths:
        # Baseline row
        baseline_metrics = extract_metrics(baseline_results, length, mode)
        if baseline_metrics:
            if mode == "end2end":
                lines.append(
                    f"| {format_length(length)} | Baseline | "
                    f"{baseline_metrics['baseline_prefill']:.4f} | "
                    f"{baseline_metrics['baseline_total'] - baseline_metrics['baseline_prefill']:.4f} | "
                    f"{baseline_metrics['baseline_total']:.4f} | - | - | - |"
                )
            else:
                lines.append(
                    f"| {format_length(length)} | Baseline | "
                    f"{baseline_metrics['baseline_prefill']:.4f} | - |"
                )
        
        # SDTP rows
        for config_name, config_results, config_label in [
            ("keep09", sdtp09_results, "SDTP (keep09)"),
            ("keep08", sdtp08_results, "SDTP (keep08)"),
            ("keep07", sdtp07_results, "SDTP (keep07)"),
        ]:
            if config_results is None:
                continue
            
            sdtp_metrics = extract_metrics(config_results, length, mode)
            if sdtp_metrics:
                if mode == "end2end":
                    lines.append(
                        f"| {format_length(length)} | {config_label} | "
                        f"{sdtp_metrics['sdtp_prefill']:.4f} | "
                        f"{sdtp_metrics['sdtp_total'] - sdtp_metrics['sdtp_prefill']:.4f} | "
                        f"{sdtp_metrics['sdtp_total']:.4f} | "
                        f"{sdtp_metrics['prefill_speedup']:.2f} | "
                        f"{sdtp_metrics['end2end_speedup']:.2f} | "
                        f"{sdtp_metrics['kv_reduction']:.2f}% |"
                    )
                else:
                    lines.append(
                        f"| {format_length(length)} | {config_label} | "
                        f"{sdtp_metrics['sdtp_prefill']:.4f} | "
                        f"{sdtp_metrics['prefill_speedup']:.2f} |"
                    )
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Table 2 style report from benchmark results"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        help="Path to baseline results JSON file",
    )
    parser.add_argument(
        "--sdtp09",
        type=str,
        default=None,
        help="Path to SDTP keep09 results JSON file",
    )
    parser.add_argument(
        "--sdtp08",
        type=str,
        default=None,
        help="Path to SDTP keep08 results JSON file",
    )
    parser.add_argument(
        "--sdtp07",
        type=str,
        default=None,
        help="Path to SDTP keep07 results JSON file",
    )
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[1024, 2048, 4096, 8192, 16384, 32768],
        help="Sequence lengths to include in table",
    )
    parser.add_argument(
        "--mode",
        choices=["prefill", "end2end"],
        default="end2end",
        help="Benchmark mode: 'prefill' or 'end2end'",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: print to stdout)",
    )
    
    args = parser.parse_args()
    
    # Load results
    baseline = load_results(args.baseline)
    sdtp09 = load_results(args.sdtp09) if args.sdtp09 else None
    sdtp08 = load_results(args.sdtp08) if args.sdtp08 else None
    sdtp07 = load_results(args.sdtp07) if args.sdtp07 else None
    
    # Generate table
    table = generate_table2_markdown(
        baseline, sdtp09, sdtp08, sdtp07, args.lengths, args.mode
    )
    
    # Output
    if args.output:
        with open(args.output, 'w') as f:
            f.write(table)
        print(f"[OK] Table saved to {args.output}")
    else:
        print(table)


if __name__ == "__main__":
    main()

