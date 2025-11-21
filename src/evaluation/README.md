# Evaluation Tools

This directory contains evaluation and visualization tools for SDTP.

## Files

- `sdtp_wrapper.py`: SDTP inference wrapper class
- `longbench_eval.py`: LongBench evaluation script
- `lmeval_runner.py`: lm-eval-harness runner
- `ablation.py`: Ablation study script
- `plot_latency.py`: Latency curve plotting script
- `plot_phase2_results.py`: **Phase 2 comprehensive visualization script** (publication-quality figures)
- `parse_latency_log.py`: Log file parser for latency data

## Usage

### Plotting Latency Curves

1. **Prepare JSON data files:**
   - `results/latency_baseline.json`: Baseline latency data
   - `results/latency_sdtp.json`: SDTP latency data
   
   Format:
   ```json
   {
     "4096": 0.7065,
     "8192": 1.2684,
     "16384": 2.3311,
     "32768": 4.1234
   }
   ```

2. **Generate plots:**
   ```bash
   bash scripts/run_plot_latency.sh
   ```
   
   Or directly:
   ```bash
   python3 src/evaluation/plot_latency.py \
       --baseline results/latency_baseline.json \
       --sdtp results/latency_sdtp.json \
       --out_dir results/fig
   ```

3. **Output files:**
   - `results/fig/latency_curve.png`: Prefill latency comparison
   - `results/fig/speedup_curve.png`: Speedup vs sequence length
   - `results/fig/flops_curve.png`: Estimated FLOPs reduction

### Parsing Log Files

If you have log files from inference runs, parse them first:

```bash
python3 src/evaluation/parse_latency_log.py \
    --log logs/inference.log \
    --baseline results/latency_baseline.json \
    --sdtp results/latency_sdtp.json
```

Expected log format:
```
[Length 4096] baseline=0.7065s  sdtp=0.2527s  speedup=2.80x
[Length 8192] baseline=1.2684s  sdtp=0.4920s  speedup=2.58x
```

### Phase 2 Comprehensive Visualization

Generate publication-quality figures for Phase 2 evaluation results:

```bash
python3 src/evaluation/plot_phase2_results.py \
    --results_dir results \
    --out_dir results/fig \
    --ref_length 16384
```

**Required JSON files:**
- `results/latency_results_keep09.json`
- `results/latency_results_keep08.json`
- `results/latency_results_keep07.json`
- `results/hotpotqa_baseline.json`
- `results/hotpotqa_sdtp_0.9.json`
- `results/hotpotqa_sdtp_0.8.json`
- `results/hotpotqa_sdtp_0.7.json`

**Generated figures:**
- `phase2_end2end_latency_comparison.png`: 3×2 subplot showing prefill, decode, total latency and speedups
- `phase2_kv_cache_reduction.png`: KV cache compression effect with dual Y-axis
- `phase2_performance_speed_tradeoff.png`: Hit Rate vs Speedup scatter plot
- `phase2_longbench_performance.png`: LongBench performance retention bar chart
- `phase2_speedup_heatmap.png`: Comprehensive speedup heatmap across configurations

