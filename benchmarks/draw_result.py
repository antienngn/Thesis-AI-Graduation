import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import argparse

def load_json_results(results_dir="RESULTS"):
    """Load all JSON files from the results directory"""
    results = []
    pattern = os.path.join(results_dir, "*.json")
    for file_path in glob.glob(pattern):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                data['filename'] = os.path.basename(file_path)
                results.append(data)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    return results

def extract_metrics(results):
    """Extract key metrics for visualization"""
    metrics = {
        'schedule_types': [],
        'request_rates': [],
        'output_throughput': [],
        'request_throughput': [],
        'mean_ttft_ms': [],
        'mean_tpot_ms': [],
    }

    for result in results:
        metrics['schedule_types'].append(result.get('schedule_type', 'unknown'))
        metrics['request_rates'].append(result.get('request_rate', 0))
        metrics['output_throughput'].append(result.get('output_throughput', 0))
        metrics['request_throughput'].append(result.get('request_throughput', 0))
        metrics['mean_ttft_ms'].append(result.get('mean_ttft_ms', 0))
        metrics['mean_tpot_ms'].append(result.get('mean_tpot_ms', 0))

    return metrics

def _plot_metric(metrics, metric_key, ylabel, title, filename, output_dir):
    """Generic helper: plot a single metric vs request rate per schedule type."""
    os.makedirs(output_dir, exist_ok=True)
    schedule_types = list(set(metrics['schedule_types']))

    plt.figure(figsize=(12, 8))

    for schedule_type in schedule_types:
        values = []
        rates = []

        for i, st in enumerate(metrics['schedule_types']):
            if st == schedule_type:
                values.append(metrics[metric_key][i])
                rates.append(metrics['request_rates'][i])

        if values:
            sorted_data = sorted(zip(rates, values))
            rates, values = zip(*sorted_data)
            plt.plot(rates, values, marker='o', label=schedule_type, linewidth=2)

    plt.xlabel('Request Rate (queries/second)')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

def plot_output_throughput(metrics, output_dir="plots"):
    _plot_metric(metrics, 'output_throughput',
                 'Output Token Throughput (tokens/second)',
                 'Output Token Throughput vs Request Rate',
                 'output_throughput.png', output_dir)

def plot_request_throughput(metrics, output_dir="plots"):
    _plot_metric(metrics, 'request_throughput',
                 'Request Throughput (requests/second)',
                 'Request Throughput vs Request Rate',
                 'request_throughput.png', output_dir)

def plot_mean_ttft(metrics, output_dir="plots"):
    _plot_metric(metrics, 'mean_ttft_ms',
                 'Mean Time to First Token (ms)',
                 'Mean TTFT vs Request Rate',
                 'mean_ttft.png', output_dir)

def plot_mean_tpot(metrics, output_dir="plots"):
    _plot_metric(metrics, 'mean_tpot_ms',
                 'Mean Time Per Output Token (ms)',
                 'Mean TPOT vs Request Rate',
                 'mean_tpot.png', output_dir)

def create_summary_table(metrics, output_dir="plots"):
    """Create a summary table of key metrics"""
    os.makedirs(output_dir, exist_ok=True)

    schedule_types = list(set(metrics['schedule_types']))
    summary_data = []

    for schedule_type in schedule_types:
        max_rate = max([r for i, r in enumerate(metrics['request_rates'])
                       if metrics['schedule_types'][i] == schedule_type])

        for i, (st, rate) in enumerate(zip(metrics['schedule_types'], metrics['request_rates'])):
            if st == schedule_type and rate == max_rate:
                summary_data.append({
                    'Schedule Type': schedule_type,
                    'Request Rate': rate,
                    'Output Throughput': metrics['output_throughput'][i],
                    'Request Throughput': metrics['request_throughput'][i],
                    'Mean TTFT (ms)': metrics['mean_ttft_ms'][i],
                    'Mean TPOT (ms)': metrics['mean_tpot_ms'][i],
                })
                break

    import pandas as pd
    df = pd.DataFrame(summary_data)
    df.to_csv(os.path.join(output_dir, 'summary_table.csv'), index=False)

    print("\n=== Summary Table ===")
    print(df.to_string(index=False, float_format='%.2f'))

def main():
    parser = argparse.ArgumentParser(description='Visualize benchmark results')
    parser.add_argument('--results-dir', default='RESULTS', help='Directory containing JSON result files')
    parser.add_argument('--output-dir', default='plots', help='Directory to save plots')

    args = parser.parse_args()

    print(f"Loading results from {args.results_dir}...")
    results = load_json_results(args.results_dir)

    if not results:
        print(f"No JSON files found in {args.results_dir}")
        return

    print(f"Loaded {len(results)} result files")

    metrics = extract_metrics(results)

    print("Creating output throughput plot...")
    plot_output_throughput(metrics, args.output_dir)

    print("Creating request throughput plot...")
    plot_request_throughput(metrics, args.output_dir)

    print("Creating mean TTFT plot...")
    plot_mean_ttft(metrics, args.output_dir)

    print("Creating mean TPOT plot...")
    plot_mean_tpot(metrics, args.output_dir)

    print("Creating summary table...")
    create_summary_table(metrics, args.output_dir)

    print(f"\nPlots saved to {args.output_dir}/")
    print("Generated files:")
    print("- output_throughput.png")
    print("- request_throughput.png")
    print("- mean_ttft.png")
    print("- mean_tpot.png")
    print("- summary_table.csv")

if __name__ == "__main__":
    main()
