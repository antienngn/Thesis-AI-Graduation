import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import argparse


def load_json_results(results_dir="RESULTS"):
    """Load all JSON files from the results directory."""
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


def compute_nlatency(result):
    """
    Tính Nlatency (normalized latency) đúng theo bài báo.

    Công thức:
        latency_i  = ttfts[i] + sum(itls[i])
        Nlatency_i = latency_i / output_lens[i]
        mean_Nlatency = mean(Nlatency_i)   over all requests
        p90_Nlatency  = percentile(Nlatency_i, 90)

    Ý nghĩa: chi phí trung bình (giây) để sinh một token,
    tính trên toàn bộ vòng đời request bao gồm cả thời gian
    chờ trong queue — đây là metric bài báo dùng để so sánh
    các scheduler trong Figure 3 và Table 1.
    """
    ttfts       = result.get('ttfts', [])
    itls        = result.get('itls', [])
    output_lens = result.get('output_lens', [])

    if not ttfts or not itls or not output_lens:
        return None, None

    nlatencies = []
    for i in range(len(ttfts)):
        if output_lens[i] > 0:
            latency_i  = ttfts[i] + sum(itls[i])   # e2e latency (s)
            nlatency_i = latency_i / output_lens[i]  # s/token
            nlatencies.append(nlatency_i)

    if not nlatencies:
        return None, None

    nlatencies = np.array(nlatencies)
    return float(np.mean(nlatencies)), float(np.percentile(nlatencies, 90))


def extract_metrics(results):
    """
    Extract Nlatency metrics (mean và P90) tính từ ttfts, itls, output_lens.
    Đơn vị: giây/token (s/token).
    """
    metrics = {
        'schedule_types': [],
        'request_rates':  [],
        'mean_nlatency_s': [],  # mean(latency_i / output_len_i)  in seconds
        'p90_nlatency_s':  [],  # p90(latency_i / output_len_i)   in seconds
    }

    for result in results:
        mean_nl, p90_nl = compute_nlatency(result)

        if mean_nl is None:
            print(f"  [WARN] Skipping {result['filename']}: missing ttfts/itls/output_lens")
            continue

        metrics['schedule_types'].append(result.get('schedule_type', 'unknown'))
        metrics['request_rates'].append(result.get('request_rate', 0))
        metrics['mean_nlatency_s'].append(mean_nl)
        metrics['p90_nlatency_s'].append(p90_nl)

        print(f"  {result['filename']}")
        print(f"    schedule_type    = {result.get('schedule_type', 'unknown')}")
        print(f"    request_rate     = {result.get('request_rate', 0)} req/s")
        print(f"    mean Nlatency    = {mean_nl*1000:.2f} ms/token")
        print(f"    P90  Nlatency    = {p90_nl*1000:.2f} ms/token")

    return metrics


def _collect_series(metrics, value_key):
    """
    Group (request_rate, value) pairs by schedule_type, sorted by rate.
    Returns dict: schedule_type -> (sorted_rates, sorted_values)
    """
    series = {}
    for i, stype in enumerate(metrics['schedule_types']):
        rate = metrics['request_rates'][i]
        val  = metrics[value_key][i]
        series.setdefault(stype, []).append((rate, val))

    for stype in series:
        series[stype].sort(key=lambda x: x[0])

    return series


def plot_metric(metrics, value_key, title, ylabel, filename, output_dir):
    """
    Draw a single plot: one line per schedule_type, x=request_rate, y=metric.
    """
    os.makedirs(output_dir, exist_ok=True)
    series = _collect_series(metrics, value_key)

    fig, ax = plt.subplots(figsize=(9, 6))

    for stype, points in sorted(series.items()):
        rates, vals = zip(*points)
        ax.plot(rates, vals, marker='o', linewidth=2, label=stype)

    ax.set_xlabel('Request Rate (queries/second)', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(title='Schedule Type', fontsize=10)
    ax.grid(True, alpha=0.3)

    out_path = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot mean and P90 Nlatency (latency/output_len) from benchmark results'
    )
    parser.add_argument('--results-dir', default='RESULTS',
                        help='Directory containing JSON result files')
    parser.add_argument('--output-dir', default='plots',
                        help='Directory to save output plots')
    args = parser.parse_args()

    print(f"Loading results from {args.results_dir}...")
    results = load_json_results(args.results_dir)

    if not results:
        print(f"No JSON files found in {args.results_dir}")
        return

    print(f"Loaded {len(results)} result file(s)\n")
    metrics = extract_metrics(results)

    if not metrics['schedule_types']:
        print("No valid results to plot.")
        return

    # Plot 1: Mean Nlatency — metric chính bài báo dùng (Figure 3, Table 1)
    plot_metric(
        metrics,
        value_key='mean_nlatency_s',
        title='Mean Normalized Latency vs Request Rate',
        ylabel='Mean Nlatency (s/token)',
        filename='mean_nlatency.png',
        output_dir=args.output_dir,
    )

    # Plot 2: P90 Nlatency — tương ứng P90 latency bài báo dùng (Table 1)
    plot_metric(
        metrics,
        value_key='p90_nlatency_s',
        title='P90 Normalized Latency vs Request Rate',
        ylabel='P90 Nlatency (s/token)',
        filename='p90_nlatency.png',
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()