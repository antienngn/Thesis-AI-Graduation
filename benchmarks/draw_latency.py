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
  
def extract_latency_metrics(results):  
    """Extract latency per token metrics for visualization"""  
    metrics = {  
        'schedule_types': [],  
        'request_rates': [],  
        'mean_tpot_s': [],      # Mean TPOT in seconds  
        'median_tpot_s': [],    # Median TPOT in seconds  
        'p99_tpot_s': [],       # P99 TPOT in seconds  
        'models': [],  
        'cv_values': []         # Coefficient of variation  
    }  
      
    for result in results:  
        # Convert TPOT from milliseconds to seconds  
        metrics['schedule_types'].append(result.get('schedule_type', 'unknown'))  
        metrics['request_rates'].append(result.get('request_rate', 0))  
        metrics['mean_tpot_s'].append(result.get('mean_tpot_ms', 0) / 1000.0)  
        metrics['median_tpot_s'].append(result.get('median_tpot_ms', 0) / 1000.0)  
        metrics['p99_tpot_s'].append(result.get('p99_tpot_ms', 0) / 1000.0)  
        metrics['models'].append(result.get('model_id', 'unknown').split('/')[-1])  
          
        # Extract CV value from filename or use default  
        filename = result.get('filename', '')  
        if 'cv' in filename:  
            try:  
                cv_part = filename.split('cv')[1].split('-')[0]  
                metrics['cv_values'].append(float(cv_part))  
            except:  
                metrics['cv_values'].append(1.0)  
        else:  
            metrics['cv_values'].append(1.0)  
      
    return metrics  
  
def plot_latency_per_token(metrics, output_dir="plots"):  
    """Plot latency per token (s/token) across schedule types and request rates"""  
    os.makedirs(output_dir, exist_ok=True)  
      
    schedule_types = list(set(metrics['schedule_types']))  
      
    # Create subplots for different latency percentiles  
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))  
    fig.suptitle('Latency per Token (s/token) vs Request Rate', fontsize=16)  
      
    # Helper function to plot data on a specific axis  
    def plot_on_axis(ax, latency_data, title, ylabel):  
        for schedule_type in schedule_types:  
            latencies = []  
            rates = []  
              
            for i, st in enumerate(metrics['schedule_types']):  
                if st == schedule_type:  
                    latencies.append(latency_data[i])  
                    rates.append(metrics['request_rates'][i])  
              
            if latencies:  
                # Sort by request rate  
                sorted_data = sorted(zip(rates, latencies))  
                rates, latencies = zip(*sorted_data)  
                ax.plot(rates, latencies, marker='o', label=schedule_type, linewidth=2)  
          
        ax.set_xlabel('Request Rate (queries/second)')  
        ax.set_ylabel(ylabel)  
        ax.set_title(title)  
        ax.legend()  
        ax.grid(True, alpha=0.3)  
        ax.set_xscale('log')  
        ax.set_yscale('log')  
      
    # Plot mean TPOT  
    plot_on_axis(ax1, metrics['mean_tpot_s'], 'Mean Latency per Token', 'Latency (s/token)')  
      
    # Plot median TPOT  
    plot_on_axis(ax2, metrics['median_tpot_s'], 'Median Latency per Token', 'Latency (s/token)')  
      
    # Plot P99 TPOT  
    plot_on_axis(ax3, metrics['p99_tpot_s'], 'P99 Latency per Token', 'Latency (s/token)')  
      
    # Plot latency comparison (mean vs median vs p99 for a specific schedule type)  
    if 'fcfs' in schedule_types:  
        fcfs_mean = []  
        fcfs_median = []  
        fcfs_p99 = []  
        rates = []  
          
        for i, st in enumerate(metrics['schedule_types']):  
            if st == 'fcfs':  
                fcfs_mean.append(metrics['mean_tpot_s'][i])  
                fcfs_median.append(metrics['median_tpot_s'][i])  
                fcfs_p99.append(metrics['p99_tpot_s'][i])  
                rates.append(metrics['request_rates'][i])  
          
        if fcfs_mean:  
            sorted_data = sorted(zip(rates, fcfs_mean, fcfs_median, fcfs_p99))  
            rates, fcfs_mean, fcfs_median, fcfs_p99 = zip(*sorted_data)  
              
            ax4.plot(rates, fcfs_mean, marker='o', label='Mean', linewidth=2)  
            ax4.plot(rates, fcfs_median, marker='s', label='Median', linewidth=2, linestyle='--')  
            ax4.plot(rates, fcfs_p99, marker='^', label='P99', linewidth=2, linestyle=':')  
            ax4.set_xlabel('Request Rate (queries/second)')  
            ax4.set_ylabel('Latency (s/token)')  
            ax4.set_title('FCFS: Latency Percentiles Comparison')  
            ax4.legend()  
            ax4.grid(True, alpha=0.3)  
            ax4.set_xscale('log')  
            ax4.set_yscale('log')  
      
    plt.tight_layout()  
    plt.savefig(os.path.join(output_dir, 'latency_per_token.png'), dpi=300, bbox_inches='tight')  
    plt.close()  
  
def create_latency_summary_table(metrics, output_dir="plots"):  
    """Create a summary table of latency per token metrics"""  
    os.makedirs(output_dir, exist_ok=True)  
      
    schedule_types = list(set(metrics['schedule_types']))  
      
    # Create summary for each schedule type at different request rates  
    summary_data = []  
      
    for schedule_type in schedule_types:  
        for rate in sorted(set([r for i, r in enumerate(metrics['request_rates'])   
                              if metrics['schedule_types'][i] == schedule_type])):  
            for i, (st, r) in enumerate(zip(metrics['schedule_types'], metrics['request_rates'])):  
                if st == schedule_type and r == rate:  
                    summary_data.append({  
                        'Schedule Type': schedule_type,  
                        'Request Rate': rate,  
                        'Mean Latency (s/token)': metrics['mean_tpot_s'][i],  
                        'Median Latency (s/token)': metrics['median_tpot_s'][i],  
                        'P99 Latency (s/token)': metrics['p99_tpot_s'][i],  
                        'CV': metrics['cv_values'][i]  
                    })  
                    break  
      
    # Save as CSV  
    import pandas as pd  
    df = pd.DataFrame(summary_data)  
    df.to_csv(os.path.join(output_dir, 'latency_summary.csv'), index=False)  
      
    # Print summary for lowest and highest request rates  
    print("\n=== Latency per Token Summary ===")  
    if not df.empty:  
        print("Lowest request rate performance:")  
        lowest_rate = df['Request Rate'].min()  
        print(df[df['Request Rate'] == lowest_rate].to_string(index=False, float_format='%.4f'))  
          
        print("\nHighest request rate performance:")  
        highest_rate = df['Request Rate'].max()  
        print(df[df['Request Rate'] == highest_rate].to_string(index=False, float_format='%.4f'))  
  
def plot_latency_distribution(metrics, output_dir="plots"):  
    """Plot distribution of latency per token for different schedule types"""  
    os.makedirs(output_dir, exist_ok=True)  
      
    plt.figure(figsize=(12, 8))  
      
    schedule_types = list(set(metrics['schedule_types']))  
      
    # Create box plot data  
    data_to_plot = []  
    labels = []  
      
    for schedule_type in schedule_types:  
        # Get all mean TPOT values for this schedule type  
        latencies = [metrics['mean_tpot_s'][i] for i, st in enumerate(metrics['schedule_types'])   
                    if st == schedule_type]  
        if latencies:  
            data_to_plot.append(latencies)  
            labels.append(schedule_type)  
      
    if data_to_plot:  
        plt.boxplot(data_to_plot, labels=labels)  
        plt.ylabel('Latency per Token (s/token)')  
        plt.title('Latency per Token Distribution by Schedule Type')  
        plt.xticks(rotation=45)  
        plt.grid(True, alpha=0.3)  
        plt.yscale('log')  
        plt.tight_layout()  
        plt.savefig(os.path.join(output_dir, 'latency_distribution.png'), dpi=300, bbox_inches='tight')  
        plt.close()  
  
def main():  
    parser = argparse.ArgumentParser(description='Visualize latency per token from benchmark results')  
    parser.add_argument('--results-dir', default='RESULTS', help='Directory containing JSON result files')  
    parser.add_argument('--output-dir', default='plots', help='Directory to save plots')  
      
    args = parser.parse_args()  
      
    # Load results  
    print(f"Loading results from {args.results_dir}...")  
    results = load_json_results(args.results_dir)  
      
    if not results:  
        print(f"No JSON files found in {args.results_dir}")  
        return  
      
    print(f"Loaded {len(results)} result files")  
      
    # Extract latency metrics  
    metrics = extract_latency_metrics(results)  
      
    # Create visualizations  
    print("Creating latency per token plots...")  
    plot_latency_per_token(metrics, args.output_dir)  
      
    print("Creating latency distribution plot...")  
    plot_latency_distribution(metrics, args.output_dir)  
      
    print("Creating latency summary table...")  
    create_latency_summary_table(metrics, args.output_dir)  
      
    print(f"\nPlots saved to {args.output_dir}/")  
    print("Generated files:")  
    print("- latency_per_token.png (comprehensive latency analysis)")  
    print("- latency_distribution.png (distribution by schedule type)")  
    print("- latency_summary.csv (detailed metrics)")  
  
if __name__ == "__main__":  
    main()