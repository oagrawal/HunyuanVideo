#!/usr/bin/env python3
"""
Display VBench evaluation results from JSON files
"""
import os
import json
import pandas as pd
import glob
import argparse

def display_results(folder=None):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Determine results directory
    if folder:
        results_dir = os.path.join(script_dir, 'vbench_results', folder)
    else:
        results_dir = os.path.join(script_dir, 'vbench_results')
    
    print(f"Results directory: {results_dir}")
    
    # Dynamically find all eval_results.json files
    result_pattern = os.path.join(results_dir, '*_eval_results.json')
    result_files = glob.glob(result_pattern)
    
    if not result_files:
        # Try looking in subfolders if no results found directly
        result_pattern_subfolder = os.path.join(results_dir, '*', '*_eval_results.json')
        result_files = glob.glob(result_pattern_subfolder)
        if result_files:
            print(f"\nFound results in subfolders. Available folders:")
            subfolders = set(os.path.basename(os.path.dirname(f)) for f in result_files)
            for sf in sorted(subfolders):
                print(f"  - {sf}")
            print(f"\nUse --folder <name> to display specific results.")
            return
        print(f"✗ No result files found matching: {result_pattern}")
        return
    
    # Load all results and extract scores
    all_results = {}
    for filepath in sorted(result_files):
        filename = os.path.basename(filepath)
        # Extract name from filename (e.g., "adaptive_0.15_0.3_eval_results.json" -> "adaptive_0.15_0.3")
        name = filename.replace('_eval_results.json', '')
        
        with open(filepath, 'r') as f:
            raw_results = json.load(f)
        
        # Extract just the scores (first element of each array)
        scores = {}
        for metric, value in raw_results.items():
            if isinstance(value, list) and len(value) > 0:
                scores[metric] = value[0]  # First element is the score
            else:
                scores[metric] = value
        
        all_results[name] = scores
        print(f"✓ Loaded results for {name}")
    
    if not all_results:
        print("\nNo results found!")
        return
    
    # Create comparison dataframe
    comparison_data = []
    for video_name, results in all_results.items():
        row = {'Video': video_name}
        row.update(results)
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    
    # Save as CSV
    csv_path = os.path.join(results_dir, 'comparison_results_clean.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Comparison table saved to: {csv_path}")
    
    # Save as JSON
    json_path = os.path.join(results_dir, 'comparison_results_clean.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"✓ Detailed results saved to: {json_path}")
    
    # Get list of video names
    video_names = sorted(all_results.keys())
    metrics = [col for col in df.columns if col != 'Video']
    
    # Print summary table
    print("\n" + "="*120)
    print("SUMMARY OF RESULTS")
    print("="*120 + "\n")
    
    # Print header
    header = f"{'Metric':<25}"
    for name in video_names:
        header += f" {name:>18}"
    print(header)
    print("-" * (25 + 19 * len(video_names)))
    
    # Print each metric
    for metric in metrics:
        row = f"{metric:<25}"
        for video in video_names:
            video_row = df[df['Video'] == video]
            if not video_row.empty and metric in video_row.columns:
                score = video_row[metric].iloc[0]
                if pd.notna(score):
                    row += f" {score:>18.4f}"
                else:
                    row += f" {'N/A':>18}"
            else:
                row += f" {'N/A':>18}"
        print(row)
    
    # Calculate and display average scores
    print("\n" + "="*120)
    print("AVERAGE SCORES (across all dimensions)")
    print("="*120 + "\n")
    
    avg_scores = {}
    for video in video_names:
        video_data = df[df['Video'] == video][metrics]
        if not video_data.empty:
            avg_score = video_data.mean(axis=1).iloc[0]
            avg_scores[video] = avg_score
            print(f"{video:<25}: {avg_score:.4f}")
    
    # Ranking
    if avg_scores:
        print("\n" + "="*120)
        print("RANKING (best to worst)")
        print("="*120 + "\n")
        
        sorted_videos = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
        for rank, (video, score) in enumerate(sorted_videos, 1):
            print(f"{rank}. {video:<25} (score: {score:.4f})")
    
    print(f"\n{'='*120}\n")
    
    # Print key insights
    print("="*120)
    print("KEY INSIGHTS")
    print("="*120 + "\n")
    
    # Find best and worst per metric
    for metric in metrics:
        metric_scores = {}
        for video in video_names:
            video_row = df[df['Video'] == video]
            if not video_row.empty and metric in video_row.columns:
                score = video_row[metric].iloc[0]
                if pd.notna(score):
                    metric_scores[video] = score
        
        if metric_scores:
            best = max(metric_scores.items(), key=lambda x: x[1])
            worst = min(metric_scores.items(), key=lambda x: x[1])
            print(f"{metric:<25} → Best: {best[0]} ({best[1]:.4f}), Worst: {worst[0]} ({worst[1]:.4f})")
    
    print(f"\n{'='*120}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Display VBench evaluation results')
    parser.add_argument('--folder', '-f', type=str, default=None,
                        help='Subfolder name to display results from (e.g., "lone_surfer", "anthropomorphic_cats")')
    args = parser.parse_args()
    
    print("="*120)
    print("VBench Results Display")
    print("="*120 + "\n")
    display_results(folder=args.folder)
