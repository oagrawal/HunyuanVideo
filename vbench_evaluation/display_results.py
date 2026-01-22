#!/usr/bin/env python3
"""
Display VBench evaluation results from JSON files
"""
import os
import json
import pandas as pd

def display_results():
    results_dir = './vbench_results'
    
    # Find all eval_results.json files
    result_files = {
        'my_policy_0.30': 'my_policy_0.30_eval_results.json',
        'teacache_0.30': 'teacache_0.30_eval_results.json',
        'teacache_0.20': 'teacache_0.20_eval_results.json',
        'teacache_0.10': 'teacache_0.10_eval_results.json',
    }
    
    # Load all results and extract scores
    all_results = {}
    for name, filename in result_files.items():
        filepath = os.path.join(results_dir, filename)
        if os.path.exists(filepath):
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
        else:
            print(f"✗ Results not found: {filepath}")
    
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
    
    # Print summary table
    print("\n" + "="*90)
    print("SUMMARY OF RESULTS")
    print("="*90 + "\n")
    
    # Print in a more readable format
    metrics = [col for col in df.columns if col != 'Video']
    print(f"{'Metric':<25} {'my_policy':>14} {'teacache_0.30':>14} {'teacache_0.20':>14} {'teacache_0.10':>14}")
    print("-" * 90)
    
    for metric in metrics:
        values = []
        for video in ['my_policy_0.30', 'teacache_0.30', 'teacache_0.20', 'teacache_0.10']:
            video_row = df[df['Video'] == video]
            if not video_row.empty and metric in video_row.columns:
                score = video_row[metric].iloc[0]
                if pd.notna(score):
                    values.append(f"{score:>14.4f}")
                else:
                    values.append(f"{'N/A':>14}")
            else:
                values.append(f"{'N/A':>14}")
        print(f"{metric:<25} {values[0]} {values[1]} {values[2]} {values[3]}")
    
    # Calculate and display average scores
    print("\n" + "="*90)
    print("AVERAGE SCORES (across all dimensions)")
    print("="*90 + "\n")
    
    avg_scores = {}
    for video in ['my_policy_0.30', 'teacache_0.30', 'teacache_0.20', 'teacache_0.10']:
        video_data = df[df['Video'] == video][metrics]
        if not video_data.empty:
            avg_score = video_data.mean(axis=1).iloc[0]
            avg_scores[video] = avg_score
            print(f"{video:<25}: {avg_score:.4f}")
    
    # Ranking
    if avg_scores:
        print("\n" + "="*90)
        print("RANKING (best to worst)")
        print("="*90 + "\n")
        
        sorted_videos = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
        for rank, (video, score) in enumerate(sorted_videos, 1):
            print(f"{rank}. {video:<25} (score: {score:.4f})")
    
    print(f"\n{'='*90}\n")
    
    # Print key insights
    print("="*90)
    print("KEY INSIGHTS")
    print("="*90 + "\n")
    
    # Find best and worst per metric
    for metric in metrics:
        metric_scores = {}
        for video in ['my_policy_0.30', 'teacache_0.30', 'teacache_0.20', 'teacache_0.10']:
            video_row = df[df['Video'] == video]
            if not video_row.empty and metric in video_row.columns:
                score = video_row[metric].iloc[0]
                if pd.notna(score):
                    metric_scores[video] = score
        
        if metric_scores:
            best = max(metric_scores.items(), key=lambda x: x[1])
            worst = min(metric_scores.items(), key=lambda x: x[1])
            print(f"{metric:<25} → Best: {best[0]} ({best[1]:.4f}), Worst: {worst[0]} ({worst[1]:.4f})")
    
    print(f"\n{'='*90}\n")

if __name__ == '__main__':
    print("="*90)
    print("VBench Results Display")
    print("="*90 + "\n")
    display_results()
