#!/usr/bin/env python3
"""
VBench evaluation script for comparing HunyuanVideo caching strategies
Run from: /workspace/vbench_evaluation/
"""
import os
import sys
import json
from pathlib import Path
import pandas as pd
import shutil
from datetime import datetime

def evaluate_all_videos():
    """Evaluate all videos using VBench on all dimensions."""
    
    # Configuration - using current directory
    current_dir = os.getcwd()
    videos_dir = current_dir
    output_dir = os.path.join(current_dir, 'vbench_results')
    
    # VBench paths
    vbench_root = '/workspace/VBench'
    vbench_full_info_path = os.path.join(vbench_root, 'vbench/VBench_full_info.json')
    
    print(f"Working directory: {current_dir}")
    print(f"Output directory: {output_dir}")
    print(f"VBench root: {vbench_root}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Verify VBench_full_info.json exists
    if not os.path.exists(vbench_full_info_path):
        print(f"\n✗ Error: VBench_full_info.json not found at: {vbench_full_info_path}")
        print("Please check your VBench installation.")
        return
    else:
        print(f"✓ Found VBench_full_info.json")
    
    # Read prompt from file
    prompt_file = os.path.join(current_dir, 'prompt_list.txt')
    with open(prompt_file, 'r') as f:
        prompt = f.read().strip()
    
    print(f"Using prompt: {prompt}")
    
    # Define video files and their labels
    video_files = {
        'my_policy_0.30': 'my_policy(weak_caching_0.30).mp4',
        'teacache_0.30': 'teacache_0.30.mp4',
        'teacache_0.20': 'teacache_0.20.mp4',
        'teacache_0.10': 'teacache_0.10.mp4',
    }
    
    # Verify all videos exist
    print("\nVerifying video files...")
    for name, video_file in video_files.items():
        video_path = os.path.join(videos_dir, video_file)
        if os.path.exists(video_path):
            size_mb = os.path.getsize(video_path) / (1024 * 1024)
            print(f"  ✓ {video_file} ({size_mb:.2f} MB)")
        else:
            print(f"  ✗ {video_file} NOT FOUND")
            return
    
    # Import VBench
    print("\nImporting VBench...")
    try:
        from vbench import VBench
        print("  ✓ VBench imported successfully")
    except ImportError as e:
        print(f"  ✗ Error importing VBench: {e}")
        print("\nPlease install VBench first:")
        print("  cd /workspace")
        print("  git clone https://github.com/Vchitect/VBench.git")
        print("  cd VBench")
        print("  pip install -e .")
        return
    
    # Initialize VBench
    print("\nInitializing VBench...")
    try:
        import torch
        
        # Check GPU availability
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"  ✓ CUDA is available")
            print(f"  GPU: {gpu_name}")
            print(f"  GPU Memory: {gpu_memory:.2f} GB")
            device = torch.device('cuda:0')
            print(f"  Using device: GPU (cuda:0)")
        else:
            print(f"  ⚠ CUDA not available - falling back to CPU")
            print(f"  (This will be significantly slower)")
            device = torch.device('cpu')
            print(f"  Using device: CPU")
        
        vbench = VBench(
            device=device,
            full_info_dir=vbench_full_info_path,
            output_path=output_dir
        )
        print("  ✓ VBench initialized successfully")
    except Exception as e:
        print(f"  ✗ Failed to initialize VBench: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Dimensions supported for custom input videos
    # Note: Some dimensions like 'object_class', 'multiple_objects', 'color', 
    # 'spatial_relationship', 'scene', and 'appearance_style' require VBench's 
    # standard dataset with additional annotations and are not supported for custom videos
    all_dimensions = [
        'subject_consistency',
        'background_consistency',
        'temporal_flickering',
        'motion_smoothness',
        'dynamic_degree',
        'aesthetic_quality',
        'imaging_quality',
        'temporal_style',
        'overall_consistency'
    ]
    
    print(f"\nWill evaluate {len(all_dimensions)} dimensions supported for custom videos:")
    for dim in all_dimensions:
        print(f"  - {dim}")
    
    print(f"\nNote: 6 dimensions are excluded (not supported for custom videos):")
    print(f"  - object_class, multiple_objects, color, spatial_relationship, scene, appearance_style")
    print(f"  These require VBench's standard dataset with additional annotations.")
    
    # Store all results
    all_results = {}
    start_time = datetime.now()
    
    # Evaluate each video
    for idx, (name, video_file) in enumerate(video_files.items(), 1):
        video_path = os.path.join(videos_dir, video_file)
        
        print(f"\n{'='*70}")
        print(f"[{idx}/{len(video_files)}] Evaluating: {name}")
        print(f"Video: {video_file}")
        print(f"{'='*70}")
        
        # Create temporary directory for this video
        temp_video_dir = os.path.join(output_dir, f'temp_{name}')
        os.makedirs(temp_video_dir, exist_ok=True)
        
        # VBench expects videos named as numbers (0.mp4, 1.mp4, etc.)
        temp_video_path = os.path.join(temp_video_dir, '0.mp4')
        
        try:
            # Copy video to temp directory
            print(f"  Preparing video...")
            shutil.copy(video_path, temp_video_path)
            
            # Create prompt file
            temp_prompt_file = os.path.join(temp_video_dir, 'prompt.txt')
            with open(temp_prompt_file, 'w') as f:
                f.write(prompt)
            
            print(f"  Running VBench evaluation on {len(all_dimensions)} dimensions...")
            print(f"  (This may take several minutes per video)")
            
            # Run evaluation on all dimensions
            # Use mode='custom_input' for custom videos with prompts
            # prompt_list must be a dict mapping video filenames to prompts
            prompt_dict = {"0.mp4": prompt}
            
            results = vbench.evaluate(
                videos_path=temp_video_dir,
                name=name,
                prompt_list=prompt_dict,
                dimension_list=all_dimensions,
                mode='custom_input'
            )
            
            # VBench saves results to JSON file, read it back
            results_file = os.path.join(output_dir, f'{name}_eval_results.json')
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    results = json.load(f)
                all_results[name] = results
                print(f"\n  ✓ Completed evaluation for {name}")
                
                # Print quick preview of results
                print(f"\n  Quick preview of scores:")
                for metric, score in list(results.items())[:5]:
                    print(f"    {metric}: {score:.4f}")
                if len(results) > 5:
                    print(f"    ... and {len(results)-5} more metrics")
            else:
                print(f"\n  ⚠ Warning: Results file not found: {results_file}")
                all_results[name] = {}
            
        except Exception as e:
            print(f"\n  ✗ Error evaluating {name}: {str(e)}")
            import traceback
            traceback.print_exc()
            all_results[name] = {}
        
        finally:
            # Clean up temp directory
            if os.path.exists(temp_video_dir):
                shutil.rmtree(temp_video_dir)
    
    # Calculate total time
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n{'='*70}")
    print(f"Total evaluation time: {duration}")
    print(f"{'='*70}")
    
    # Create comparison table
    print(f"\n{'='*70}")
    print("Creating comparison table...")
    print(f"{'='*70}\n")
    
    if not any(all_results.values()):
        print("No results to compare. Please check errors above.")
        return
    
    comparison_data = []
    for video_name, results in all_results.items():
        row = {'Video': video_name}
        if results:
            row.update(results)
        comparison_data.append(row)
    
    # Save results as CSV
    df = pd.DataFrame(comparison_data)
    csv_path = os.path.join(output_dir, 'comparison_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"✓ Comparison table saved to: {csv_path}")
    
    # Save as JSON for detailed analysis
    json_path = os.path.join(output_dir, 'comparison_results.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"✓ Detailed results saved to: {json_path}")
    
    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY OF RESULTS")
    print("="*70 + "\n")
    
    # Print in a more readable format
    print(f"{'Metric':<25} {'my_policy':>12} {'teacache_0.30':>14} {'teacache_0.20':>14} {'teacache_0.10':>14}")
    print("-" * 85)
    
    metrics = [col for col in df.columns if col != 'Video']
    for metric in metrics:
        values = []
        for video in ['my_policy_0.30', 'teacache_0.30', 'teacache_0.20', 'teacache_0.10']:
            val = df[df['Video'] == video][metric].values
            if len(val) > 0 and pd.notna(val[0]):
                values.append(f"{val[0]:>12.4f}")
            else:
                values.append(f"{'N/A':>12}")
        print(f"{metric:<25} {values[0]} {values[1]} {values[2]} {values[3]}")
    
    # Calculate and display average scores
    print("\n" + "="*70)
    print("AVERAGE SCORES (across all dimensions)")
    print("="*70 + "\n")
    
    avg_scores = {}
    for video in ['my_policy_0.30', 'teacache_0.30', 'teacache_0.20', 'teacache_0.10']:
        video_data = df[df['Video'] == video][metrics]
        if not video_data.empty:
            avg_score = video_data.mean(axis=1).values[0]
            avg_scores[video] = avg_score
            print(f"{video:<20}: {avg_score:.4f}")
    
    # Ranking
    if avg_scores:
        print("\n" + "="*70)
        print("RANKING (best to worst)")
        print("="*70 + "\n")
        
        sorted_videos = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
        for rank, (video, score) in enumerate(sorted_videos, 1):
            print(f"{rank}. {video:<20} (score: {score:.4f})")
    
    print(f"\n{'='*70}")
    print(f"✓ Evaluation complete!")
    print(f"✓ All results saved in: {output_dir}")
    print(f"{'='*70}\n")
    
    return all_results

if __name__ == '__main__':
    print("="*70)
    print("VBench Video Quality Evaluation")
    print("="*70)
    evaluate_all_videos()