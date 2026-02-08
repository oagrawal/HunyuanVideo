#!/usr/bin/env python3
"""
VBench evaluation script for comparing HunyuanVideo caching strategies
Run from: /nfs/oagrawal/HunyuanVideo/vbench_evaluation/
"""
import os
import sys
import json
from pathlib import Path
import pandas as pd
import shutil
from datetime import datetime
import argparse
import glob

def discover_videos(source_dir, include_patterns=None):
    """Discover all video.mp4 files in subdirectories of source_dir.
    
    Args:
        source_dir: Directory containing video subdirectories
        include_patterns: Optional list of patterns to include (e.g., ['fixed_0.4', 'fixed_0.5'])
                         If None, all videos are included.
    """
    video_files = {}
    
    # Look for all video.mp4 files in subdirectories
    for subdir in sorted(os.listdir(source_dir)):
        subdir_path = os.path.join(source_dir, subdir)
        if os.path.isdir(subdir_path):
            video_path = os.path.join(subdir_path, 'video.mp4')
            if os.path.exists(video_path):
                # Extract a shorter name from the folder name
                # e.g., "adaptive_0.15_0.3_2026-01-24-03:33:23_seed12345_Two_..." -> "adaptive_0.15_0.3"
                # e.g., "adaptive_0.0_0.5_f5l10_2026-01-25..." -> "adaptive_0.0_0.5_f5l10"
                parts = subdir.split('_')
                if parts[0] == 'adaptive':
                    # Check if there's a step config suffix (e.g., f5l10)
                    if len(parts) > 3 and parts[3].startswith('f') and 'l' in parts[3]:
                        short_name = f"adaptive_{parts[1]}_{parts[2]}_{parts[3]}"
                    else:
                        short_name = f"adaptive_{parts[1]}_{parts[2]}"
                elif parts[0] == 'fixed':
                    short_name = f"fixed_{parts[1]}"
                elif parts[0] == 'nocache':
                    short_name = 'nocache'
                else:
                    short_name = '_'.join(parts[:3])
                
                # Filter by include patterns if specified
                if include_patterns:
                    # Check if any pattern matches the short_name OR the full subdir name
                    if any(pattern in short_name or pattern in subdir for pattern in include_patterns):
                        video_files[short_name] = video_path
                else:
                    video_files[short_name] = video_path
    
    return video_files

def evaluate_all_videos(source_dir=None, include_patterns=None):
    """Evaluate all videos using VBench on all dimensions.
    
    Args:
        source_dir: Directory containing video subdirectories
        include_patterns: Optional list of patterns to include (e.g., ['fixed_0.4', 'fixed_0.5'])
    """
    
    # Configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Default source directory is teacache_results
    if source_dir is None:
        source_dir = os.path.join(os.path.dirname(script_dir), 'teacache_results')
    
    # Create prompt-specific output subfolder based on source directory name
    # e.g., source_dir = ".../teacache_results/lone_surfer" -> subfolder = "lone_surfer"
    source_folder_name = os.path.basename(os.path.normpath(source_dir))
    output_dir = os.path.join(script_dir, 'vbench_results', source_folder_name)
    
    # VBench paths - check multiple possible locations
    vbench_paths = [
        '/nfs/oagrawal/HunyuanVideo/VBench',
        '/workspace/VBench',
        os.path.join(os.path.dirname(script_dir), 'VBench'),
    ]
    
    vbench_root = None
    for path in vbench_paths:
        if os.path.exists(path):
            vbench_root = path
            break
    
    if vbench_root is None:
        print(f"\n✗ Error: VBench not found in any of these locations:")
        for path in vbench_paths:
            print(f"  - {path}")
        print("Please install VBench first.")
        return
    
    vbench_full_info_path = os.path.join(vbench_root, 'vbench/VBench_full_info.json')
    
    print(f"Source directory: {source_dir}")
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
    
    # Read prompt from file - first check source directory, then fall back to script directory
    prompt_file_in_source = os.path.join(source_dir, 'prompt.txt')
    prompt_file_in_script = os.path.join(script_dir, 'prompt_list.txt')
    
    if os.path.exists(prompt_file_in_source):
        prompt_file = prompt_file_in_source
        print(f"✓ Found prompt.txt in source directory")
    elif os.path.exists(prompt_file_in_script):
        prompt_file = prompt_file_in_script
        print(f"✓ Using prompt_list.txt from script directory")
    else:
        print(f"✗ Error: No prompt file found!")
        print(f"  Checked: {prompt_file_in_source}")
        print(f"  Checked: {prompt_file_in_script}")
        return
    
    with open(prompt_file, 'r') as f:
        prompt = f.read().strip()
    
    print(f"Using prompt: {prompt}")
    
    # Discover video files
    print("\nDiscovering videos...")
    if include_patterns:
        print(f"Filtering to include only: {include_patterns}")
    video_files = discover_videos(source_dir, include_patterns=include_patterns)
    
    if not video_files:
        print(f"✗ No videos found in {source_dir}")
        if include_patterns:
            print(f"  (with filter: {include_patterns})")
        return
    
    # Verify all videos exist and display them
    print(f"\nFound {len(video_files)} videos:")
    for name, video_path in video_files.items():
        size_mb = os.path.getsize(video_path) / (1024 * 1024)
        print(f"  ✓ {name}: {os.path.basename(os.path.dirname(video_path))} ({size_mb:.2f} MB)")
    
    # Import VBench
    print("\nImporting VBench...")
    # Add VBench to path if needed
    if vbench_root not in sys.path:
        sys.path.insert(0, vbench_root)
    
    try:
        from vbench import VBench
        print("  ✓ VBench imported successfully")
    except ImportError as e:
        print(f"  ✗ Error importing VBench: {e}")
        print("\nPlease install VBench first:")
        print("  cd /nfs/oagrawal/HunyuanVideo")
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
    for idx, (name, video_path) in enumerate(video_files.items(), 1):
        video_file = os.path.basename(video_path)
        
        print(f"\n{'='*70}")
        print(f"[{idx}/{len(video_files)}] Evaluating: {name}")
        print(f"Video: {video_path}")
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
                    # VBench returns scores as lists, extract first element
                    score_val = score[0] if isinstance(score, list) else score
                    print(f"    {metric}: {score_val:.4f}")
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
            # VBench returns scores as lists, extract first element
            for metric, score in results.items():
                row[metric] = score[0] if isinstance(score, list) else score
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
    print("\n" + "="*100)
    print("SUMMARY OF RESULTS")
    print("="*100 + "\n")
    
    # Get list of video names
    video_names = list(video_files.keys())
    metrics = [col for col in df.columns if col != 'Video']
    
    # Print header
    header = f"{'Metric':<25}"
    for name in video_names:
        header += f" {name:>15}"
    print(header)
    print("-" * (25 + 16 * len(video_names)))
    
    # Print each metric
    for metric in metrics:
        row = f"{metric:<25}"
        for video in video_names:
            val = df[df['Video'] == video][metric].values
            if len(val) > 0 and pd.notna(val[0]):
                row += f" {val[0]:>15.4f}"
            else:
                row += f" {'N/A':>15}"
        print(row)
    
    # Calculate and display average scores
    print("\n" + "="*100)
    print("AVERAGE SCORES (across all dimensions)")
    print("="*100 + "\n")
    
    avg_scores = {}
    for video in video_names:
        video_data = df[df['Video'] == video][metrics]
        if not video_data.empty:
            avg_score = video_data.mean(axis=1).values[0]
            avg_scores[video] = avg_score
            print(f"{video:<25}: {avg_score:.4f}")
    
    # Ranking
    if avg_scores:
        print("\n" + "="*100)
        print("RANKING (best to worst)")
        print("="*100 + "\n")
        
        sorted_videos = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
        for rank, (video, score) in enumerate(sorted_videos, 1):
            print(f"{rank}. {video:<25} (score: {score:.4f})")
    
    print(f"\n{'='*100}")
    print(f"✓ Evaluation complete!")
    print(f"✓ All results saved in: {output_dir}")
    print(f"{'='*100}\n")
    
    return all_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VBench Video Quality Evaluation')
    parser.add_argument('--source', '-s', type=str, default=None,
                        help='Source directory containing video subdirectories (default: ../teacache_results)')
    parser.add_argument('--include', '-i', type=str, default=None,
                        help='Comma-separated patterns to include (e.g., "fixed_0.4,fixed_0.5"). If not specified, all videos are evaluated.')
    args = parser.parse_args()
    
    # Parse include patterns
    include_patterns = None
    if args.include:
        include_patterns = [p.strip() for p in args.include.split(',')]
    
    print("="*100)
    print("VBench Video Quality Evaluation")
    print("="*100)
    evaluate_all_videos(source_dir=args.source, include_patterns=include_patterns)