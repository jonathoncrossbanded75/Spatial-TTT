# python evaluation/spatial/misc/summarize_vsibench_results.py \
#     evaluation/spatial/results/vsibench/Spatial-2B-base-32f \
#     evaluation/spatial/results/vsibench/Spatial-2B-base-32f-low-res \
#     evaluation/spatial/results/vsibench/Spatial-2B-base-64f \
#     evaluation/spatial/results/vsibench/Spatial-2B-base-64f-low-res \
#     evaluation/spatial/results/vsibench/Spatial-2B-base-128f \
#     evaluation/spatial/results/vsibench/Spatial-2B-base-128f-low-res

import argparse
import json
import os
import glob
from pathlib import Path
import sys

def find_result_file(folder_path):
    # Search for json files ending with _results.json recursively
    files = list(Path(folder_path).rglob("*_results.json"))
    if not files:
        return None
    # Sort by name (which usually starts with timestamp e.g. 20260118_...) 
    # taking the last one (latest)
    files.sort(key=lambda x: x.name, reverse=True)
    return files[0]

def parse_results(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        results = data.get("results", {})
        
        metrics = None
        # Look for the task result that contains "vsibench_score,none"
        # The key for the task might be "vsibench" or similar
        for task, task_res in results.items():
            if "vsibench_score,none" in task_res:
                metrics = task_res["vsibench_score,none"]
                break
        
        if metrics is None:
            # Fallback: check if the root has "vsibench_score,none" (unlikely based on structure but possible)
            if "vsibench_score,none" in results:
                 metrics = results["vsibench_score,none"]
            else:
                 return None

        # Extract specific keys
        keys_map = {
            "Appr Order": "obj_appearance_order_accuracy",
            "Abs Dist": "object_abs_distance_MRA:.5:.95:.05",
            "Counting": "object_counting_MRA:.5:.95:.05",
            "Rel Dist": "object_rel_distance_accuracy",
            "Obj Size": "object_size_estimation_MRA:.5:.95:.05",
            "Room Size": "room_size_estimation_MRA:.5:.95:.05",
            "Route": "route_planning_accuracy",
            "Rel Dir": "object_rel_direction_accuracy",
            "Overall": "overall"
        }
        
        row = {}
        for short_name, key in keys_map.items():
            val = metrics.get(key, "N/A")
            if isinstance(val, (int, float)):
                row[short_name] = round(val * 100, 2) # Convert to percentage for readability
            else:
                row[short_name] = val
        
        return row
    except Exception as e:
        print(f"Error parsing {file_path}: {e}", file=sys.stderr)
        return None

def main():
    parser = argparse.ArgumentParser(description="Summarize VSIBench results.")
    parser.add_argument("folders", nargs="+", help="List of result folders")
    args = parser.parse_args()

    headers = ["Model", "Appr Order", "Abs Dist", "Counting", "Rel Dist", "Obj Size", "Room Size", "Route", "Rel Dir", "Overall"]
    table_data = []

    for folder in args.folders:
        folder_path = Path(folder)
        if not folder_path.exists():
            print(f"Warning: Folder {folder} does not exist.", file=sys.stderr)
            continue
            
        result_file = find_result_file(folder_path)
        if not result_file:
            print(f"Warning: No result file found in {folder}", file=sys.stderr)
            continue
            
        metrics = parse_results(result_file)
        if metrics:
            row = [folder_path.name] # Use folder name as model name
            for h in headers[1:]:
                row.append(metrics.get(h, "N/A"))
            table_data.append(row)
        else:
            print(f"Warning: Could not parse results from {result_file}", file=sys.stderr)

    if not table_data:
        print("No results found.")
        return

    # Print table
    col_widths = [len(h) for h in headers]
    for row in table_data:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))
    
    # Add spacing
    col_widths = [w + 2 for w in col_widths]
    
    header_str = "".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print("-" * len(header_str))
    print(header_str)
    print("-" * len(header_str))
    
    for row in table_data:
        print("".join(str(val).ljust(w) for val, w in zip(row, col_widths)))
    print("-" * len(header_str))

if __name__ == "__main__":
    main()
