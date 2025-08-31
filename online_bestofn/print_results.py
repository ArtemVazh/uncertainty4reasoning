#!/usr/bin/env python3
"""
Print a formatted table of online best-of-n results for a given dataset folder.
"""

import argparse
import os
import torch
import glob
from typing import Dict, List, Any
from tabulate import tabulate
import numpy as np


def load_results(file_path: str) -> Dict[str, Any]:
    """Load results from a .pt file"""
    try:
        results = torch.load(file_path, map_location='cpu')
        return results
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def extract_metrics(results: List[Dict]) -> Dict[str, Any]:
    """Extract key metrics from results"""
    if not results:
        return {}
    
    # Count total, completed, and correct
    total = len(results)
    completed = sum(1 for r in results if r.get("completed", False))
    errors = sum(1 for r in results if "error" in r)
    
    # Check for different correctness modes
    correct_exact = sum(1 for r in results if r.get("is_correct_exact_match", False))
    correct_deepseek = sum(1 for r in results if r.get("is_correct_deepseek", False))
    
    # Use whichever correctness mode is available
    if any("is_correct_exact_match" in r for r in results):
        correct = correct_exact
        correctness_mode = "exact_match"
    elif any("is_correct_deepseek" in r for r in results):
        correct = correct_deepseek
        correctness_mode = "deepseek"
    else:
        # Fallback to is_correct
        correct = sum(1 for r in results if r.get("is_correct", False))
        correctness_mode = "unknown"
    
    # Calculate accuracy
    accuracy = correct / total * 100 if total > 0 else 0
    accuracy_of_completed = correct / completed * 100 if completed > 0 else 0
    
    # Calculate average steps
    all_steps = [len(r.get("steps", [])) for r in results if "steps" in r and r.get("completed", False)]
    avg_steps = np.mean(all_steps) if all_steps else 0
    
    # For models with specific scores
    metrics = {
        "total": total,
        "completed": completed,
        "correct": correct,
        "errors": errors,
        "accuracy": accuracy,
        "accuracy_of_completed": accuracy_of_completed,
        "avg_steps": avg_steps,
        "correctness_mode": correctness_mode
    }
    
    # Add model-specific metrics
    if any("step_scores" in r for r in results):
        # UHead or PRM with step scores
        all_scores = []
        for r in results:
            if "step_scores" in r and r["step_scores"]:
                all_scores.extend(r["step_scores"])
        if all_scores:
            metrics["avg_step_score"] = np.mean(all_scores)
            
    if any("validity_scores" in r for r in results):
        # ReasonEval with validity/redundancy scores
        all_validity = []
        all_redundancy = []
        for r in results:
            if "validity_scores" in r and r["validity_scores"]:
                all_validity.extend(r["validity_scores"])
            if "redundancy_scores" in r and r["redundancy_scores"]:
                all_redundancy.extend(r["redundancy_scores"])
        if all_validity:
            metrics["avg_validity"] = np.mean(all_validity)
        if all_redundancy:
            metrics["avg_redundancy"] = np.mean(all_redundancy)
    
    return metrics


def get_model_type(filename: str) -> str:
    """Determine model type from filename"""
    if "ReasonEval" in filename:
        if "_validity" in filename:
            return "ReasonEval (validity)"
        elif "_redundancy" in filename:
            return "ReasonEval (redundancy)"
        elif "_both" in filename:
            return "ReasonEval (both)"
        else:
            return "ReasonEval"
    elif "uhead" in filename.lower():
        return "UHead"
    elif "prm" in filename.lower() or "PRM" in filename:
        return "PRM"
    else:
        return "Unknown"


def print_results_table(dataset_folder: str, sort_by: str = "name"):
    """Print a formatted table of all results in the dataset folder"""
    
    if not os.path.exists(dataset_folder):
        print(f"Error: Dataset folder '{dataset_folder}' does not exist")
        return
    
    # Find all .pt files in the folder
    pt_files = glob.glob(os.path.join(dataset_folder, "*.pt"))
    
    if not pt_files:
        print(f"No .pt result files found in '{dataset_folder}'")
        return
    
    # Sort files for consistent ordering
    pt_files.sort()
    
    # Collect data for table
    table_data = []
    
    for pt_file in pt_files:
        filename = os.path.basename(pt_file)
        model_name = filename.replace(".pt", "")
        model_type = get_model_type(filename)
        
        # Load results
        results = load_results(pt_file)
        if results is None:
            continue
        
        # Extract metrics
        metrics = extract_metrics(results)
        
        if not metrics:
            continue
        
        # Add row to table
        table_data.append([
            model_name,
            model_type,
            metrics["total"],
            metrics["completed"],
            metrics["errors"],
            f"{metrics['accuracy']:.1f}%",
            f"{metrics['accuracy_of_completed']:.1f}%",
            f"{metrics['avg_steps']:.1f}",
            metrics["correctness_mode"]
        ])
    
    if not table_data:
        print("No valid results found to display")
        return
    
    # Sort table based on sort_by parameter
    if sort_by == "accuracy":
        # Sort by accuracy (descending)
        table_data.sort(key=lambda x: float(x[5].rstrip('%')), reverse=True)
    elif sort_by == "type":
        # Sort by type, then by name
        table_data.sort(key=lambda x: (x[1], x[0]))
    # else: already sorted by name (default)
    
    # Print dataset name
    dataset_name = os.path.basename(dataset_folder)
    print(f"\n{'='*80}")
    print(f"Results for dataset: {dataset_name}")
    print(f"{'='*80}\n")
    
    # Define headers
    headers = [
        "Model",
        "Type",
        "Total",
        "Completed",
        "Errors",
        "Accuracy",
        "Acc (Completed)",
        "Avg Steps",
        "Correctness"
    ]
    
    # Print table
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Print summary statistics
    print(f"\nTotal result files: {len(table_data)}")
    
    # Group by model type
    type_counts = {}
    for row in table_data:
        model_type = row[1]
        if model_type not in type_counts:
            type_counts[model_type] = 0
        type_counts[model_type] += 1
    
    print("\nResults by type:")
    for model_type, count in sorted(type_counts.items()):
        print(f"  - {model_type}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description="Print a table of online best-of-n results for a dataset folder"
    )
    parser.add_argument(
        "dataset_folder",
        help="Path to the dataset folder containing .pt result files"
    )
    parser.add_argument(
        "--sort-by",
        choices=["name", "accuracy", "type"],
        default="name",
        help="Sort table by specified column (default: name)"
    )
    
    args = parser.parse_args()
    
    print_results_table(args.dataset_folder, sort_by=args.sort_by)


if __name__ == "__main__":
    main()