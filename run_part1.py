#!/usr/bin/env python3
"""
Part 1: Run uncertainty evaluation without StepFactCheck (faster, ~1h)
This script runs everything except the expensive DeepSeek annotation.
"""

import subprocess
import sys
import os
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser(description="Part 1: Run evaluation without StepFactCheck", 
                                   add_help=False)
    
    # Add support for the same arguments as the original script
    parser.add_argument('--help', '-h', action='store_true', help='Show this help message and run original script help')
    
    # Parse known args to separate our arguments from hydra arguments
    args, unknown_args = parser.parse_known_args()
    
    if args.help:
        print(__doc__)
        print("\nThis script wraps eval_uhead.py with the Part 1 config.")
        print("All arguments are passed through to eval_uhead.py.")
        print("Example usage (adapt your original command):")
        print("  PYTHONPATH=./ WANDB_PROJECT=ue-reasoning DEEPSEEK_API_KEY=$(<configs/deepseek_api_key.txt) \\")
        print("    python run_part1.py model.path=Qwen/Qwen3-8B dataset=test_plan_Qwen3-8B_texts ...")
        return
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    
    # Use the Part 1 config
    eval_script = script_dir / "eval_uhead.py"
    
    if not eval_script.exists():
        print(f"Error: Eval script not found at {eval_script}")
        sys.exit(1)
    
    print("="*80)
    print("PART 1: Running uncertainty evaluation without StepFactCheck")
    print("This should take approximately 1 hour")
    print("="*80)
    
    # Change to the script directory to ensure relative paths work
    os.chdir(script_dir)
    
    # Set the hydra config to our Part 1 config
    os.environ["HYDRA_CONFIG"] = "configs/polygraph_eval_claim_reasoning_part1.yaml"
    
    # Build command with all arguments passed through
    cmd = [sys.executable, str(eval_script)] + unknown_args
    
    print(f"Running command: {' '.join(cmd)}")
    print(f"Working directory: {os.getcwd()}")
    print(f"HYDRA_CONFIG: {os.environ.get('HYDRA_CONFIG', 'not set')}")
    
    # Show environment variables that are commonly used
    env_vars = ['PYTHONPATH', 'WANDB_PROJECT', 'DEEPSEEK_API_KEY']
    for var in env_vars:
        value = os.environ.get(var, 'not set')
        if var == 'DEEPSEEK_API_KEY' and value != 'not set':
            value = '*' * 8 + ' (hidden)'
        print(f"{var}: {value}")
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n" + "="*80)
        print("PART 1 COMPLETED SUCCESSFULLY!")
        print("You can now run Part 2 to add StepFactCheck annotation.")
        print("Check the output above for the save path, you'll need it for Part 2.")
        print("="*80)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"\nError: Part 1 failed with return code {e.returncode}")
        print("Check the output above for error details")
        sys.exit(e.returncode)

if __name__ == "__main__":
    main() 