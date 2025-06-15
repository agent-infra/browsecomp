#!/usr/bin/env python3
"""
BrowseComp evaluation command-line interface.
"""

import argparse
import os
import signal
import sys

from browsecomp_eval import run_browsecomp_eval


def setup_signal_handlers():
    """Setup signal handlers to ensure graceful shutdown on interruption."""
    def signal_handler(sig, frame):
        print("\n\nEvaluation interrupted! Interim results have been saved.")
        print("You can resume later or check the *_in_progress.html report.")
        sys.exit(0)
    
    # Register handler for interrupt signals
    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGBREAK'):  # Windows-specific
        signal.signal(signal.SIGBREAK, signal_handler)


def main():
    # Example usage: python main.py
    parser = argparse.ArgumentParser(description="Run BrowseComp evaluation")
    parser.add_argument("--python-script", type=str, default="model_runner.py",
                       help="Path to Python script that generates model responses")
    parser.add_argument("--command", type=str,
                       help="Shell command that generates model responses (e.g., 'agent-tars run')")
    parser.add_argument("--model-name", type=str,
                       help="Model name to pass to runner")
    parser.add_argument("--examples", type=int, default=10,
                       help="Number of examples to evaluate")
    # Add new arguments for grader model configuration
    parser.add_argument("--grader-model-name", type=str, default="gpt-4",
                       help="Model name to use for grading (default: gpt-4)")
    parser.add_argument("--grader-api-key", type=str,
                       help="API key to use for grader model")
    parser.add_argument("--grader-base-url", type=str,
                       help="Base URL to use for grader model API")
    parser.add_argument("--exclude", type=str, nargs="+", dest="exclude_keywords",
                       help="Keywords to exclude from evaluation (examples containing these keywords will be skipped)")
    args = parser.parse_args()

    # Setup signal handlers to handle interruptions gracefully
    setup_signal_handlers()

    # Check for mutually exclusive parameters
    if args.command and args.python_script != "model_runner.py":
        print("Warning: Both --command and --python-script specified. Using --command.")

    run_browsecomp_eval(
        runner_path=args.python_script,
        model_name=args.model_name,
        num_examples=args.examples,
        cli_format=args.command,
        grader_model_name=args.grader_model_name,
        grader_api_key=args.grader_api_key,
        grader_base_url=args.grader_base_url,
        exclude_keywords=args.exclude_keywords
    )


if __name__ == "__main__":
    main()
