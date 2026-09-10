#!/usr/bin/env python3
"""
Single-TPU Orchestrator - Main CLI
"""
import argparse
import logging
from typing import List

from work_queue import WorkQueue
from tpu_manager import TPUOrchestrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")

def parse_parquets(parquet_arg: str) -> List[int]:
    if "-" in parquet_arg:
        start, end = map(int, parquet_arg.split("-"))
        return list(range(start, end + 1))
    else:
        return [int(x.strip()) for x in parquet_arg.split(",")]

def main():
    parser = argparse.ArgumentParser(description="Single-TPU Preemption Orchestrator")
    parser.add_argument("--project", required=True)
    parser.add_argument("--zone", required=True)
    parser.add_argument("--tpu-name", required=True, help="Queued resource ID to spawn and babysit")
    parser.add_argument("--tpu-type", required=True, help="e.g. v6e-8 or v6e-4")
    parser.add_argument("--env-file", default="/home/suhas/test/memory-layers/.env", help="Local env file to load and push to TPU")
    parser.add_argument("--parquets", required=True, help="Range e.g. 0-100 or list 0,1,2,3")
    parser.add_argument("--chunk-size", type=int, default=1, help="Parquets to send to the TPU per loop iteration")
    parser.add_argument("--state-file", default="orchestrator_state.json", help="File to track global pending/completed queue crossing sessions")
    parser.add_argument("--setup-script", required=True, help="Path to local bash script containing apt-get, pip install, git clone, etc.")
    parser.add_argument("--run-command", required=True, help="Execution command containing {CHUNKS} placeholder. e.g. 'uv run foo.py --parquets {CHUNKS}'")
    parser.add_argument("--work-dir", default="memory-layers", help="Directory on the TPU VM to cd into before executing the run command")
    parser.add_argument("--max-retries", type=int, default=5, help="Number of times to retry a failed chunk before permanently skipping it.")
    
    args = parser.parse_args()

    # Build full chunks array
    all_parquets = parse_parquets(args.parquets)
    all_chunks = [all_parquets[i:i + args.chunk_size] for i in range(0, len(all_parquets), args.chunk_size)]

    # Initialize queue manager
    queue = WorkQueue(args.state_file, all_chunks, max_retries=args.max_retries)
    queue.print_status()

    orchestrator = TPUOrchestrator(
        tpu_id=args.tpu_name,
        tpu_type=args.tpu_type,
        zone=args.zone,
        project_id=args.project,
        env_file_path=args.env_file,
        work_queue=queue,
        setup_script_path=args.setup_script,
        run_command_template=args.run_command,
        work_dir=args.work_dir
    )
    
    try:
        orchestrator.run()
    except KeyboardInterrupt:
        logging.info("Caught Ctrl+C! The queue was safely persisted to disk. Exiting orchestrator.")

if __name__ == "__main__":
    main()
