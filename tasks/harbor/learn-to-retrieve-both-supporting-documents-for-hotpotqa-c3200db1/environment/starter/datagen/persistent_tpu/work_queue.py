import json
import logging
import os
import threading
from typing import List

class WorkQueue:
    """State manager for the orchestration queue."""
    def __init__(self, state_file: str, all_chunks: List[List[int]], max_retries: int = 5):
        self.state_file = state_file
        self.max_retries = max_retries
        
        if os.path.exists(state_file):
            with open(state_file, "r") as f:
                self.state = json.load(f)
            logging.info(f"Loaded queue state from '{state_file}'.")
            
            if "in_progress" in self.state and len(self.state["in_progress"]) > 0:
                recovered = len(self.state["in_progress"])
                logging.info(f"Recovering {recovered} chunks from 'in_progress' -> 'pending'.")
                self.state["pending"].extend(self.state["in_progress"])
                self.state["in_progress"] = []
                self._save()
                
            # Backwards compatibility for older state files
            if "failures" not in self.state:
                self.state["failures"] = {}
            if "permanently_failed" not in self.state:
                self.state["permanently_failed"] = []
        else:
            logging.info("Creating new queue state.")
            self.state = {
                "pending": all_chunks,
                "in_progress": [],
                "completed": [],
                "permanently_failed": [],
                "failures": {}
            }
            self._save()

    def _save(self):
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2)

    def pop_pending(self) -> List[int]:
        if not self.state["pending"]:
            return None
        chunk = self.state["pending"].pop(0)
        self.state["in_progress"].append(chunk)
        self._save()
        return chunk

    def mark_completed(self, chunk: List[int]):
        if chunk in self.state["in_progress"]:
            self.state["in_progress"].remove(chunk)
        self.state["completed"].append(chunk)
        self._save()

    def mark_failed(self, chunk: List[int]):
        """Increments failure count and either retry or map to permanently failed."""
        if chunk in self.state["in_progress"]:
            self.state["in_progress"].remove(chunk)
            
        chunk_str = str(chunk)
        self.state["failures"][chunk_str] = self.state["failures"].get(chunk_str, 0) + 1
        
        if self.state["failures"][chunk_str] > self.max_retries:
            logging.error(f"Chunk {chunk} exceeded max retries ({self.max_retries}). Marking as PERMANENTLY FAILED.")
            self.state["permanently_failed"].append(chunk)
        else:
            logging.warning(f"Chunk {chunk} failed. Re-queuing in pending (Retry {self.state['failures'][chunk_str]}/{self.max_retries}).")
            self.state["pending"].append(chunk)
            
        self._save()

    def is_done(self) -> bool:
        return len(self.state["pending"]) == 0 and len(self.state["in_progress"]) == 0

    def print_status(self):
        logging.info(f"QUEUE STATE: {len(self.state['pending'])} pending, "
                     f"{len(self.state['in_progress'])} in progress, "
                     f"{len(self.state['completed'])} completed, "
                     f"{len(self.state['permanently_failed'])} permanently failed.")
