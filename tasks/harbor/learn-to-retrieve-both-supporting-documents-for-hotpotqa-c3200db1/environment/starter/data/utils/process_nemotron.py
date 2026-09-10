import os
import time
import logging
from typing import List, Dict, Any
from multiprocessing import Pool
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import HfApi

# Configuration
HF_TOKEN = "XXX"
TARGET_REPO = "vm2825/nemotron-cc-v21-Parsed-QA3"
LOCAL_DIR = "./temp_nemotron_pq"
BATCH_SIZE = 5000  # records per IPC batch
PARQUET_ROWS = 2_000_000  # rows per parquet file
UPLOAD_THRESHOLD_BYTES = 20 * 1024 * 1024 * 1024  # 20 GB upload threshold
NUM_WORKERS = 200  # Leave some cores for the main process and OS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def process_item(item: Dict[str, Any]) -> List[Dict[str, str]]:
    text = item.get("text", "")
    uuid = item.get("uuid", "")
    
    parts = text.split("\nQuestion: ")
    if not parts:
        return []
        
    pos_doc = parts[0].strip()
    
    results = []
    for part in parts[1:]:
        if "\nAnswer:" in part:
            splits = part.split("\nAnswer:", 1)
            results.append({
                "id": uuid,
                "pos_doc": pos_doc,
                "question": splits[0].strip(),
                "answer": splits[1].strip()
            })
    return results

def process_batch(batch: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    res = []
    for item in batch:
        res.extend(process_item(item))
    return res

def batch_generator(iterable, batch_size):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def get_dir_size(path):
    total = 0
    for file in os.listdir(path):
        if file.endswith(".parquet"):
            total += os.path.getsize(os.path.join(path, file))
    return total

def upload_and_clear(api: HfApi, local_dir: str, upload_count: int):
    logging.info(f"Uploading batch {upload_count} to {TARGET_REPO}...")
    try:
        api.upload_folder(
            folder_path=local_dir,
            repo_id=TARGET_REPO,
            repo_type="dataset",
            allow_patterns="*.parquet",
            commit_message=f"Upload batch {upload_count}",
        )
        logging.info("Upload successful. Clearing local files...")
        for file in os.listdir(local_dir):
            if file.endswith(".parquet"):
                os.remove(os.path.join(local_dir, file))
    except Exception as e:
        logging.error(f"Upload failed: {e}. Keeping local files for next retry.")

def main():
    os.makedirs(LOCAL_DIR, exist_ok=True)
    
    api = HfApi(token=HF_TOKEN)
    try:
        api.create_repo(repo_id=TARGET_REPO, repo_type="dataset", private=False, exist_ok=True)
        logging.info(f"Repository {TARGET_REPO} ready.")
    except Exception as e:
        logging.info(f"Repo check: {e}")

    logging.info("Initializing dataset stream (SINGLE connection to avoid DDOS flagging)...")
    ds = load_dataset(
        "nvidia/Nemotron-CC-v2.1",
        "High-Quality-DQA",
        split="train",
        streaming=True,
        token=HF_TOKEN
    )
    
    schema = pa.schema([
        ('id', pa.string()),
        ('pos_doc', pa.string()),
        ('question', pa.string()),
        ('answer', pa.string())
    ])

    pool = Pool(processes=NUM_WORKERS)
    
    current_chunk = []
    chunk_index = 0
    upload_count = 0
    total_processed_rows = 0
    start_time = time.time()
    
    try:
        batches = batch_generator(iter(ds), BATCH_SIZE)
        for results in pool.imap_unordered(process_batch, batches):
            current_chunk.extend(results)
            
            if len(current_chunk) >= PARQUET_ROWS:
                # To prevent memory spikes, we take exactly PARQUET_ROWS
                to_write = current_chunk[:PARQUET_ROWS]
                current_chunk = current_chunk[PARQUET_ROWS:]
                
                pq_path = os.path.join(LOCAL_DIR, f"nemotron_dqa_{chunk_index:06d}.parquet")
                table = pa.Table.from_pylist(to_write, schema=schema)
                pq.write_table(table, pq_path)
                
                total_processed_rows += len(to_write)
                chunk_index += 1
                logging.info(f"Wrote {len(to_write)} rows to {pq_path}. Total so far: {total_processed_rows}")
                
                dir_size = get_dir_size(LOCAL_DIR)
                if dir_size >= UPLOAD_THRESHOLD_BYTES:
                    upload_count += 1
                    upload_and_clear(api, LOCAL_DIR, upload_count)
                    
        # Final flush
        if current_chunk:
            pq_path = os.path.join(LOCAL_DIR, f"nemotron_dqa_{chunk_index:06d}.parquet")
            table = pa.Table.from_pylist(current_chunk, schema=schema)
            pq.write_table(table, pq_path)
            total_processed_rows += len(current_chunk)
            logging.info(f"Wrote final {len(current_chunk)} rows to {pq_path}")
            
        dir_size = get_dir_size(LOCAL_DIR)
        if dir_size > 0:
            upload_count += 1
            upload_and_clear(api, LOCAL_DIR, upload_count)
            
        logging.info(f"Complete! Total rows: {total_processed_rows}. Time taken: {time.time() - start_time:.2f}s")
            
    except KeyboardInterrupt:
        logging.info("Interrupted! Exiting gracefully...")
    except Exception as e:
        logging.error(f"Error during processing: {e}")
    finally:
        pool.terminate()
        pool.join()

if __name__ == "__main__":
    main()
