import os
import shutil
import time
from huggingface_hub import snapshot_download

# 1. Force clear the corrupted cache and locks
cache_dir = r"D:\huggingface_cache\hub\models--BAAI--bge-base-en-v1.5"
locks_dir = r"D:\huggingface_cache\hub\.locks\models--BAAI--bge-base-en-v1.5"

print("Cleaning up corrupted partial downloads and locks...")
try:
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    if os.path.exists(locks_dir):
        shutil.rmtree(locks_dir)
    print("Cleanup successful.")
except Exception as e:
    print(f"Cleanup warning: {e}")

time.sleep(1)

# 2. Download cleanly
print("\n" + "="*50)
print("Starting fresh download of BAAI/bge-base-en-v1.5")
print("This is 438MB. Please wait...")
print("="*50 + "\n")

snapshot_download(repo_id="BAAI/bge-base-en-v1.5")

print("\n" + "="*50)
print("SUCCESS! Model downloaded.")
print("="*50)
