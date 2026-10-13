import csv
import requests
import concurrent.futures
import time

CSV_PATH = r"D:\clone_repo_ats-1\s1jobs.csv"
API_URL = "http://127.0.0.1:8000/api/v1/jobs"

def import_jd(row):
    try:
        jd_content = row[6] # 7th column
        title = row[15] # 16th column
        
        content = f"Title: {title}\n\n{jd_content}"
        
        response = requests.post(API_URL, data={"content": content}, timeout=30)
        if response.status_code == 200:
            return True
        return False
    except Exception as e:
        return False

def main():
    print(f"Reading {CSV_PATH}...")
    rows_to_process = []
    
    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if len(row) > 15:
                rows_to_process.append(row)
                
    print(f"Importing {len(rows_to_process)} job descriptions...")
    start_time = time.time()
    
    success_count = 0
    # Process all rows using thread pool
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(import_jd, row): row for row in rows_to_process}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            success = future.result()
            if success:
                success_count += 1
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(rows_to_process)}...")
                
    elapsed = time.time() - start_time
    print(f"Done! Successfully imported {success_count} JDs with embeddings in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    main()
