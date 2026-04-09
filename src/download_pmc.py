"""
download_pmc.py
Downloads PMC Open Access XML archives via HTTPS (more stable than FTP).
Usage: python src/download_pmc.py
Output: data/pmc_literature/xml/ will contain thousands of .xml files
"""
 
import os
import tarfile
import time
import requests
from tqdm import tqdm
 
# ── Configuration ──────────────────────────────────────────
BASE_URL  = "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/xml/"
RAW_DIR   = "data/pmc_literature/raw/"
XML_DIR   = "data/pmc_literature/xml/"
NUM_ARCHIVES = 3      # 3 baseline files = ~45K–60K articles, more than enough
                      # Increase to 5 only if you need more
# ───────────────────────────────────────────────────────────
 
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(XML_DIR, exist_ok=True)
 
 
def list_archives():
    """Scrape the HTTPS index page for baseline .tar.gz filenames."""
    print("Fetching archive list from NCBI...")
    r = requests.get(BASE_URL, timeout=30)
    r.raise_for_status()
 
    archives = []
    for line in r.text.splitlines():
        # The NCBI index page lists files as href links
        if ".tar.gz" in line and "baseline" in line and "incr" not in line:
            # Extract filename from href="filename.tar.gz"
            start = line.find('href="') + 6
            end   = line.find('"', start)
            if start > 5 and end > start:
                fname = line[start:end]
                if fname.endswith(".tar.gz"):
                    archives.append(fname)
 
    # Fallback: if parsing fails, hardcode known 2026 baseline filenames
    if not archives:
        print("  Index parse failed — using known baseline filenames.")
        archives = [
            "oa_comm_xml.PMC000xxxxxx.baseline.2026-01-23.tar.gz",
            "oa_comm_xml.PMC001xxxxxx.baseline.2026-01-23.tar.gz",
            "oa_comm_xml.PMC002xxxxxx.baseline.2026-01-23.tar.gz",
            "oa_comm_xml.PMC003xxxxxx.baseline.2026-01-23.tar.gz",
            "oa_comm_xml.PMC004xxxxxx.baseline.2026-01-23.tar.gz",
        ]
 
    print(f"Found {len(archives)} baseline archives.")
    return archives
 
 
def download_file_https(fname, max_retries=20):
    """
    Download a file via HTTPS with resume support.
    Retries indefinitely on connection drops — the resume picks up
    exactly where it left off each time.
    """
    url        = BASE_URL + fname
    local_path = os.path.join(RAW_DIR, fname)
 
    for attempt in range(max_retries):
        try:
            # Check how much we already have
            existing = os.path.getsize(local_path) if os.path.exists(local_path) else 0
 
            headers = {}
            if existing > 0:
                headers["Range"] = f"bytes={existing}-"
                print(f"  Resuming from {existing / 1e6:.1f} MB...")
            else:
                print(f"  Starting download: {fname}")
 
            r = requests.get(
                url,
                headers=headers,
                stream=True,
                timeout=60,          # 60s timeout per chunk read — generous
            )
 
            # 206 = partial content (resume), 200 = full file
            if r.status_code == 416:
                # Range not satisfiable = file already complete
                print(f"  File already fully downloaded.")
                return local_path
 
            r.raise_for_status()
 
            total = int(r.headers.get("content-length", 0))
            mode  = "ab" if existing > 0 else "wb"   # append vs overwrite
 
            with open(local_path, mode) as f, tqdm(
                total=existing + total,
                initial=existing,
                unit="B",
                unit_scale=True,
                desc=fname[:35],
                miniters=1,
            ) as pbar:
                for chunk in r.iter_content(chunk_size=1024 * 256):   # 256 KB chunks
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
 
            print(f"  Download complete: {fname}")
            return local_path
 
        except (requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ConnectionError,
                requests.exceptions.ReadTimeout) as e:
            wait = min(10 * (attempt + 1), 60)    # back off: 10s, 20s, 30s... max 60s
            print(f"\n  Connection dropped (attempt {attempt+1}/{max_retries}): {e}")
            print(f"  Waiting {wait}s then resuming...")
            time.sleep(wait)
 
        except Exception as e:
            print(f"\n  Unexpected error: {e}")
            raise
 
    raise RuntimeError(f"Failed to download {fname} after {max_retries} attempts.")
 
 
def extract_archive(tar_path):
    """Extract a .tar.gz and delete it to save disk space."""
    print(f"  Extracting: {os.path.basename(tar_path)}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        members = tar.getmembers()
        for member in tqdm(members, desc="  Extracting", unit="file"):
            tar.extract(member, path=XML_DIR)
    os.remove(tar_path)
    print(f"  Archive deleted (disk space freed).")
 
 
def count_xml_files():
    count = 0
    for root, dirs, files in os.walk(XML_DIR):
        for f in files:
            if f.endswith(".xml"):
                count += 1
    return count
 
 
if __name__ == "__main__":
    archives     = list_archives()
    to_download  = archives[:NUM_ARCHIVES]
 
    print(f"\nWill download {len(to_download)} archives.")
    print("Connection drops are handled automatically — just leave it running.\n")
 
    for i, archive_name in enumerate(to_download, 1):
        print(f"\n── [{i}/{len(to_download)}] {archive_name} ──")
        tar_path = download_file_https(archive_name)
        extract_archive(tar_path)
        print(f"  Total XML files so far: {count_xml_files():,}")
 
    print(f"\n✓ Done! Total XML files in {XML_DIR}: {count_xml_files():,}")
    print("You now have more than enough to parse 500 articles.")