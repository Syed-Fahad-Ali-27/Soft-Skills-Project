import requests, zipfile, io, os

GTFS_URL = "https://api-management-discovery-production.azure-api.net/api/gtfs/feed/stibmivb/static"

def download_gtfs_static(dest_folder="data/raw/gtfs_static"):
    os.makedirs(dest_folder, exist_ok=True)
    r = requests.get(GTFS_URL)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(dest_folder)
    print(f"Extracted {len(z.namelist())} files to {dest_folder}")

if __name__ == "__main__":
    download_gtfs_static()