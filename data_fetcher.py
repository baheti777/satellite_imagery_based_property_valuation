import pandas as pd
import os
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ================= LOAD CSV =================
train = pd.read_csv(r"C:\Users\bhuvi\Downloads\train_cdc.csv")
test  = pd.read_csv(r"C:\Users\bhuvi\Downloads\test_cdc.csv")

# ================= CONFIG =================
MAPBOX_TOKEN = os.getenv("MAPBOX_TOKEN")   # make sure token is set
IMAGE_SIZE = "224x224"
ZOOM = 18

BASE_DIR  = "data/images/raw"
TRAIN_DIR = f"{BASE_DIR}/train"
TEST_DIR  = f"{BASE_DIR}/test"

MAX_WORKERS = 8            # SAFE for 20k images
TIMEOUT = 20               # seconds
RETRIES = 3

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

# =========================================

def build_mapbox_url(lat, lon):
    return (
        f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/"
        f"{lon},{lat},{ZOOM}/{IMAGE_SIZE}"
        f"?access_token={MAPBOX_TOKEN}"
    )

def download_image(idx, row, out_dir, session):
    lat, lon = row["lat"], row["long"]

    if pd.isna(lat) or pd.isna(lon):
        return idx, None

    img_path = f"{out_dir}/{idx}.jpg"
    if os.path.exists(img_path):
        return idx, img_path

    url = build_mapbox_url(lat, lon)

    for _ in range(RETRIES):
        try:
            r = session.get(url, timeout=TIMEOUT)
            if r.status_code == 200:
                with open(img_path, "wb") as f:
                    f.write(r.content)
                return idx, img_path

            elif r.status_code == 429:
                time.sleep(3)  # rate-limit backoff

        except Exception:
            time.sleep(2)

    return idx, None


def process_dataframe(df, out_dir, output_csv):
    existing = {
        int(f.split(".")[0])
        for f in os.listdir(out_dir)
        if f.endswith(".jpg")
    }

    df_to_download = df.loc[~df.index.isin(existing)]

    image_paths = [None] * len(df)

    with requests.Session() as session:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [
                executor.submit(download_image, idx, row, out_dir, session)
                for idx, row in df_to_download.iterrows()
            ]

            for future in tqdm(as_completed(futures),
                               total=len(futures),
                               desc=f"Downloading → {out_dir}"):
                idx, path = future.result()
                image_paths[idx] = path

    for idx in existing:
        image_paths[idx] = f"{out_dir}/{idx}.jpg"

    df["image_path"] = image_paths

    # ✅ drop only for output CSV (NOT in memory)
    df.dropna(subset=["image_path"]).to_csv(output_csv, index=False)

# ================= RUN =================
start = time.time()

process_dataframe(train, TRAIN_DIR, "train_with_images.csv")
process_dataframe(test, TEST_DIR, "test_with_images.csv")

# ================= LOG FAILED DOWNLOADS =================

failed_train = train[train["image_path"].isna()]
failed_test  = test[test["image_path"].isna()]

failed_train.to_csv("failed_train_images.csv", index=False)
failed_test.to_csv("failed_test_images.csv", index=False)

print(f"❌ Failed train images: {len(failed_train)}")
print(f"❌ Failed test images : {len(failed_test)}")

print(f"\n✅ Completed in {round((time.time() - start) / 60, 2)} minutes")
