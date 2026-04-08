"""
One-shot ImageNet-1k download into the Modal Volume.

Run once from the project root:
    modal run infra/download_data.py

Idempotent - skips the download if the dataset is already present.
Expected runtime: 1-2 hours on first run (mostly bandwidth).
"""
from modal_config import app, data_vol, hf_secret, image

DATA_DIR = "/data/imagenet"
HF_CACHE_DIR = "/data/.hf_cache"


@app.function(
    image=image,
    volumes={"/data": data_vol},
    secrets=[hf_secret],
    timeout=7_200,
    memory=16_384,
)
def download_imagenet() -> None:
    from pathlib import Path

    import datasets as hf_datasets

    out = Path(DATA_DIR)
    if (out / "train").exists():
        print(f"ImageNet already present at {DATA_DIR} - skipping.")
        return

    out.mkdir(parents=True, exist_ok=True)
    Path(HF_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    print("Streaming ImageNet-1k from HuggingFace (ILSVRC/imagenet-1k)…")
    ds = hf_datasets.load_dataset(
        "ILSVRC/imagenet-1k",
        cache_dir=HF_CACHE_DIR,
    )

    print(f"Saving to {DATA_DIR}…")
    ds.save_to_disk(str(out))

    data_vol.commit()
    print(f"Done. Train: {len(ds['train']):,}  Val: {len(ds['validation']):,}")


@app.local_entrypoint()
def main() -> None:
    download_imagenet.remote()
