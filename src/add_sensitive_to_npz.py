from __future__ import annotations

"""Add s_train/s_val/s_test to data/processed/<dataset>/transformed.npz.

This makes the main experimental pipeline independent from pyarrow at run time.
"""

from pathlib import Path
import numpy as np

from src.data_loader import PROCESSED_ROOT, SUPPORTED_DATASETS, _load_sensitive_from_files


def patch_dataset(dataset: str, force: bool = False) -> None:
    processed_dir = PROCESSED_ROOT / dataset
    npz_path = processed_dir / "transformed.npz"
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    old = np.load(npz_path, allow_pickle=True)
    if (not force) and all(k in old.files for k in ["s_train", "s_val", "s_test"]):
        print(f"[SKIP] {dataset}: transformed.npz đã có s_train/s_val/s_test")
        return

    data = {k: old[k] for k in old.files if k not in {"s_train", "s_val", "s_test", "sensitive_name"}}

    s_train, sensitive_name = _load_sensitive_from_files(dataset, "train", processed_dir)
    s_val, _ = _load_sensitive_from_files(dataset, "val", processed_dir)
    s_test, _ = _load_sensitive_from_files(dataset, "test", processed_dir)

    data["s_train"] = s_train.astype(int)
    data["s_val"] = s_val.astype(int)
    data["s_test"] = s_test.astype(int)
    data["sensitive_name"] = np.array(sensitive_name)

    backup = npz_path.with_suffix(".npz.bak_before_sensitive")
    if not backup.exists():
        backup.write_bytes(npz_path.read_bytes())

    np.savez_compressed(npz_path, **data)
    print(f"[OK] {dataset}: added sensitive='{sensitive_name}' to {npz_path}")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="Ghi đè s_train/s_val/s_test nếu đã tồn tại")
    args = ap.parse_args()

    for dataset in SUPPORTED_DATASETS:
        patch_dataset(dataset, force=args.force)


if __name__ == "__main__":
    main()
