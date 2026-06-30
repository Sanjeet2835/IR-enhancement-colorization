from pathlib import Path

import numpy as np
import yaml


# ==========================================================
# Paths
# ==========================================================

WORKSPACE = Path(
    "/home/sanjeet/ai_workspace/IR image enhancement and colorization"
)

DATASET_ROOT = WORKSPACE / "IR-colorization-BAH2026"

PATCHES_DIR = DATASET_ROOT / "output" / "patches"

TRAIN_SPLIT = DATASET_ROOT / "splits" / "train.txt"

OUTPUT_FILE = (
    WORKSPACE
    / "IR-enhancement-colorization"
    / "configs"
    / "dataset_stats.yaml"
)


def compute_stats(file_name):

    with open(TRAIN_SPLIT, "r") as f:
        train_samples = [
            line.strip()
            for line in f
            if line.strip()
        ]

    first = np.load(
        PATCHES_DIR / train_samples[0] / file_name,
        allow_pickle=False,
    )

    channels = first.shape[0]

    pixel_sum = np.zeros(channels, dtype=np.float64)
    pixel_sq_sum = np.zeros(channels, dtype=np.float64)
    pixel_count = np.zeros(channels, dtype=np.int64)

    pixel_min = np.full(channels, np.inf)
    pixel_max = np.full(channels, -np.inf)

    for sample in train_samples:

        arr = np.load(
            PATCHES_DIR / sample / file_name,
            allow_pickle=False,
        ).astype(np.float64)

        for c in range(channels):

            channel = arr[c]

            # Ignore NoData pixels
            channel = channel[channel != 0]

            pixel_sum[c] += channel.sum()
            pixel_sq_sum[c] += np.square(channel).sum()
            pixel_count[c] += channel.size

            pixel_min[c] = min(pixel_min[c], channel.min())
            pixel_max[c] = max(pixel_max[c], channel.max())

    mean = pixel_sum / pixel_count

    std = np.sqrt(
        pixel_sq_sum / pixel_count - mean**2
    )

    data_range = pixel_max - pixel_min

    print("=" * 60)
    print(file_name)
    print("=" * 60)

    if channels == 1:

        print(f"Samples    : {len(train_samples)}")
        print(f"Pixels     : {pixel_count[0]:,}")
        print(f"Mean       : {mean[0]:.6f}")
        print(f"Std        : {std[0]:.6f}")
        print(f"Min        : {pixel_min[0]:.6f}")
        print(f"Max        : {pixel_max[0]:.6f}")
        print(f"Data Range : {data_range[0]:.6f}")

    else:

        names = ["R", "G", "B"]

        for i in range(channels):

            print(
                f"{names[i]}\n"
                f"  Mean       : {mean[i]:.6f}\n"
                f"  Std        : {std[i]:.6f}\n"
                f"  Min        : {pixel_min[i]:.6f}\n"
                f"  Max        : {pixel_max[i]:.6f}\n"
                f"  Data Range : {data_range[i]:.6f}\n"
            )

    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "min": pixel_min.tolist(),
        "max": pixel_max.tolist(),
        "data_range": data_range.tolist(),
    }


if __name__ == "__main__":

    stats = {
        "tir_200m": compute_stats("tir_200m.npy"),
        "tir_100m": compute_stats("tir_100m_512.npy"),
        "rgb_100m": compute_stats("rgb_100m_512.npy"),
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w") as f:
        yaml.safe_dump(
            stats,
            f,
            sort_keys=False,
        )

    print("=" * 60)
    print(f"Statistics saved to:\n{OUTPUT_FILE}")
    print("=" * 60)