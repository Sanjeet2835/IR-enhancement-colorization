from pathlib import Path
import argparse

from inference_utils.scripts.geotiff_utils import (
    read_geotiff,
    write_geotiff,
)

from inference_utils.scripts.patch_utils import (
    extract_patches,
    stitch_patches,
)

from inference_utils.scripts.sr_inference import (
    SRInference,
)

from inference_utils.scripts.colorization_inference import (
    ColorizationInference,
)


def parse_args():

    parser = argparse.ArgumentParser(
        description="Thermal Image Enhancement and Colorization"
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input thermal GeoTIFF.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save outputs.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda or cpu",
    )

    return parser.parse_args()

def main():

    args = parse_args()

    output_dir = Path(
        args.output_dir,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Read Input GeoTIFF
    # ---------------------------------------------------------

    tir_image, metadata = read_geotiff(
        args.input,
    )

    # ---------------------------------------------------------
    # Initialize Models
    # ---------------------------------------------------------

    sr_model = SRInference(
        device=args.device,
    )

    colorization_model = ColorizationInference(
        device=args.device,
    )

    # ---------------------------------------------------------
    # Super-Resolution
    # ---------------------------------------------------------

    sr_patches, sr_locations = extract_patches(
        image=tir_image,
        patch_size=256,
        stride=128,
    )
    
    sr_predictions = []

    # ---------------------------------------------------------
    # Super-Resolve Each Patch
    # ---------------------------------------------------------

    for i, patch in enumerate(sr_patches):

        sr_patch = sr_model.predict(patch)

        if i == 0:
            print("\n========== SR ==========")
            print("Input :", patch.min(), patch.max(), patch.mean())
            print("Output:", sr_patch.min(), sr_patch.max(), sr_patch.mean())

        sr_predictions.append(sr_patch)

    # ---------------------------------------------------------
    # Stitch Super-Resolved Image
    # ---------------------------------------------------------

    _, height, width = tir_image.shape

    sr_image = stitch_patches(
        patches=sr_predictions,
        locations=sr_locations,
        output_shape=(
            1,
            height * 2,
            width * 2,
        ),
        scale_factor=2,
    )
    
    # ---------------------------------------------------------
    # Save Super-Resolved TIR
    # ---------------------------------------------------------

    product_id = Path(
        args.input,
    ).parent.name

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    sr_output_dir = (
        output_dir
        / "model_outputs"
        / "tir_superresolved_100m"
    )

    color_output_dir = (
        output_dir
        / "model_outputs"
        / "colorized_tir_100m"
    )

    sr_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    color_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_geotiff(
        image=sr_image,
        metadata=metadata,
        output_path=sr_output_dir / f"{product_id}.tif",
        scale_factor=2,
    )
    
    
    # ---------------------------------------------------------
    # Colorization
    # ---------------------------------------------------------

    color_patches, color_locations = extract_patches(
        image=sr_image,
        patch_size=512,
    )

    rgb_predictions = []

    for i, patch in enumerate(color_patches):

        rgb_patch = colorization_model.predict(patch)

        if i == 0:
            print("\n====== COLORIZATION ======")
            print("Input :", patch.min(), patch.max(), patch.mean())
            print("Output:", rgb_patch.min(), rgb_patch.max(), rgb_patch.mean())

        rgb_predictions.append(rgb_patch)
        
        
    # ---------------------------------------------------------
    # Stitch Colorized Image
    # ---------------------------------------------------------

    _, sr_height, sr_width = sr_image.shape

    rgb_image = stitch_patches(
        patches=rgb_predictions,
        locations=color_locations,
        output_shape=(
            3,
            sr_height,
            sr_width,
        ),
        scale_factor=1,
    )
    
    # ---------------------------------------------------------
    # Save Colorized RGB
    # ---------------------------------------------------------

    color_output_dir = (
        output_dir
        / "model_outputs"
        / "colorized_tir_100m"
    )

    color_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_geotiff(
        image=rgb_image,
        metadata=metadata,
        output_path=color_output_dir / f"{product_id}.tif",
        scale_factor=2,
    )


if __name__ == "__main__":
    main()