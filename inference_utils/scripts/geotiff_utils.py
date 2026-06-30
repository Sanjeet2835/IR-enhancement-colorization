from pathlib import Path
from typing import Union

import numpy as np
import rasterio
from affine import Affine


def read_geotiff(
    image_path: Union[str, Path],
) -> tuple[np.ndarray, dict]:
    """
    Read a GeoTIFF.

    Parameters
    ----------
    image_path : str or Path
        Path to input GeoTIFF.

    Returns
    -------
    image : np.ndarray
        Image in (C, H, W) format.

    metadata : dict
        Raster metadata required for saving.
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"File not found:\n{image_path}"
        )

    with rasterio.open(image_path) as src:

        image = src.read()

        metadata = src.meta.copy()

    return image, metadata


def write_geotiff(
    image: np.ndarray,
    metadata: dict,
    output_path: Union[str, Path],
    scale_factor: int = 1,
) -> None:
    """
    Save an image as a GeoTIFF.

    Parameters
    ----------
    image : np.ndarray
        Image in (C, H, W) format.

    metadata : dict
        Metadata copied from the source GeoTIFF.

    output_path : str or Path
        Destination GeoTIFF.

    scale_factor : int, default=1
        Spatial scaling factor relative to the input.
        Example:
            1 -> Colorization output (100 m -> 100 m)
            2 -> Super-resolution output (200 m -> 100 m)
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = metadata.copy()

    # ---------------------------------------------------------
    # Update image metadata
    # ---------------------------------------------------------

    metadata.update(
        count=image.shape[0],
        height=image.shape[1],
        width=image.shape[2],
        dtype=image.dtype,
    )

    # ---------------------------------------------------------
    # Update spatial resolution
    # ---------------------------------------------------------

    if scale_factor != 1:

        transform = metadata["transform"]

        metadata["transform"] = Affine(
            transform.a / scale_factor,
            transform.b,
            transform.c,
            transform.d,
            transform.e / scale_factor,
            transform.f,
        )

    # ---------------------------------------------------------
    # Write GeoTIFF
    # ---------------------------------------------------------

    with rasterio.open(
        output_path,
        "w",
        **metadata,
    ) as dst:

        dst.write(image)