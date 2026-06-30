from typing import List, Tuple

import numpy as np


def extract_patches(
    image: np.ndarray,
    patch_size: int,
) -> tuple[List[np.ndarray], List[Tuple[int, int]]]:
    """
    Extract non-overlapping patches from an image.

    Parameters
    ----------
    image : np.ndarray
        Image in (C, H, W) format.

    patch_size : int
        Square patch size.

    Returns
    -------
    patches : list[np.ndarray]
        List of image patches.

    locations : list[tuple[int, int]]
        Top-left (y, x) coordinate of each patch.
    """

    if image.ndim != 3:
        raise ValueError(
            f"Expected image shape (C, H, W), got {image.shape}"
        )

    _, height, width = image.shape

    patches = []
    locations = []

    for y in range(
        0,
        height - patch_size + 1,
        patch_size,
    ):

        for x in range(
            0,
            width - patch_size + 1,
            patch_size,
        ):

            patch = image[
                :,
                y:y + patch_size,
                x:x + patch_size,
            ]

            patches.append(patch)
            locations.append((y, x))

    return patches, locations


def stitch_patches(
    patches: List[np.ndarray],
    locations: List[Tuple[int, int]],
    output_shape: Tuple[int, int, int],
    scale_factor: int = 1,
) -> np.ndarray:
    """
    Reconstruct an image from non-overlapping patches.

    Parameters
    ----------
    patches : list[np.ndarray]
        Model output patches in (C, H, W).

    locations : list[tuple[int, int]]
        Original patch coordinates.

    output_shape : tuple
        (C, H, W) of reconstructed image.

    scale_factor : int
        Coordinate scaling factor.
        Example:
            SR -> 2
            Colorization -> 1
    """

    reconstructed = np.zeros(
        output_shape,
        dtype=patches[0].dtype,
    )

    patch_size = patches[0].shape[-1]

    for patch, (y, x) in zip(
        patches,
        locations,
    ):

        y *= scale_factor
        x *= scale_factor

        reconstructed[
            :,
            y:y + patch_size,
            x:x + patch_size,
        ] = patch

    return reconstructed