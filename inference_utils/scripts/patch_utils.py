from typing import List, Tuple

import numpy as np


def extract_patches(
    image: np.ndarray,
    patch_size: int,
    stride: int | None = None,
) -> tuple[List[np.ndarray], List[Tuple[int, int]]]:
    """
    Extract overlapping patches from an image.

    Parameters
    ----------
    image : np.ndarray
        Image in (C, H, W) format.

    patch_size : int
        Square patch size.

    stride : int, optional
        Distance between consecutive patches.
        Defaults to patch_size (non-overlapping).

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

    if stride is None:
        stride = patch_size

    _, height, width = image.shape

    patches = []
    locations = []

    # ---------------------------------------------------------
    # Compute start coordinates so the last patch always reaches
    # the image boundary.
    # ---------------------------------------------------------

    y_positions = list(
        range(
            0,
            max(height - patch_size + 1, 1),
            stride,
        )
    )

    if y_positions[-1] != height - patch_size:
        y_positions.append(height - patch_size)

    x_positions = list(
        range(
            0,
            max(width - patch_size + 1, 1),
            stride,
        )
    )

    if x_positions[-1] != width - patch_size:
        x_positions.append(width - patch_size)

    # ---------------------------------------------------------

    for y in y_positions:

        for x in x_positions:

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
    Reconstruct an image from overlapping patches using
    Hann-window weighted blending.

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

    channels = output_shape[0]
    patch_size = patches[0].shape[-1]

    # ---------------------------------------------------------
    # Accumulators
    # ---------------------------------------------------------

    reconstructed = np.zeros(
        output_shape,
        dtype=np.float32,
    )

    weights = np.zeros(
        output_shape,
        dtype=np.float32,
    )

    # ---------------------------------------------------------
    # 2D Hann Window
    # ---------------------------------------------------------

    hann_y = np.hanning(patch_size)
    hann_x = np.hanning(patch_size)

    window = np.outer(
        hann_y,
        hann_x,
    ).astype(np.float32)

    # Prevent exact zeros at the borders
    window = np.maximum(
        window,
        1e-6,
    )

    # Expand to (C, H, W)
    window = np.broadcast_to(
        window,
        (channels, patch_size, patch_size),
    )

    # ---------------------------------------------------------
    # Blend patches
    # ---------------------------------------------------------

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
        ] += patch * window

        weights[
            :,
            y:y + patch_size,
            x:x + patch_size,
        ] += window

    # ---------------------------------------------------------
    # Normalize weighted sum
    # ---------------------------------------------------------

    reconstructed /= np.maximum(
        weights,
        1e-6,
    )

    return reconstructed.astype(
        patches[0].dtype,
    )