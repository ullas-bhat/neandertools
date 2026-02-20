"""Image loading, PNG rendering, and GIF assembly."""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import ZScaleInterval, ImageNormalize
from PIL import Image


def get_exposure(butler, visit_id, detector_id):
    """Load a visit_image exposure from Butler.

    Parameters
    ----------
    butler : lsst.daf.butler.Butler
        Initialized Butler instance.
    visit_id : int
        LSST visit ID.
    detector_id : int
        LSST detector ID.

    Returns
    -------
    exposure or None
        The loaded ExposureF, or None on failure.
    """
    try:
        return butler.get(
            "visit_image",
            dataId={"visit": int(visit_id), "detector": int(detector_id)},
        )
    except Exception as e:
        print(f"Skipping visit={visit_id} det={detector_id}: {e}")
        return None


def sigma_clipped_stats(arr, sigma=3.0, maxiters=5):
    """Compute sigma-clipped background and RMS of an array.

    Parameters
    ----------
    arr : np.ndarray
        Input pixel data.
    sigma : float
        Clipping threshold in standard deviations.
    maxiters : int
        Maximum number of clipping iterations.

    Returns
    -------
    bg : float
        Robust background estimate (clipped median).
    rms : float
        Robust noise estimate (1.4826 * clipped MAD).
    """
    values = np.asarray(arr, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0

    clipped = values
    for _ in range(maxiters):
        med = float(np.median(clipped))
        mad = float(np.median(np.abs(clipped - med)))
        rms = 1.4826 * mad
        if not np.isfinite(rms) or rms <= 0:
            break
        keep = np.abs(clipped - med) <= sigma * rms
        if keep.all() or keep.sum() == 0:
            break
        clipped = clipped[keep]

    bg = float(np.median(clipped))
    mad = float(np.median(np.abs(clipped - bg)))
    rms = 1.4826 * mad
    if not np.isfinite(rms) or rms <= 0:
        rms = float(np.std(clipped)) if clipped.size > 0 else 1.0
    if not np.isfinite(rms) or rms <= 0:
        rms = 1.0
    return bg, rms


def normalize_cutouts(cutout_arrays, match_background=True, match_noise=False):
    """Normalize a list of cutout arrays to a common background and scale.

    Parameters
    ----------
    cutout_arrays : list of np.ndarray
        Raw pixel arrays from cutouts.
    match_background : bool
        If ``True``, subtract per-cutout sigma-clipped background so all
        frames share a zero-centered background.
    match_noise : bool
        If ``True``, also divide by per-cutout RMS so all frames share a
        common noise level (SNR-like display).

    Returns
    -------
    normalized : list of np.ndarray
        Processed arrays.
    vmin : float
        Shared display lower bound (1st percentile of all pixels).
    vmax : float
        Shared display upper bound (99th percentile of all pixels).
    """
    processed = []
    for arr in cutout_arrays:
        arr = arr.astype(np.float32, copy=True)
        if match_background or match_noise:
            bg, rms = sigma_clipped_stats(arr)
            if match_background:
                arr = arr - bg
            if match_noise:
                arr = arr / max(rms, 1e-12)
        processed.append(arr)

    # Compute shared display scale across all frames
    all_finite = np.concatenate([a[np.isfinite(a)].ravel() for a in processed])
    if all_finite.size == 0:
        return processed, 0.0, 1.0

    vmin = float(np.percentile(all_finite, 1))
    vmax = float(np.percentile(all_finite, 99))
    if vmax <= vmin:
        vmax = vmin + 1e-12

    return processed, vmin, vmax


def cutout_to_png(cutout_exposure, output_path, title="", vmin=None, vmax=None, array_override=None):
    """Render an LSST ExposureF cutout to a PNG file.

    Parameters
    ----------
    cutout_exposure : lsst.afw.image.ExposureF
        Cutout image to render.
    output_path : str or Path
        Output PNG file path.
    title : str
        Plot title.
    vmin : float or None
        Fixed lower display bound. If ``None``, uses ZScale auto-stretch.
    vmax : float or None
        Fixed upper display bound. If ``None``, uses ZScale auto-stretch.
    array_override : np.ndarray or None
        If provided, render this array instead of ``cutout_exposure.image.array``.
        Used for background-subtracted / normalized data.
    """
    if array_override is not None:
        image_array = array_override
    else:
        image_array = cutout_exposure.image.array

    fig, ax = plt.subplots(figsize=(3, 3))

    if vmin is not None and vmax is not None:
        ax.imshow(image_array, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    else:
        norm = ImageNormalize(image_array, interval=ZScaleInterval())
        ax.imshow(image_array, origin="lower", cmap="gray", norm=norm)

    ax.set_title(title, fontsize=8)
    ax.axis("off")
    plt.savefig(output_path, dpi=100, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def cutouts_grid(
    cutouts,
    metadata=None,
    ncols=5,
    figsize_per_cell=(3.0, 3.0),
    match_background=True,
    match_noise=True,
    cmap="gray",
    output_path=None,
    dpi=100,
    show=True,
):
    """Display cutouts in a grid with matched background/noise.

    Parameters
    ----------
    cutouts : list
        List of LSST ExposureF cutout objects (with ``.image.array``).
    metadata : list of dict or None
        Per-cutout metadata dicts with ``"band"`` and ``"time"`` keys
        (as produced by ``AsteroidCutoutPipeline``). Used for titles.
        If ``None``, frames are numbered.
    ncols : int
        Number of columns in the grid.
    figsize_per_cell : tuple of float
        ``(width, height)`` per subplot cell in inches.
    match_background : bool
        If ``True``, subtract per-cutout background for a uniform zero level.
    match_noise : bool
        If ``True``, also divide by per-cutout RMS for uniform noise scale.
    cmap : str
        Matplotlib colormap name.
    output_path : str, Path, or None
        If provided, save the figure to this path.
    dpi : int
        Resolution for the saved figure.
    show : bool
        If ``True``, call ``plt.show()``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : np.ndarray of matplotlib.axes.Axes
    """
    import math

    n = len(cutouts)
    if n == 0:
        raise ValueError("No cutouts to display.")

    raw_arrays = [c.image.array for c in cutouts]
    norm_arrays, vmin, vmax = normalize_cutouts(
        raw_arrays,
        match_background=match_background,
        match_noise=match_noise,
    )

    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_cell[0] * ncols, figsize_per_cell[1] * nrows),
        squeeze=False,
    )

    for i in range(n):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        ax.imshow(
            norm_arrays[i], origin="lower", cmap=cmap,
            vmin=vmin, vmax=vmax, interpolation="nearest",
        )
        if metadata is not None and i < len(metadata):
            meta = metadata[i]
            title = f"{meta['band']}-band\n{meta['time'].utc.iso[:16]}"
        else:
            title = f"Frame {i + 1}"
        ax.set_title(title, fontsize=8)
        ax.axis("off")

    # Turn off unused cells
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].axis("off")

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(str(output_path), dpi=dpi, bbox_inches="tight")
        print(f"Grid saved to {output_path}")

    if show:
        plt.show()

    return fig, axes


def create_gif(png_dir, output_path, duration=500):
    """Create an animated GIF from a directory of PNG frames.

    Parameters
    ----------
    png_dir : str or Path
        Directory containing PNG frames, sorted by filename.
    output_path : str or Path
        Output GIF file path.
    duration : int
        Duration of each frame in milliseconds.

    Returns
    -------
    Path
        Path to the created GIF file.
    """
    png_files = sorted(glob.glob(os.path.join(str(png_dir), "*.png")))
    if not png_files:
        raise ValueError(f"No PNG files found in {png_dir}")

    frames = [Image.open(f) for f in png_files]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        str(output_path),
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
    )
    print(f"GIF saved to {output_path} ({len(frames)} frames)")
    return output_path
