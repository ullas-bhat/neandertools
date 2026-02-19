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


def cutout_to_png(cutout_exposure, output_path, title=""):
    """Render an LSST ExposureF cutout to a PNG file.

    Parameters
    ----------
    cutout_exposure : lsst.afw.image.ExposureF
        Cutout image to render.
    output_path : str or Path
        Output PNG file path.
    title : str
        Plot title.
    """
    image_array = cutout_exposure.image.array
    norm = ImageNormalize(image_array, interval=ZScaleInterval())

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(image_array, origin="lower", cmap="gray", norm=norm)
    ax.set_title(title, fontsize=8)
    ax.axis("off")
    plt.savefig(output_path, dpi=100, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


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
