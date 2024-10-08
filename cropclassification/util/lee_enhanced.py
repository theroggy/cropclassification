"""Implementation of the Enhanced Lee Filter for speckle reduction in SAR images.

References:
  - https://stackoverflow.com/questions/4959171/improving-memory-usage-in-an-array-wide-filter-to-avoid-block-processing
  - https://catalyst.earth/catalyst-system-files/help/concepts/orthoengine_c/Chapter_825.html
"""

import warnings

import numexpr as ne
import numpy as np
import rasterio as rio
import scipy.ndimage
import scipy.signal

from . import raster_util


def _moving_average(image, size):
    Im = np.empty(image.shape, dtype=np.float32)
    # scipy.ndimage.filters.uniform_filter(image, filtsize, output=Im)
    # scipy.ndimage.generic_filter(image, function=np.nanmean, size=size, output=Im)
    Im = _filter_nanmean(image, size=size)
    return Im


def _moving_stddev(image, size):
    Im = np.empty(image.shape, dtype=np.float32)
    # scipy.ndimage.filters.uniform_filter(image, filtersize, output=Im)
    # scipy.ndimage.generic_filter(image, function=np.nanmean, size=size, output=Im)
    Im = _filter_nanmean(image, size=size)
    Im = ne.evaluate("((image-Im) ** 2)")
    # scipy.ndimage.filters.uniform_filter(Im, filtersize, output=Im)
    # scipy.ndimage.generic_filter(Im, function=np.nanmean, size=size, output=Im)
    Im = _filter_nanmean(Im, size=size)
    return ne.evaluate("sqrt(Im)")


def _filter_nanmean(image, size):
    kernel = np.ones((size, size))
    kernel[1, 1] = 0

    neighbor_sum = scipy.signal.convolve2d(
        image, kernel, mode="same", boundary="fill", fillvalue=0
    )

    num_neighbor = scipy.signal.convolve2d(
        np.ones(image.shape), kernel, mode="same", boundary="fill", fillvalue=0
    )

    return neighbor_sum / num_neighbor


def lee_enhanced(
    image, filtersize: int = 5, nlooks: float = 10.0, dfactor: float = 10.0
):
    # Implementation based on PCI Geomatimagea's FELEE function documentation
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning, message="Mean of empty slice"
        )
        Ci = _moving_stddev(image, filtersize)
        Im = _moving_average(image, filtersize)

    Ci /= Im

    Cu = np.sqrt(1 / nlooks).astype(np.float32)  # noqa: F841
    Cmax = np.sqrt(1 + (2 * nlooks)).astype(np.float32)  # noqa: F841

    W = ne.evaluate("exp(-dfactor * (Ci - Cu) / (Cmax - Ci))")
    If = ne.evaluate("Im * W + image * (1 - W)")
    del W

    out = ne.evaluate("where(Ci <= Cu, Im, If)")
    del Im
    del If

    out = ne.evaluate("where(Ci >= Cmax, image, out)")
    return out


def lee_enhanced_file(
    input_path,
    output_path,
    filtersize: int = 5,
    nlooks: float = 10.0,
    dfactor: float = 10.0,
    force: bool = False,
):
    if output_path.exists():
        if force:
            output_path.unlink()
        else:
            return

    with rio.open(input_path) as input:
        profile = input.profile
        band1 = input.read(1)
        band1_lee = lee_enhanced(
            band1, filtersize=filtersize, nlooks=nlooks, dfactor=dfactor
        )
        band2 = input.read(2)
        band2_lee = lee_enhanced(
            band2, filtersize=filtersize, nlooks=nlooks, dfactor=dfactor
        )
    band_descriptions = raster_util.get_band_descriptions(input_path)

    with rio.open(output_path, "w", **profile) as dst:
        dst.write(band1_lee, 1)
        dst.write(band2_lee, 2)
    raster_util.set_band_descriptions(output_path, band_descriptions.keys())
