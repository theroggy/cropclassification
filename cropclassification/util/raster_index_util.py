import logging
from pathlib import Path

import numpy as np
import rioxarray

from . import io_util

logger = logging.getLogger(__name__)


def calc_index(
    input_path: Path,
    output_path: Path,
    index: str,
    save_as_byte: bool = True,
    force: bool = False,
):
    if io_util.output_exists(output_path, remove_if_exists=force):
        return

    # Open the image file and calculate indexes
    with rioxarray.open_rasterio(input_path, cache=False, masked=True) as image_file:
        image = image_file.to_dataset("band")
        if "long_name" not in image.attrs:
            raise ValueError(
                "input file doesn't have band descriptions (image.attrs['long_name']) "
                f"specified: {input_path}"
            )
        image = image.rename({i + 1: n for i, n in enumerate(image.attrs["long_name"])})

        # Allow division by 0
        np.seterr(divide="ignore", invalid="ignore")

        scale_factor = None
        add_offset = None
        if index == "ndvi":
            red = image["B04"]
            nir = image["B08"]

            if save_as_byte:
                scale_factor = 0.004
                add_offset = -0.08

            # By default, dividing ints results in float64, but we only need float32
            index_data = np.divide((nir - red), (nir + red), dtype=np.float32)
            index_data.name = "NDVI"

        elif index == "bsi":
            # A Modified Bare Soil Index to Identify Bare Land Features during
            # Agricultural Fallow-Period in Southeast Asia Using Landsat 8.
            # Can Trong Nguyen, Amnat Chidthaisong, Phan Kieu Diem, Lian-Zhi Huo.
            # Land 2021, 10, 231. Page 3.
            # URL: https://www.mdpi.com/2073-445X/10/3/231/pdf
            blue = image["B02"]
            red = image["B04"]
            nir = image["B08"]
            swir2 = image["B11"]

            # By default, dividing ints results in float64, but we only need float32
            index_data = np.divide(
                ((swir2 + red) - (nir + blue)),
                ((swir2 + red) + (nir + blue)),
                dtype=np.float32,
            )
            index_data.name = "BSI"

        elif index == "dprvi":
            # "New" dual-pol radar vegetation index for Sentinel-1.
            # Paper:= https://www.sciencedirect.com/science/article/abs/pii/S0034425720303242
            # Implementation derived from one of by Dr. Dipankar Mandal:
            # https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-1/radar_vegetation_index/

            if save_as_byte:
                scale_factor = 0.004
                add_offset = 0

            vh = image["VH"]
            vv = image["VV"]

            # Calculate VH/VV ratio
            # By default, dividing ints results in float64, but we only need float32
            q = np.divide(vh, vv, dtype=np.float32)

            # Calculate ~degree of polarization: co-pol purity parameter m
            # (values are 0-1).
            # For low vegetation conditions, the co-pol backscatter will be high and the
            # cross-pol backscatter is low (i.e., q->0). As a consequence, one can
            # observe that, for bare field conditions m is high, and decreases gradually
            # with an increase in vegetation canopy density.
            # m = np.divide((1 - q), (1 + q), dtype=np.float32)

            # Further, the beta (normalized co-pol intensity parameter) can be expressed
            # as, beta = 1/(1+q). where 0.5<beta<1.
            # beta = np.divide(1, (1 + q), dtype=np.float32)

            # Now, the overall purity of the co-pol component can be obtained by
            # multiplying the co-pol purity parameter m and normalized co-pol intensity
            # parameter beta. Subsequently, by subtracting overall purity, we obtain a
            # quantitative measure of scattering randomness, as RVI4S1.
            # with 0<RVI4S1<1.0.

            # It quantifies impurity in the co-pol component of scattered wave. The
            # index also separates urban areas and bare soil from the vegetated terrain.
            # However, for very rough soil (likely after tillage) or water surface (high
            # windy condition), the DOP would be lower, which turns the RVI4S1 to be
            # quite higher than a smooth surface. Hence care should be taken with this
            # particular condition. For example, RVI4S1 = 0 for a pure or point target
            # scattering which corresponds to copol purity parameter m = 1, and beta =1.
            # On the other extreme case m = 0 and beta = 0.5 for a completely random
            # scattering. Therefore, RVI4S1=1 for a completely random scattering.
            # DpRVIc = 1-(m*beta)
            # Depolarization within the vegetation
            # value = (Math.sqrt(dop)) * ((4 * (vh)) / (vv + vh))

            # It can be written in fewer steps like this:
            N = np.multiply(q, (q + 3), dtype=np.float32)
            D = np.multiply((q + 1), (q + 1), dtype=np.float32)
            index_data = np.divide(N, D, dtype=np.float32)
            index_data.name = "DpRVI"

        else:
            raise ValueError(f"unsupported index type: {index}")

    if not save_as_byte:
        # Set nodata pixels to nan
        index_data.rio.write_nodata(np.nan, inplace=True)
        # Save as float. Use only 16 bit precision to save diskspace.
        index_data.rio.to_raster(
            output_path, nbits=16, tiled=True, compress="DEFLATE", predictor=3
        )
    else:
        # Scale factor specified, so rescale and save as Byte.
        if scale_factor is None or add_offset is None:
            raise ValueError(
                "to save as byte, scale_factor and add_offset should have a value "
                f"(scale_factor: {scale_factor}, add_offset: {add_offset})"
            )

        # Apply the scale factors + clip the data to maximum value
        index_data_scaled = (index_data - add_offset) / scale_factor
        index_data_scaled = index_data_scaled.clip(max=250)
        # Set nodata pixels from the original index (value=nan)
        # to 255 in the scaled index
        index_data_scaled = index_data_scaled.where(~np.isnan(index_data), other=255)
        # We don't need index_data anymore, so set to None to free memory
        index_data = None
        # Now clip low values to 0
        index_data_scaled = index_data_scaled.clip(min=0)

        # Now we can write the output
        index_data_scaled.attrs["scale_factor"] = scale_factor
        index_data_scaled.attrs["add_offset"] = add_offset
        index_data_scaled.rio.write_nodata(255, inplace=True)
        index_data_scaled = index_data_scaled.astype("B")
        index_data_scaled.rio.to_raster(
            output_path, tiled=True, compress="DEFLATE", predictor=2
        )


def remove(path: Path, missing_ok: bool = False):
    path.unlink(missing_ok=missing_ok)
    for file_path in path.parent.glob(f"{path.name}*"):
        file_path.unlink(missing_ok=missing_ok)
