import logging
from pathlib import Path
import numpy as np

import rioxarray

logger = logging.getLogger(__name__)


def calc_index(
    input_path: Path,
    output_path: Path,
    index: str,
    save_as_byte: bool = True,
    force: bool = False,
):
    if output_path.exists():
        if force:
            remove(output_path)
        else:
            logger.info(f"output_path exists already: {output_path}")
            return

    # Open the image file and calculate indexes
    with rioxarray.open_rasterio(input_path, cache=False) as image_file:
        image = image_file.to_dataset("band")
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

        else:
            raise ValueError(f"unsupported index type: {index}")

    if not save_as_byte:
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
        # Set nodata pixels from the original index (value 0) to 255 in the scaled index
        index_data_scaled = index_data_scaled.where(index_data != 0, other=255)
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
