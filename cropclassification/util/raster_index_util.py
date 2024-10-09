"""Module to calculate indexes from raster files."""

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import rioxarray
import xarray as xr

from . import io_util, lee_enhanced

# Disable some mypy errors for this file aI don't get them solved
# mypy: disable-error-code="union-attr, attr-defined, assignment"

logger = logging.getLogger(__name__)


def calc_index(
    input_path: Path,
    output_path: Path,
    index: str,
    pixel_type: str,
    process_options: dict = {},
    force: bool = False,
):
    """Calculate an index or derived file from a raster file.

    Args:
        input_path (Path): path to the input raster file.
        output_path (Path): path to the output file.
        index (str): index/derived file to calculate.
        pixel_type (str): output pixel type. One of "BYTE", "FLOAT16" or "FLOAT32".
        process_options (dict, optional): extra processing options to apply.
            Defaults to {}.
        force (bool, optional): True to overwrite an existing output file.
            Defaults to False.
    """
    if io_util.output_exists(output_path, remove_if_exists=force):
        return

    # Some processing options can be applied for multiple indexes
    tmp_dir = None
    if process_options is None:
        process_options = {}

    for process_option in process_options:
        # Remark: it's normal to encounter unknown processing options here as some are
        # applied later on.
        if process_option == "lee_enhanced":
            # Despeckle filter for sentinel 1 images
            filter = process_options[process_option]
            if filter is None or not isinstance(filter, dict):
                raise ValueError("process_option lee_enhanced should be a dict")

            filtersize = filter.get("filtersize")
            if filtersize is None or not isinstance(filtersize, int):
                raise ValueError(
                    "process_option lee_enhanced should have a filtersize int parameter"
                )

            tmp_dir = Path(tempfile.mkdtemp(prefix="calc_index_"))
            tmp_path = tmp_dir / f"{output_path.stem}_tmp_lee_{filtersize}.tif"
            lee_enhanced.lee_enhanced_file(
                input_path, tmp_path, filtersize=filtersize, nlooks=10.0, dfactor=10.0
            )
            input_path = tmp_path

    # Open the image file and calculate indexes
    #
    # Remarks:
    #   - use chunks=True to reduce memory usage.
    #   - lock=False is not faster, but uses more memory.
    try:
        image_file = rioxarray.open_rasterio(
            input_path, cache=False, masked=True, chunks=True
        )
        image = image_file.to_dataset("band")
        if "long_name" not in image.attrs:
            raise ValueError(
                "input file doesn't have band descriptions (image.attrs['long_name']) "
                f"specified: {input_path}"
            )
        image = image.rename({i + 1: n for i, n in enumerate(image.attrs["long_name"])})

        # Allow division by 0
        np.seterr(divide="ignore", invalid="ignore")

        scale_factor: float = None
        add_offset: float = None
        if index == "ndvi":
            red = image["B04"]
            nir = image["B08"]

            if pixel_type == "BYTE":
                scale_factor = 0.004
                add_offset = -0.08

            ndvi = (nir - red) / (nir + red)
            ndvi.name = index
            _save_index(ndvi, output_path, pixel_type, scale_factor, add_offset)

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

            bsi = ((swir2 + red) - (nir + blue)) / ((swir2 + red) + (nir + blue))
            bsi.name = index
            _save_index(bsi, output_path, pixel_type, scale_factor, add_offset)

        elif index == "dprvi":
            # "New" dual-pol radar vegetation index for Sentinel-1.
            # Paper:= https://www.sciencedirect.com/science/article/abs/pii/S0034425720303242
            # Implementation derived from one of by Dr. Dipankar Mandal:
            # https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-1/radar_vegetation_index/

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

            if pixel_type == "BYTE":
                scale_factor = 0.004
                add_offset = 0

            vv = image["VV"]
            vh = image["VH"]

            # It can be written in fewer steps like this:
            q = vh / vv
            dprvi = (q * (q + 3)) / ((q + 1) * (q + 1))

            dprvi.name = index
            _save_index(dprvi, output_path, pixel_type, scale_factor, add_offset)

        elif index == "rvi":
            if pixel_type == "BYTE":
                scale_factor = 0.004
                add_offset = 0

            vv = image["VV"]
            vh = image["VH"]

            # Calculate RVI ratio
            # Source: https://forum.step.esa.int/t/creating-radar-vegetation-index/12444/28
            # RVI=4*VH/(VV+VH)
            # var rvi = image.expression('sqrt(vv/(vv + vh))*(vv/vh)'
            rvi = (4 * vh) / (vv + vh)
            rvi.name = index
            _save_index(rvi, output_path, pixel_type, scale_factor, add_offset)

        elif index == "vvdvh":
            if pixel_type not in ("FLOAT16", "FLOAT32"):
                raise ValueError("vvdvh index can only be saved as FLOAT16 or FLOAT32")

            vv = image["VV"]
            vh = image["VH"]

            # Calculate VV versus VH ratio
            vvdvh = vv / vh
            vvdvh.name = index
            _save_index(vvdvh, output_path, pixel_type, scale_factor, add_offset)

        elif index == "sarrgb":
            log10 = process_options.get("log10", False)
            despeckled = "lee_enhanced" in process_options

            _calc_sar_rgb(
                image=image,
                output_path=output_path,
                log10=log10,
                despeckled=despeckled,
                pixel_type=pixel_type,
            )

        elif index == "sarfalse":
            """
            // SAR False Color Visualization
            // The script visualizes Earth surface in False Color from Sentinel-1 data.
            // Author: Annamaria Luongo (Twitter: @annamaria_84, https://www.linkedin.com/in/annamaria-luongo-RS )
            // License: CC BY 4.0 International

            var c1 = 10e-4;
            var c2 = 0.01;
            var c3 = 0.02;
            var c4 = 0.03;
            var c5 = 0.045;
            var c6 = 0.05;
            var c7 = 0.9;
            var c8 = 0.25;

            //Enhanced or non-enhanced option (set to "true" if you want enhanced)
            var enhanced = false;

            if (enhanced != true) {
                //Non-enhanced option
                var band1 = c4 + Math.log(c1 - Math.log(c6 / (c3 + 2 * VV)));
                var band2 = c6 + Math.exp(c8 * (Math.log(c2 + 2 * VV) + Math.log(c3 + 5 * VH)));
                var band3 = 1 - Math.log(c6 / (c5 - c7 * VV));
            }
            else {
                //Enhanced option
                var band1 = c4 + Math.log(c1 - Math.log(c6 / (c3 + 2.5 * VV)) + Math.log(c6 / (c3 + 1.5 * VH)));
                var band2 = c6 + Math.exp(c8 * (Math.log(c2 + 2 * VV) + Math.log(c3 + 7 * VH)));
                var band3 = 0.8 - Math.log(c6 / (c5 - c7 * VV));
            }
            """  # noqa: E501

        else:
            raise ValueError(f"unsupported index type: {index}")
    finally:
        image_file.close()

        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def _save_index(
    index_data: xr.DataArray,
    output_path: Path,
    pixel_type: str,
    scale_factor: Optional[float],
    add_offset: Optional[float],
):
    if pixel_type == "BYTE":
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

        try:
            index_data_scaled.rio.to_raster(
                output_path, tiled=True, windowed=True, compress="DEFLATE", predictor=2
            )
        except Exception as ex:
            remove(output_path, missing_ok=True)
            raise ex
    else:
        # Save as float
        # Set nodata pixels to nan
        index_data.rio.write_nodata(np.nan, inplace=True)

        kwargs = {}
        if pixel_type == "FLOAT16":
            # Use only 16 bit precision to save diskspace.
            kwargs["nbits"] = 16

        try:
            index_data.rio.to_raster(
                output_path,
                tiled=True,
                compress="DEFLATE",
                predictor=3,
                **kwargs,
            )
        except Exception as ex:
            remove(output_path, missing_ok=True)
            raise ex


def _calc_sar_rgb(
    image: xr.Dataset,
    output_path: Path,
    log10: bool,
    despeckled: bool,
    pixel_type: str,
):
    """Calculates and saves a SAR RGB image.

    Args:
        image (xr.Dataset): the input SAR image.
        output_path (Path): the output path to save the SAR RGB image.
        log10 (bool): True to convert the bands to dB.
        despeckled (bool): True if the input image is despeckled.
        pixel_type (str): One of ...
    """
    # Init some variables
    if pixel_type == "BYTE":
        if not log10:
            raise ValueError("pixel_type==BYTE is only implemented for log10=True")

        if despeckled:
            vvdb_min, vvdb_max = (-21, -5)
            vhdb_min, vhdb_max = (-23, -12)
            vvdvhdb_min, vvdvhdb_max = (3, 12)
        else:
            vvdb_min, vvdb_max = (-22, -5)
            vhdb_min, vhdb_max = (-23, -12)
            vvdvhdb_min, vvdvhdb_max = (2, 13)

    vv = image["VV"]
    vh = image["VH"]

    if not np.isnan(vv.rio.nodata) or not np.isnan(vh.rio.nodata):
        raise ValueError("VV and VH should have nodata values set to nan")

    # Convert to dB if needed + add VV/VH ratio band
    if log10:
        # Convert to dB + add VV/VH ratio band
        vv = 10 * np.log10(vv, dtype=np.float32)
        vv.name = "vvdb"
        vv.rio.write_nodata(np.nan, inplace=True)
        vh = 10 * np.log10(vh, dtype=np.float32)
        vh.name = "vhdb"
        vh.rio.write_nodata(np.nan, inplace=True)

        # Add VV/VH ratio band. In dB dividing becomes minus
        vvdvh = vv - vh
        vvdvh.name = "vvdvhdb"
        vvdvh.rio.write_nodata(np.nan, inplace=True)
    else:
        vv.name = "vv"
        vh.name = "vh"

        # Add VV/VH ratio band
        vvdvh = vv / vh
        vvdvh.name = "vvdvh"
        vvdvh.rio.write_nodata(np.nan, inplace=True)

    # Save the SAR RGB image
    if pixel_type.startswith("FLOAT"):
        # Save float version to file
        sar_rgb_db_float = xr.merge([vv, vh, vvdvh])  # type: ignore[list-item]

        kwargs = {}
        if pixel_type == "FLOAT16":
            # Use only 16 bit precision to save diskspace.
            kwargs["nbits"] = 16

        try:
            # Remarks:
            #   - specifying lock to enable parallel writing halves the memory usage,
            #     but it is a lot slower and results in double the file size.
            sar_rgb_db_float.rio.to_raster(
                output_path,
                tiled=True,
                compress="DEFLATE",
                predictor=3,
                **kwargs,
            )
        except Exception as ex:
            remove(output_path, missing_ok=True)
            raise ex

    elif pixel_type == "BYTE":
        # Scale the data to 0-255 so it is ready for visualisation
        #
        # Initial min_val and max_val values based on following page:
        # https://gis.stackexchange.com/questions/400726/creating-composite-rgb-images-from-sentinel-1-channels
        #
        # Then "optimized" based on the histogram of S1 mosaic of Belgium-Flanders of
        # 2023-09-04 -> 2023-09-10.

        # Scale VV (Red) to byte
        # Remarks:
        #   - lower VV values can be important for difference water vs bare soil:
        #       -> Based on an example: water (mean): -21 db, bare soil (mean): -15 db
        vvdb_scaled = (vv - vvdb_min) * 254 / (vvdb_max - vvdb_min)
        vvdb_scaled = vvdb_scaled.clip(0, 254)
        vvdb_scaled = vvdb_scaled.where(~np.isnan(image["VV"]), other=255)
        vvdb_scaled.rio.write_nodata(255, inplace=True)
        vvdb_scaled.name = "vvdb"

        # Scale VH (Green) to byte
        vhdb_scaled = (vh - vhdb_min) * 254 / (vhdb_max - vhdb_min)
        vhdb_scaled = vhdb_scaled.clip(0, 254)
        vhdb_scaled = vhdb_scaled.where(~np.isnan(image["VV"]), other=255)
        vhdb_scaled.rio.write_nodata(255, inplace=True)
        vhdb_scaled.name = "vhdb"

        # Scale VV/VH (Blue) to byte
        vvdvhdb_scaled = (vvdvh - vvdvhdb_min) * 254 / (vvdvhdb_max - vvdvhdb_min)
        vvdvhdb_scaled = vvdvhdb_scaled.clip(0, 254)
        vvdvhdb_scaled = vvdvhdb_scaled.where(~np.isnan(image["VV"]), other=255)
        vvdvhdb_scaled.rio.write_nodata(255, inplace=True)
        vvdvhdb_scaled.name = "vvdvhdb"

        # Save to file
        sar_rgb_db = xr.merge([vvdb_scaled, vhdb_scaled, vvdvhdb_scaled])  # type: ignore[list-item]

        try:
            sar_rgb_db.rio.to_raster(
                output_path, tiled=True, compress="DEFLATE", predictor=2, dtype="uint8"
            )
        except Exception as ex:
            remove(output_path, missing_ok=True)
            raise ex
    else:
        raise ValueError(f"unsupported pixel type: {pixel_type}")


def remove(path: Path, missing_ok: bool = False):
    """Remove the file and any related files.

    Args:
        path (Path): path to the file to remove.
        missing_ok (bool, optional): If True, no error is raised if the file doesn't
            exist. Defaults to False.
    """
    path.unlink(missing_ok=missing_ok)
    for file_path in path.parent.glob(f"{path.name}*"):
        file_path.unlink(missing_ok=missing_ok)
