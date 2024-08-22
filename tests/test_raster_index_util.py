import shutil
from contextlib import nullcontext

import numpy as np
import pytest
import rasterio
from osgeo import gdal

from cropclassification.util import raster_index_util, raster_util
from tests.test_helper import SampleData

gdal.UseExceptions()


def create_gdal_raster(
    fname,
    values,
    *,
    gt=None,
    gdal_type=None,
    nodata=None,
    scale=None,
    offset=None,
    band_descriptions=None,
):
    gdal = pytest.importorskip("osgeo.gdal")
    gdal_array = pytest.importorskip("osgeo.gdal_array")
    drv = gdal.GetDriverByName("GTiff")
    bands = 1 if len(values.shape) == 2 else values.shape[0]
    if gdal_type is None:
        gdal_type = gdal_array.NumericTypeCodeToGDALTypeCode(values.dtype)
    ds = drv.Create(
        str(fname),
        values.shape[-2],
        values.shape[-1],
        bands=bands,
        eType=gdal_type,
    )
    if gt is None:
        ds.SetGeoTransform((0.0, 1.0, 0.0, values.shape[-2], 0.0, -1.0))
    else:
        ds.SetGeoTransform(gt)
    if nodata:
        if type(nodata) in {list, tuple}:
            for i, v in enumerate(nodata):
                ds.GetRasterBand(i + 1).SetNoDataValue(v)
        else:
            ds.GetRasterBand(1).SetNoDataValue(nodata)
    if scale:
        for i in range(bands):
            ds.GetRasterBand(i + 1).SetScale(scale)
    if offset:
        for i in range(bands):
            ds.GetRasterBand(i + 1).SetOffset(offset)
    if len(values.shape) == 2:
        ds.WriteArray(values)
    else:
        for i in range(bands):
            rasterband = ds.GetRasterBand(i + 1)
            if band_descriptions:
                rasterband.SetDescription(band_descriptions[i])
            rasterband.WriteArray(values[i, :, :])
            # ds.GetRasterBand(i + 1).WriteArray(values[i, :, :])

    # if band_descriptions:
    #     for i, band in enumerate(band_descriptions):
    #         rasterband = ds.GetRasterBand(i + 1)
    #         rasterband.SetDescription(band)
    #         rasterband.WriteArray(values[i, :, :])


def make_rect(xmin, ymin, xmax, ymax, id=None, properties=None):
    f = {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax], [xmin, ymin]]
            ],
        },
    }
    if id is not None:
        f["id"] = id
    if properties is not None:
        f["properties"] = properties
    return f


@pytest.mark.parametrize("force", [True, False])
def test_calc_index_force(tmp_path, force):
    # Prepare test data
    input_path = SampleData.image_s2_mean_path
    output_path = tmp_path / f"{input_path.stem}_ndvi.tif"
    output_path.touch()

    # Test
    raster_index_util.calc_index(
        input_path=input_path, output_path=output_path, index="ndvi", force=force
    )

    if force:
        assert output_path.stat().st_size > 0
    else:
        assert output_path.stat().st_size == 0


def test_calc_index_invalid(tmp_path):
    # Prepare test data
    input_path = SampleData.image_s2_mean_path
    test_input_path = tmp_path / input_path.name
    shutil.copy(input_path, test_input_path)
    # Remove the band descriptions
    empty_band_descriptions = [None, None, None, None, None, None]
    raster_util.set_band_descriptions(test_input_path, empty_band_descriptions)
    output_path = tmp_path / f"{input_path.stem}_ndvi.tif"

    # Test
    with pytest.raises(ValueError, match="input file doesn't have band descriptions"):
        raster_index_util.calc_index(
            input_path=test_input_path, output_path=output_path, index="ndvi"
        )


@pytest.mark.parametrize("index", ["dprvi"])
@pytest.mark.parametrize("save_as_byte", [True, False])
def test_calc_index_s1(tmp_path, index, save_as_byte):
    # Prepare test data
    input_path = SampleData.image_s1_asc_path
    output_path = tmp_path / f"{input_path.stem}_{index}_{save_as_byte}.tif"

    # Prepare parameters
    assert not output_path.exists()
    raster_index_util.calc_index(
        input_path=input_path, output_path=output_path, index=index
    )
    assert output_path.exists()


@pytest.mark.parametrize(
    "index, save_as_byte", [("ndvi", True), ("ndvi", False), ("bsi", False)]
)
def test_calc_index_s2(tmp_path, index, save_as_byte):
    # Prepare test data
    input_path = SampleData.image_s2_mean_path
    output_path = tmp_path / f"{input_path.stem}_{index}_{save_as_byte}.tif"

    # Prepare parameters
    assert not output_path.exists()
    raster_index_util.calc_index(
        input_path=input_path,
        output_path=output_path,
        index=index,
        save_as_byte=save_as_byte,
    )
    assert output_path.exists()


@pytest.mark.parametrize(
    "index, save_as_byte, gdal_type, nodata, bands_descriptions, exp_dtype, exp_nodata, exp_error",  # noqa: E501
    [
        ("ndvi", True, gdal.GDT_UInt16, 32676, None, "uint8", 255, True),
        ("ndvi", True, gdal.GDT_Float32, np.nan, ["B04", "B08"], "uint8", 255, False),
        (
            "ndvi",
            False,
            gdal.GDT_UInt16,
            32676,
            ["B04", "B08"],
            "float32",
            np.nan,
            False,
        ),
        (
            "ndvi",
            False,
            gdal.GDT_Float32,
            np.nan,
            ["B04", "B08"],
            "float32",
            np.nan,
            False,
        ),
        ("dprvi", True, gdal.GDT_UInt16, 32676, ["VH", "VV"], "uint8", 255, False),
        ("dprvi", True, gdal.GDT_Float32, np.nan, ["VH", "VV"], "uint8", 255, False),
        (
            "dprvi",
            False,
            gdal.GDT_UInt16,
            32676,
            ["VH", "VV"],
            "float32",
            np.nan,
            False,
        ),
        (
            "dprvi",
            False,
            gdal.GDT_Float32,
            np.nan,
            ["VH", "VV"],
            "float32",
            np.nan,
            False,
        ),
    ],
)
def test_calc_index(
    tmp_path,
    index,
    save_as_byte,
    gdal_type,
    nodata,
    bands_descriptions,
    exp_dtype,
    exp_nodata,
    exp_error,
):
    input_path = tmp_path / "input.tif"
    output_path = tmp_path / "output.tif"

    raster_fname = str(input_path)
    raster_array = [
        [
            [nodata, nodata, nodata],
            [nodata, nodata, nodata],
            [nodata, nodata, nodata],
        ]
    ]
    if bands_descriptions:
        for _ in range(len(bands_descriptions) - 1):
            raster_array.append(
                [
                    [nodata, nodata, nodata],
                    [nodata, nodata, nodata],
                    [nodata, nodata, nodata],
                ]
            )

    create_gdal_raster(
        raster_fname,
        np.array(raster_array),
        gdal_type=gdal_type,
        nodata=nodata,
        band_descriptions=bands_descriptions,
    )

    if exp_error:
        handler = pytest.raises(
            ValueError, match="input file doesn't have band descriptions"
        )
    else:
        handler = nullcontext()

    with handler:
        raster_index_util.calc_index(
            input_path=input_path,
            output_path=output_path,
            index=index,
            save_as_byte=save_as_byte,
        )

        with rasterio.open(output_path, "r") as src:
            assert src.dtypes[0] == exp_dtype
            if np.isnan(exp_nodata):
                assert np.isnan(src.nodata)
            else:
                assert src.nodata == exp_nodata
