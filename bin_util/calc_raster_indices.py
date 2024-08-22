import logging
from pathlib import Path

from cropclassification.util import openeo_util, raster_index_util


def main():
    logging.basicConfig(level=logging.INFO)

    index = "ndvi"
    save_as_byte = True
    index = "bsi"
    save_as_byte = False

    input_dir = Path("//dg3.be/alp/Datagis/satellite_periodic/BEFL/s2-agri")
    input_paths = input_dir.glob("*.tif")
    output_dir = Path(f"//dg3.be/alp/Datagis/satellite_periodic/BEFL/s2-{index}")

    for input_path in input_paths:
        # Set band descriptions in input file
        band_descriptions = {1: "B02", 2: "B03", 3: "B04", 4: "B08", 5: "B11", 6: "B12"}
        openeo_util.add_band_descriptions(
            input_path, band_descriptions=band_descriptions
        )

        # Prepare output name: replace band part with index name
        name_parts = input_path.stem.split("_")
        name_parts[3] = index
        output_path = output_dir / f"{'_'.join(name_parts)}.tif"
        # output_path = output_path.with_stem(f"{output_path.stem}_byte")
        # print(rioxarray.open_rasterio(output_path).to_dataset("band"))
        raster_index_util.calc_index(
            input_path, output_path, index=index, save_as_byte=save_as_byte, force=False
        )


if __name__ == "__main__":
    main()
