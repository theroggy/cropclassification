import logging
import os
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

from osgeo import gdal

# Suppress gdal warnings/errors
gdal.PushErrorHandler("CPLQuietErrorHandler")

import rasterio
import shapely.geometry as sh_geom

from cropclassification.util import io_util

# General init
logger = logging.getLogger(__name__)


class BandInfo:
    def __init__(
        self,
        path: str,
        relative_path: str,
        filename: str,
        bandindex: int,
        bounds: Optional[Tuple[float, float, float]] = None,
        affine=None,
        crs: Optional[str] = None,
        epsg: Optional[int] = None,
    ):
        self.path = path
        self.relative_path = relative_path
        self.filename = filename
        self.bandindex = bandindex
        self.bounds = bounds
        self.affine = affine
        self.crs = crs
        self.epsg = epsg


class ImageInfo:
    def __init__(
        self,
        imagetype: str,
        filetype: str,
        image_id: str,
        filename: str,
        footprint: dict,
        image_epsg: int,
        image_crs: str,
        image_bounds: Tuple[float, float, float],
        image_affine,
        bands: Dict[str, BandInfo],
        extra: dict,
    ):
        self.imagetype = imagetype
        self.filetype = filetype
        self.image_id = image_id
        self.filename = filename
        self.footprint = footprint
        self.image_epsg = image_epsg
        self.image_crs = image_crs
        self.image_bounds = image_bounds
        self.image_affine = image_affine
        self.bands = bands
        self.extra = extra


def get_image_data(
    image_path: Path,
    bounds: Tuple[float, float, float, float],
    bands: List[str],
    pixel_buffer: int = 0,
) -> dict:
    """
    Reads the data from the image.

    Adds a small buffer around the bounds asked to evade possible rounding issues.

    Returns a dict of the following structure:
    imagedata[band]['data']: the data read as numpy array
                   ['transform']: the Affine transform of the band read

    Args
        image_path: the path the image
        bounds: the bounds to be read, in coordinates in the projection of the image
        bands: list of bands to be read, eg. "VV", "VH",...
        pixel_buffer: number to pixels to take as buffer around the bounds provided in
            pixels
    """
    # Get info about the image
    logger.debug("get_image_data start")
    image_info = get_image_info(image_path)

    # Now get the data
    image_data = {}  # Dict for the transforms and the data per band
    if image_info.filetype in ("CARD", "SAFE", "TIF"):
        # Loop over bands and get data
        for band in bands:
            band_relative_path = image_info.bands[band].relative_path
            if band_relative_path is not None:
                image_band_path = image_path / band_relative_path
            else:
                image_band_path = image_path
            band_index = image_info.bands[band].bandindex
            logger.info(
                f"Read image data from {image_band_path}, with band_index: {band_index}"
            )
            image_data[band] = {}
            with rasterio.open(str(image_band_path)) as src:
                # Determine the window we need to read from the image:
                window_to_read = projected_bounds_to_window(
                    bounds, src.transform, src.width, src.height, pixel_buffer
                )
                image_data[band][
                    "transform"
                ] = rasterio.windows.transform(  # type: ignore
                    window_to_read, src.transform
                )
                image_data[band]["nodata"] = src.nodata
                # Read!
                # Remark: bandindex in rasterio is 1-based instead of 0-based -> +1
                logger.debug(f"Read image data from {image_band_path}")
                image_data[band]["data"] = src.read(
                    band_index + 1, window=window_to_read
                )
                logger.debug("Image data read")
    else:
        message = f"Format currently not supported: {image_path}"
        logger.error(message)
        raise NotImplementedError(message)

    logger.debug("get_image_data ready")

    return image_data


def get_image_info(image_path: Path) -> ImageInfo:
    """
    Returns basic information about an image.

    Args:
        image_path (Path): the path to the image.

    Raises:
        NotImplementedError: image is of an unsupported image type.

    Returns:
        ImageInfo: basic information about the image
    """

    image_info = {}

    # Specific code per image type
    suffix_upper = image_path.suffix.upper()
    if suffix_upper == ".CARD":
        image_info = _get_image_info_card(image_path)
    elif suffix_upper == ".SAFE":
        # This is a level 2 sentinel 2 file
        image_info = _get_image_info_safe(image_path)
    elif suffix_upper == ".TIF":
        image_info = _get_image_info_tif(image_path)
    else:
        message = f"Not a supported image format: {image_path}"
        logger.error(message)
        raise NotImplementedError(message)

    return image_info


def projected_bounds_to_window(
    projected_bounds,
    image_transform,
    image_pixel_width: int,
    image_pixel_height: int,
    pixel_buffer: int = 0,
):
    """
    Returns a rasterio.windows.Window to be used in rasterio to read the part of the
    image specified.

    Args
        projected_bounds: bounds to created the window from, in projected coordinates
        image_transform: Affine transform of the image you want to create the pixel
            window for
        image_pixel_width: total width of the image you want to create the pixel window
            for, in pixels
        image_pixel_height: total height of the image you want to create the pixel
            window for, in pixels
        pixel_buffer: number to pixels to take as buffer around the bounds provided in
            pixels
    """
    # Take bounds of the features + convert to image pixels
    xmin, ymin, xmax, ymax = projected_bounds
    window_to_read_raw = rasterio.windows.from_bounds(  # type: ignore
        xmin, ymin, xmax, ymax, image_transform
    )

    # Round
    window_to_read = window_to_read_raw.round_offsets().round_lengths()

    # Now some general math on window properties, but as they are readonly properties,
    # work on copy
    col_off, row_off, width, height = window_to_read.flatten()
    # Add buffer of 1 pixel extra around
    col_off -= pixel_buffer
    row_off -= pixel_buffer
    width += 2 * pixel_buffer
    height += 2 * pixel_buffer

    # Make sure the offsets aren't negative, as the pixels that are 'read' there acually
    # get some value instead of eg. nan...!
    if col_off < 0:
        width -= abs(col_off)
        col_off = 0
    if row_off < 0:
        height -= abs(row_off)
        row_off = 0

    # Make sure there won't be extra pixels to the top and right that will be read
    if (col_off + width) > image_pixel_width:
        width = image_pixel_width - col_off
    if (row_off + height) > image_pixel_height:
        height = image_pixel_height - row_off

    # Ready... prepare to return...
    window_to_read = rasterio.windows.Window(  # type: ignore
        col_off, row_off, width, height
    )

    # Debug info
    """
    bounds_to_read = rasterio.windows.bounds(window_to_read, image_transform)
    logger.debug(f"projected_bounds: {projected_bounds}, "
        f"window_to_read_raw: {window_to_read_raw}, window_to_read: {window_to_read}, "
        f"image_pixel_width: {image_pixel_width}, "
        f"image_pixel_height: {image_pixel_height}, "
        f"file transform: {image_transform}, bounds_to_read: {bounds_to_read}"
    )
    """

    return window_to_read


def prepare_image(image_path: Path, temp_dir: Path) -> Path:
    """
    Prepares the input image for usages.

    In case of a zip file, the file is unzipped to the temp dir specified.

    Returns the path to the prepared file/directory.
    """

    # If the input path is not a zip file, don't make local copy + just return path
    if image_path.suffix.lower() != ".zip":
        return image_path
    else:
        # It is a zip file, so it needs to be unzipped first...
        # Create destination file path
        image_basename_withzipext = os.path.basename(image_path)
        image_basename = os.path.splitext(image_basename_withzipext)[0]
        image_unzipped_path = temp_dir / image_basename

        image_unzipped_path_busy = Path(f"{image_unzipped_path}_busy")
        # If the input is a zip file, unzip file to temp local location if it doesn't
        # exist yet. If the file doesn't exist yet in right projection, read original
        # input file to reproject/write to new file with correct epsg
        if not (image_unzipped_path_busy.exists() or image_unzipped_path.exists()):
            # Create temp dir if it doesn't exist yet
            os.makedirs(temp_dir, exist_ok=True)

            # Create lock file in an atomic way, so we are sure we are the only process
            # working on it. If function returns true, there isn't any other
            # thread/process already working on it
            if io_util.create_file_atomic(image_unzipped_path_busy):
                try:
                    logger.info(
                        f"Unzip image {image_path} to local location {temp_dir}"
                    )

                    # Create the dest dir where the file will be unzipped to + unzip!
                    if not image_unzipped_path.exists():
                        import zipfile

                        with zipfile.ZipFile(image_path, "r") as zippedfile:
                            zippedfile.extractall(temp_dir)
                finally:
                    # Remove lock file when we are ready
                    os.remove(image_unzipped_path_busy)

        # If a "busy file" still exists, the file isn't ready yet, but another process
        # is working on it, so wait till it disappears
        while image_unzipped_path_busy.exists():
            time.sleep(1)

        # Now we are ready to return the path...
        return image_unzipped_path


def _get_image_info_card(image_path: Path) -> ImageInfo:
    # This is a sentinel 1 image (GRD or coherence)
    # First extract and fill out some basic info
    imagetype = image_path.stem.split("_")[0]
    filetype = "CARD"
    image_id = image_path.stem

    # Read info from the metadata file
    metadata_xml_path = image_path / "metadata.xml"
    metadata = ET.parse(str(metadata_xml_path))
    metadata_root = metadata.getroot()

    logger.debug(f"Parse metadata info from {metadata_xml_path}")

    try:
        # Get the filename
        filename = metadata_root.find("productFilename").text  # type: ignore

        # Get the footprint
        footprint = {}
        footprint_elem = metadata_root.find("footprint")
        assert footprint_elem is not None
        poly = footprint_elem.find("{http://www.opengis.net/gml}Polygon")
        assert poly is not None
        footprint["srsname"] = poly.attrib.get("srsName")
        linear_ring = []
        # for coord in poly.findall(
        #   "{http://www.opengis.net/gml}outerBoundaryIs/"
        #   "{http://www.opengis.net/gml}LinearRing/"
        #   "{http://www.opengis.net/gml}coord"
        # ):
        #    linear_ring.append(
        #        (float(coord.findtext(
        #            "{http://www.opengis.net/gml}X"
        #        )),
        #        float(coord.findtext("{http://www.opengis.net/gml}Y")))
        #    )
        coord_str = poly.find(
            "{http://www.opengis.net/gml}outerBoundaryIs/"
            "{http://www.opengis.net/gml}LinearRing/"
            "{http://www.opengis.net/gml}coordinates"
        )
        assert coord_str is not None and coord_str.text is not None
        logger.debug(f"coord_str: {coord_str}, coord_str.text: {coord_str.text}")
        coord_list = coord_str.text.split(" ")
        for coord in coord_list:
            # Watch out, latitude (~y) is first, than longitude (~x)
            # TODO: add check if the projection is in degrees (latlon) or
            # coordinates (xy) instead of hardcoded latlon
            y, x = coord.split(",")
            linear_ring.append((float(x), float(y)))
        footprint["shape"] = sh_geom.polygon.Polygon(linear_ring)

        # get epsg
        epsg = metadata_root.find("imageProjection/EPSG")
        assert epsg is not None and epsg.text is not None
        image_epsg = float(epsg.text)
        image_crs = f"EPSG:{epsg.text}"
    except Exception as ex:
        raise Exception(f"Exception extracting info from {metadata_xml_path}") from ex

    # Read info from the manifest.safe file
    manifest_xml_searchstring = "*_manifest.safe"
    manifest_xml_paths = list(image_path.glob(manifest_xml_searchstring))

    # The number of .safe indicates whether it is a GRD or a Coherence image
    extra = {}
    nb_safefiles = len(manifest_xml_paths)
    if nb_safefiles == 1:
        # Now parse the .safe file
        manifest_xml_path = manifest_xml_paths[0]

        try:
            manifest = ET.parse(str(manifest_xml_path))
            manifest_root = manifest.getroot()

            # Define namespaces...
            ns = {
                "safe": "http://www.esa.int/safe/sentinel-1.0",
                "s1": "http://www.esa.int/safe/sentinel-1.0/sentinel-1",
                "s1sarl1": (
                    "http://www.esa.int/safe/sentinel-1.0/sentinel-1/sar/level-1"
                ),
            }

            logger.debug(f"Parse manifest info from {metadata_xml_path}")
            extra["transmitter_receiver_polarisation"] = []
            for polarisation in manifest_root.findall(
                "metadataSection/metadataObject/metadataWrap/xmlData/"
                "s1sarl1:standAloneProductInformation/"
                "s1sarl1:transmitterReceiverPolarisation",
                ns,
            ):
                extra["transmitter_receiver_polarisation"].append(polarisation.text)
            extra["productTimelinessCategory"] = manifest_root.find(
                "metadataSection/metadataObject/metadataWrap/xmlData/"
                "s1sarl1:standAloneProductInformation/"
                "s1sarl1:productTimelinessCategory",
                ns,
            ).text  # type: ignore

            extra["instrument_mode"] = manifest_root.find(
                "metadataSection/metadataObject/metadataWrap/xmlData/"
                "safe:platform/safe:instrument/safe:extension/"
                "s1sarl1:instrumentMode/s1sarl1:mode",
                ns,
            ).text  # type: ignore
            extra["orbit_properties_pass"] = manifest_root.find(
                "metadataSection/metadataObject/metadataWrap/xmlData/"
                "safe:orbitReference/safe:extension/s1:orbitProperties/s1:pass",
                ns,
            ).text  # type: ignore
            extra["acquisition_date"] = manifest_root.find(
                "metadataSection/metadataObject/metadataWrap/xmlData/"
                "safe:acquisitionPeriod/safe:startTime",
                ns,
            ).text  # type: ignore

            # Now have a look in the files themselves to get band info,...
            # TODO: probably cleaner/easier to read from metadata files...
            image_basename_noext_nodate = image_path.stem[
                : image_path.stem.rfind("_", 0)
            ]
            image_datadirname = f"{image_basename_noext_nodate}.data"
            image_datadir = image_path / image_datadirname
            band_paths = list(image_datadir.glob("*.img"))

            # If no files were found, check if the images are in .tif format!
            if len(band_paths) == 0:
                tif_path = image_path / "Gamma0_VH.tif"
                if tif_path.exists():
                    band_paths.append(tif_path)
                tif_path = image_path / "Gamma0_VV.tif"
                if tif_path.exists():
                    band_paths.append(tif_path)

            # If no files were found, error
            if len(band_paths) == 0:
                message = f"No image files found in {image_datadir} or {image_path}!"
                logger.error(message)
                raise Exception(message)

            bands = {}
            for i, band_path in enumerate(band_paths):
                # Extract bound,... info from the first file only
                # (they are all the same)
                if i == 0:
                    # logger.debug(f"Read image metadata from {band_path}")
                    with rasterio.open(str(band_path)) as src:
                        image_bounds = src.bounds
                        image_affine = src.transform
                        if src.crs is not None:
                            image_crs = src.crs.to_string()
                            image_epsg = src.crs.to_epsg()
                    # logger.debug(f"Image metadata read: {image_info}")
                if band_path.stem == "Gamma0_VH":
                    band = "VH"
                elif band_path.stem == "Gamma0_VV":
                    band = "VV"
                else:
                    raise NotImplementedError(f"Filename not supported: {band_path}")

                # Add specific info about the band
                bands[band] = BandInfo(
                    path=str(band_path),
                    relative_path=str(band_path).replace(
                        str(image_path) + os.pathsep, ""
                    ),
                    filename=band_path.name,
                    bandindex=0,
                )

        except Exception as ex:
            raise Exception(
                f"Exception extracting info from {manifest_xml_path}"
            ) from ex

    elif nb_safefiles == 2 or nb_safefiles == 3:
        # 2 safe files -> coherence
        # Now parse the first .safe file
        # TODO: maybe check if the info in all safe files  are the same or?.?
        manifest_xml_path = manifest_xml_paths[0]

        try:
            manifest = ET.parse(str(manifest_xml_path))
            manifest_root = manifest.getroot()

            # Define namespaces...
            ns = {
                "safe": "http://www.esa.int/safe/sentinel-1.0",
                "s1": "http://www.esa.int/safe/sentinel-1.0/sentinel-1",
                "s1sarl1": (
                    "http://www.esa.int/safe/sentinel-1.0/sentinel-1/sar/level-1"
                ),
            }

            # logger.debug(f"Parse manifest info from {metadata_xml_path}")
            extra["transmitter_receiver_polarisation"] = []
            for polarisation in manifest_root.findall(
                "metadataSection/metadataObject/metadataWrap/xmlData/"
                "s1sarl1:standAloneProductInformation/"
                "s1sarl1:transmitterReceiverPolarisation",
                ns,
            ):
                extra["transmitter_receiver_polarisation"].append(polarisation.text)
            extra["productTimelinessCategory"] = manifest_root.find(
                "metadataSection/metadataObject/metadataWrap/xmlData/"
                "s1sarl1:standAloneProductInformation/"
                "s1sarl1:productTimelinessCategory",
                ns,
            ).text  # type: ignore
            extra["instrument_mode"] = manifest_root.find(
                "metadataSection/metadataObject/metadataWrap/xmlData/"
                "safe:platform/safe:instrument/safe:extension/"
                "s1sarl1:instrumentMode/s1sarl1:mode",
                ns,
            ).text  # type: ignore
            extra["orbit_properties_pass"] = manifest_root.find(
                "metadataSection/metadataObject/metadataWrap/xmlData/"
                "safe:orbitReference/safe:extension/s1:orbitProperties/s1:pass",
                ns,
            ).text  # type: ignore
            extra["acquisition_date"] = manifest_root.find(
                "metadataSection/metadataObject/metadataWrap/xmlData/"
                "safe:acquisitionPeriod/safe:startTime",
                ns,
            ).text  # type: ignore

            # Now have a look in the files themselves to get band info,...
            # TODO: probably cleaner/easier to read from metadata files...
            # For coherence filename, remove extra .LCO1 extension + date
            # image_datafilebasename, _ = os.path.splitext(image_basename_noext)
            # image_datafilebasename = image_datafilebasename[
            #     :image_datafilebasename.rindex('_')
            # ]

            metadata = ET.parse(str(image_path / "metadata.xml"))
            metadata_root = metadata.getroot()

            image_basename_noext_nodate = image_path.stem[
                : image_path.stem.rfind("_", 0)
            ]
            image_datafilename = f"{image_basename_noext_nodate}_byte.tif"
            image_datapath = image_path / image_datafilename

            # logger.debug(f"Read image metadata from {image_datapath}")
            with rasterio.open(str(image_datapath)) as src:
                image_bounds = tuple(src.bounds)
                image_affine = src.transform
                if src.crs is not None:
                    image_crs = src.crs.to_string()
                    image_epsg = src.crs.to_epsg()

            # Add specific info about the bands
            bands = {}
            bands["VH"] = BandInfo(
                path=str(image_datapath),
                relative_path=image_datafilename,
                filename=image_datafilename,
                bandindex=0,
            )
            bands["VV"] = BandInfo(
                path=str(image_datapath),
                relative_path=image_datafilename,
                filename=image_datafilename,
                bandindex=1,
            )

        except Exception as ex:
            raise Exception(
                f"Exception extracting info from {manifest_xml_path}"
            ) from ex
    else:
        message = (
            f"Error: found {nb_safefiles} .safe files doing glob in {image_path} "
            f"with {manifest_xml_searchstring}"
        )
        logger.error(message)
        raise Exception(message)

    return ImageInfo(
        imagetype=imagetype,
        filetype=filetype,
        image_id=image_id,
        filename=filename,
        footprint=footprint,
        image_epsg=image_epsg,
        image_crs=image_crs,
        image_bounds=image_bounds,
        image_affine=image_affine,
        bands=bands,
        extra=extra,
    )


def _get_image_info_safe(image_path: Path) -> ImageInfo:
    # First extract and fill out some basic info
    imagetype = image_path.stem.split("_")[0]
    filetype = "SAFE"
    image_id = image_path.stem
    extra = {}

    # Read info from the manifest.safe file
    metadata_xml_path = image_path / "MTD_MSIL2A.xml"
    try:
        metadata = ET.parse(str(metadata_xml_path))
        metadata_root = metadata.getroot()

        # Define namespaces...
        # xsi:schemaLocation=
        #     "https://psd-14.sentinel2.eo.esa.int/PSD/User_Product_Level-2A.xsd"
        ns = {
            "n1": ("https://psd-14.sentinel2.eo.esa.int/PSD/User_Product_Level-2A.xsd"),
            "xsi": "http://www.w3.org/2001/XMLSchema-instance",
        }

        # logger.debug(f"Parse metadata info from {metadata_xml_path}")
        extra["Cloud_Coverage_Assessment"] = float(
            metadata_root.find(
                "n1:Quality_Indicators_Info/Cloud_Coverage_Assessment", ns
            ).text  # type: ignore
        )
        extra["acquisition_date"] = metadata_root.find(
            "n1:General_Info/Product_Info/PRODUCT_START_TIME", ns
        ).text  # type: ignore

    except Exception as ex:
        raise Exception(f"Exception extracting info from {metadata_xml_path}") from ex

    # Now have a look in the files themselves to get band info,...
    image_datadir = image_path / "GRANULE"
    band_paths = list(image_datadir.rglob("*.jp2"))

    metadata_path = list(image_datadir.rglob("MTD_TL.xml"))[0]
    try:
        # read epsg
        metadata = ET.parse(str(metadata_path))
        metadata_root = metadata.getroot()

        ns = {
            "n1": (
                "https://psd-14.sentinel2.eo.esa.int/PSD/"
                "S2_PDI_Level-2A_Tile_Metadata.xsd"
            ),
            "xsi": "http://www.w3.org/2001/XMLSchema-instance",
        }

        image_crs = metadata_root.find(
            "n1:Geometric_Info/Tile_Geocoding/HORIZONTAL_CS_CODE", ns
        ).text  # type: ignore
        image_epsg = int(image_crs.replace("EPSG:", ""))  # type: ignore
    except Exception as ex:
        raise Exception(f"Exception extracting info from {metadata_path}") from ex

    # If no files were found, error!
    if len(band_paths) == 0:
        message = f"No image files found in {image_datadir}!"
        logger.error(message)
        raise Exception(message)

    bands = {}
    for i, band_path in enumerate(band_paths):
        band_filename = os.path.basename(band_path)
        band_filename_noext, _ = os.path.splitext(band_filename)
        band_filename_noext_split = band_filename_noext.split("_")

        if len(band_filename_noext_split) == 4:
            # IMG_DATA files
            band = f"{band_filename_noext_split[2]}-{band_filename_noext_split[3]}"
        elif len(band_filename_noext_split) == 5:
            # IMG_DATA files
            band = f"{band_filename_noext_split[3]}-{band_filename_noext_split[4]}"
        elif len(band_filename_noext_split) == 3:
            # QI_DATA files
            band = band_filename_noext
        else:
            message = f"Filename of band doesn't have supported format: {band_path}"
            logger.error(message)
            raise Exception(message)

        # Extract bound,... info for band
        logger.debug(f"Read image metadata from {band_path}")
        with rasterio.open(str(band_path)) as src:
            band_bounds = tuple(src.bounds)
            band_affine = src.transform
            if src.crs is None:
                band_crs = image_crs
                band_epsg = image_epsg
            else:
                band_crs = src.crs.to_string()
                band_epsg = src.crs.to_epsg()

        # Store the crs also on image level + check if all bands have the same crs
        if i == 0:
            image_bounds = band_bounds
        else:
            if image_crs != band_crs or image_epsg != band_epsg:
                message = f"Not all bands have the same crs for {image_path}"
                logger.error(message)
                raise Exception(message)

        bands[band] = BandInfo(
            path=str(band_path),
            relative_path=str(band_path).replace(str(image_path), "")[1:],
            filename=band_filename,
            bandindex=0,
            bounds=band_bounds,
            affine=band_affine,
            crs=band_crs,
            epsg=band_epsg,
        )

    return ImageInfo(
        imagetype=imagetype,
        filetype=filetype,
        image_id=image_id,
        filename=image_path.name,
        footprint=None,
        image_epsg=image_epsg,
        image_crs=image_crs,
        image_bounds=image_bounds,
        image_affine=None,
        bands=bands,
        extra=extra,
    )


def _get_image_info_tif(image_path: Path) -> ImageInfo:
    with rasterio.open(image_path) as src:
        image_bounds = tuple(src.bounds)
        image_affine = src.transform
        if src.crs is not None:
            image_crs = src.crs.to_string()
            image_epsg = src.crs.to_epsg()

        # Add all bands
        bands = {}
        for band_idx in src.indexes:
            band_tags = src.tags(band_idx)
            band_name = band_tags.get("DESCRIPTION", None)
            band_name = band_name if band_name is not None else str(band_idx)
            bands[band_name] = BandInfo(
                path=str(image_path),
                relative_path=None,
                filename=image_path.name,
                bandindex=band_idx - 1,
            )

    footprint = {}
    footprint["shape"] = None
    footprint["crs"] = None
    footprint["epsg"] = None

    extra = {}
    imagetype = image_path.stem.split("_")[0].upper()
    if imagetype in ("S1-ASC", "S1-DESC"):
        orbit = imagetype.split("-")[1]
        if orbit == "ASC":
            extra["orbit_properties_pass"] = "ASCENDING"
        elif orbit == "DESC":
            extra["orbit_properties_pass"] = "DESCENDING"
    return ImageInfo(
        imagetype=imagetype,
        filetype="TIF",
        image_id=image_path.stem,
        filename=image_path.name,
        footprint=footprint,
        image_epsg=image_epsg,
        image_crs=image_crs,
        image_bounds=image_bounds,
        image_affine=image_affine,
        bands=bands,
        extra=extra,
    )
