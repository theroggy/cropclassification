from datetime import datetime
import logging
from pathlib import Path
import time
from typing import List

import geofileops as gfo
import geopandas as gpd

from cropclassification.util import io_util

logger = logging.getLogger(__name__)


def _load_features_file(
    features_path: Path,
    columns_to_retain: List[str],
    target_epsg: int,
    bbox=None,
    polygon=None,
) -> gpd.GeoDataFrame:
    """
    Load the features and reproject to the target crs.

    Remarks:
        * Reprojected version is "cached" so on a next call, it can be directly read.
        * Locking and waiting is used to ensure correct results if used in parallel.

    Args
        features_path:
        columns_to_retain:
        target_srs:
        bbox: bounds of the area to be loaded, in the target_epsg
    """
    # Load parcel file and preprocess it: remove excess columns + reproject if needed.
    # By convention, the features filename should end on the projection... so extract
    # epsg from filename
    start_time = datetime.now()
    """
    features_epsg = None
    if features_path.stem.find("_") != -1:
        splitted = features_path.stem.split("_")
        features_epsg = splitted[len(splitted) - 1]
        try:
            features_epsg = int(features_epsg)
        except Exception:
            logger.info(
                f"features_epsg {features_epsg } could not be cast to int for "
                f"{features_path.name}"
            )
            features_epsg = None
    """
    features_info = gfo.get_layerinfo(features_path)
    assert features_info.crs is not None
    features_epsg = features_info.crs.to_epsg()

    # Determine the correct filename for the input features in the correct projection.
    if features_epsg != target_epsg:
        features_prepr_path = (
            features_path.parent / f"{features_path.stem}_{target_epsg:.0f}.gpkg"
        )
    else:
        features_prepr_path = features_path

    # Prepare filename for a "busy file" to ensure proper behaviour in a parallel
    # processing context
    features_prepr_path_busy = Path(f"{str(features_prepr_path)}_busy")

    # If the file doesn't exist yet in right projection, read original input file to
    # reproject/write to new file with correct epsg
    features_gdf = None
    if not (features_prepr_path_busy.exists() or features_prepr_path.exists()):
        # Create lock file in an atomic way, so we are sure we are the only process
        # working on it. If function returns true, there isn't any other thread/process
        # already working on it
        if io_util.create_file_atomic(features_prepr_path_busy):
            try:
                # Read (all) original features + remove unnecessary columns...
                logger.info(f"Read original file {features_path}")
                start_time = datetime.now()
                logging.getLogger("fiona.ogrext").setLevel(logging.INFO)
                features_gdf = gfo.read_file(features_path)
                logger.info(
                    f"Read ready, found {len(features_gdf.index)} features, "
                    f"crs: {features_gdf.crs}, took "
                    f"{(datetime.now()-start_time).total_seconds()} s"
                )
                for column in features_gdf.columns:
                    if column not in columns_to_retain and column not in [
                        "geometry",
                        "x_ref",
                    ]:
                        features_gdf.drop(columns=column, inplace=True)

                # Reproject them
                logger.info(
                    f"Reproject features from {features_gdf.crs} to epsg {target_epsg}"
                )
                features_gdf = features_gdf.to_crs(epsg=target_epsg)
                logger.info("Reprojected, now sort on x_ref")

                if features_gdf is None:
                    raise Exception("features_gdf is None")
                # Order features on x coordinate
                if "x_ref" not in features_gdf.columns:
                    features_gdf["x_ref"] = features_gdf.geometry.bounds.minx
                features_gdf.sort_values(by=["x_ref"], inplace=True)
                features_gdf.reset_index(inplace=True)

                # Cache the file for future use
                logger.info(
                    f"Write {len(features_gdf.index)} reprojected features to "
                    f"{features_prepr_path}"
                )
                gfo.to_file(
                    features_gdf, features_prepr_path, index=False  # type: ignore
                )
                logger.info("Reprojected features written")

            except Exception as ex:
                # If an exception occurs...
                message = f"Delete possibly incomplete file: {features_prepr_path}"
                logger.exception(message)
                gfo.remove(features_prepr_path)
                raise Exception(message) from ex
            finally:
                # Remove lock file as everything is ready for other processes to use it
                features_prepr_path_busy.unlink()

            # Now filter the parcels that are in bbox provided
            if bbox is not None:
                logger.info(f"bbox provided, so filter features in the bbox of {bbox}")
                xmin, ymin, xmax, ymax = bbox
                features_gdf = features_gdf.cx[xmin:xmax, ymin:ymax]
                logger.info(f"Found {len(features_gdf.index)} features in bbox")

    # If there exists already a file with the features in the right projection, we can
    # just read the data
    if features_gdf is None:
        # If a "busy file" still exists, the file isn't ready yet, but another process
        # is working on it, so wait till it disappears
        wait_secs_max = 600
        wait_start_time = datetime.now()
        while features_prepr_path_busy.exists():
            time.sleep(1)
            wait_secs = (datetime.now() - wait_start_time).total_seconds()
            if wait_secs > wait_secs_max:
                raise Exception(
                    f"Waited {wait_secs} for busy file "
                    f"{features_prepr_path_busy} and it is still there!"
                )

        logger.info(f"Read {features_prepr_path}")
        start_time = datetime.now()
        features_gdf = gfo.read_file(features_prepr_path, bbox=bbox)
        logger.info(
            f"Read ready, found {len(features_gdf.index)} features, crs: "
            f"{features_gdf.crs}, took {(datetime.now()-start_time).total_seconds()} s"
        )

        # Order features on x_ref to (probably) have more clustering of features in
        # further action...
        if "x_ref" not in features_gdf.columns:
            features_gdf["x_ref"] = features_gdf.geometry.bounds.minx
        features_gdf.sort_values(by=["x_ref"], inplace=True)
        features_gdf.reset_index(inplace=True)

        # To be sure, remove the columns anyway...
        for column in features_gdf.columns:
            if column not in columns_to_retain and column not in ["geometry"]:
                features_gdf.drop(columns=column, inplace=True)

    # If there is a polygon provided, filter on the polygon (as well)
    if polygon is not None:
        logger.info("Filter polygon provided, start filter")
        polygon_gdf = gpd.GeoDataFrame(
            geometry=[polygon], crs="EPSG:4326", index=[0]  # type: ignore
        )
        logger.debug(f"polygon_gdf: {polygon_gdf}")
        logger.debug(
            f"polygon_gdf.crs: {polygon_gdf.crs}, features_gdf.crs: {features_gdf.crs}"
        )
        polygon_gdf = polygon_gdf.to_crs(features_gdf.crs)
        assert polygon_gdf is not None
        logger.debug(f"polygon_gdf, after reproj: {polygon_gdf}")
        logger.debug(
            f"polygon_gdf.crs: {polygon_gdf.crs}, features_gdf.crs: {features_gdf.crs}"
        )
        features_gdf = gpd.sjoin(
            features_gdf, polygon_gdf, how="inner", predicate="within"
        )

        # Drop column added by sjoin
        features_gdf.drop(columns="index_right", inplace=True)
        """
        spatial_index = gdf.sindex
        possible_matches_index = list(spatial_index.intersection(polygon.bounds))
        possible_matches = gdf.iloc[possible_matches_index]
        precise_matches = possible_matches[possible_matches.intersects(polygon)]
        """
        logger.info(f"Filter ready, found {len(features_gdf.index)}")

    # Ready, so return result...
    logger.debug(
        f"Loaded {len(features_gdf)} to calculate on in {datetime.now()-start_time}"
    )
    assert isinstance(features_gdf, gpd.GeoDataFrame)
    return features_gdf
