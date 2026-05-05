import logging

import pandas as pd

from cropclassification import calc_cover


def test_categorize_consolidated_predictions_matches_expected_values():
    probabilities = pd.Series([0.8, 0.5, 0.3, 0.7, 0.45, None, "bad"])
    sources = pd.Series(
        ["ML", "ML", "ML", "THRESHOLD", "THRESHOLD", "THRESHOLD", "THRESHOLD"]
    )

    threshold_onbedekt_threshold = 0.5
    ml_onbedekt_threshold = 0.65

    result = calc_cover._categorize_consolidated_predictions(
        probabilities=probabilities,
        sources=sources,
        threshold_onbedekt_threshold=threshold_onbedekt_threshold,
        ml_onbedekt_doubt=ml_onbedekt_threshold,
    )
    expected = pd.Series(
        ["ONBEDEKT", "BEDEKT", "BEDEKT", "ONBEDEKT", "DOUBT", "NODATA", "NODATA"],
        dtype="object",
    )

    pd.testing.assert_series_equal(result, expected)


def test_categorize_consolidated_predictions_preserve_boundary_behavior():
    probabilities = pd.Series([0.5, 0.4, 0.65, 0.4])
    sources = pd.Series(["THRESHOLD", "THRESHOLD", "ML", "ML"])

    result = calc_cover._categorize_consolidated_predictions(
        probabilities=probabilities,
        sources=sources,
        threshold_onbedekt_threshold=0.5,
        ml_onbedekt_doubt=0.65,
    )

    expected = pd.Series(["DOUBT", "BEDEKT", "BEDEKT", "BEDEKT"], dtype="object")

    pd.testing.assert_series_equal(result, expected)


def test_determine_groundtruth_cover_series_prefers_vaststellingen_codes():
    merged_df = pd.DataFrame(
        {
            "VASTSTELLINGEN": ["pc35et extra", None, None, None],
            "MON_VRU": ["MON_VRU_ANDERS", "MON_VRU_BRAAK", "MON_VRU_ANDERS", None],
        }
    )

    result = calc_cover._determine_groundtruth_cover_series(merged_df)

    assert result.tolist() == ["ONBEDEKT", "ONBEDEKT", "BEDEKT", "NODATA"]


def test_consolidate_cover_predictions_promotes_confident_ml_with_s2_available(
    monkeypatch, tmp_path
):
    input_parcel_path = tmp_path / "input.gpkg"
    cover_path = tmp_path / "cover.sqlite"

    monkeypatch.setattr(
        calc_cover.gfo,
        "get_layerinfo",
        lambda _path: type("LayerInfo", (), {"columns": ("UID",)})(),
    )

    def fake_read_file(path, *_args, **_kwargs):
        if path == cover_path:
            return pd.DataFrame(
                {
                    "UID": [1],
                    "pred1": ["DOUBT"],
                    "pred1_prob": [0.45],
                    "cover_s2_ndvi": ["other"],
                    "ml_prob": [0.99],
                }
            )
        raise AssertionError(f"Unexpected read_file path: {path}")

    monkeypatch.setattr(calc_cover.gfo, "read_file", fake_read_file)

    result = calc_cover._consolidate_cover_predictions(
        input_parcel_path=input_parcel_path,
        parcels_cover_paths=[cover_path],
        id_column="UID",
        ml_threshold_onbedekt=0.65,
    )

    assert result["pred_source"].tolist() == ["ML"]
    assert result["pred1_prob"].tolist() == [0.99]
    assert result["pred_consolidated"].tolist() == ["ONBEDEKT"]
    assert result["s2_available"].tolist() == [True]


def test_consolidate_cover_predictions_keeps_threshold_when_ml_is_not_confident(
    monkeypatch, tmp_path
):
    input_parcel_path = tmp_path / "input.gpkg"
    cover_path = tmp_path / "cover.sqlite"

    monkeypatch.setattr(
        calc_cover.gfo,
        "get_layerinfo",
        lambda _path: type("LayerInfo", (), {"columns": ("UID",)})(),
    )

    def fake_read_file(path, *_args, **_kwargs):
        if path == cover_path:
            return pd.DataFrame(
                {
                    "UID": [1],
                    "pred1": ["DOUBT"],
                    "pred1_prob": [0.45],
                    "cover_s2_ndvi": ["other"],
                    "ml_prob": [0.65],
                }
            )
        raise AssertionError(f"Unexpected read_file path: {path}")

    monkeypatch.setattr(calc_cover.gfo, "read_file", fake_read_file)

    result = calc_cover._consolidate_cover_predictions(
        input_parcel_path=input_parcel_path,
        parcels_cover_paths=[cover_path],
        id_column="UID",
        ml_threshold_onbedekt=0.65,
    )

    assert result["pred_source"].tolist() == ["THRESHOLD"]
    assert result["pred1_prob"].tolist() == [0.45]
    assert result["pred_consolidated"].tolist() == ["DOUBT"]


def test_calc_cover_and_predict_recreates_missing_geo_without_rebuilding_result(
    monkeypatch, tmp_path
):
    input_parcel_path = tmp_path / "input.gpkg"
    ts_path = tmp_path / "timeseries.sqlite"
    output_path = tmp_path / "cover.sqlite"
    output_geo_path = tmp_path / "cover.gpkg"
    output_path.touch()

    monkeypatch.setattr(
        calc_cover,
        "logger",
        logging.getLogger("test.calc_cover"),
        raising=False,
    )
    monkeypatch.setattr(calc_cover.conf, "columns", {"id": "UID"}, raising=False)

    def fail_calc_cover(**_kwargs):
        raise AssertionError(
            "_calc_cover should not run when only the geo output is missing"
        )

    monkeypatch.setattr(calc_cover, "_calc_cover", fail_calc_cover)

    def noop_remove(*_args, **_kwargs):
        return None

    monkeypatch.setattr(calc_cover.gfo, "remove", noop_remove)

    written = {}

    def fake_to_file(df, path):
        written["path"] = path
        written["df"] = df.copy()

    monkeypatch.setattr(calc_cover.gfo, "to_file", fake_to_file)

    def fake_read_file(path, *_args, **kwargs):
        if path == input_parcel_path and kwargs.get("ignore_geometry"):
            return pd.DataFrame({"UID": [1]})
        if path == input_parcel_path:
            return pd.DataFrame({"UID": [1], "PRC_NIS": [23025]})
        if path == output_path:
            return pd.DataFrame(
                {
                    "UID": [1],
                    "pred1": ["other"],
                    "pred1_prob": [0.2],
                    "ml_prob": [0.1],
                }
            )
        if path == ts_path:
            return pd.DataFrame({"UID": [1], "s1_feature": [1.0]})
        raise AssertionError(f"Unexpected read_file path: {path}")

    monkeypatch.setattr(calc_cover.pdh, "read_file", fake_read_file)

    calc_cover._calc_cover_and_predict(
        input_parcel_path=input_parcel_path,
        ts_path=ts_path,
        parcel_columns=None,
        ndvi_threshold=0.35,
        output_path=output_path,
        classifier=None,
        feature_names=[],
        output_geo_path=output_geo_path,
        force=False,
    )

    assert written["path"] == output_geo_path
    assert written["df"]["provincie"].tolist() == ["VLBR"]
    assert written["df"]["pred1"].tolist() == ["other"]
