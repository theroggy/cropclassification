"""
Module with helper functions to prepare the data to use for the classification.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

import cropclassification.helpers.config_helper as conf
import cropclassification.helpers.pandas_helper as pdh

# Get a logger...
logger = logging.getLogger(__name__)


def create_train_test_sample(
    input_parcel_path: Path,
    output_parcel_train_path: Path,
    output_parcel_test_path: Path,
    balancing_strategy: str,
    training_query: Optional[str] = None,
    force: bool = False,
):
    """Create a seperate train and test sample from the general input file."""

    # If force False and the output files exist already, stop.
    if (
        force is False
        and output_parcel_train_path.exists() is True
        and output_parcel_test_path.exists() is True
    ):
        logger.warning(
            "create_train_test_sample: output files already exist and force is False: "
            f"{output_parcel_train_path}, {output_parcel_test_path}"
        )
        return

    # Load input data...
    logger.info(
        f"Start create_train_test_sample with balancing_strategy {balancing_strategy}"
    )
    logger.info(f"Read input file {input_parcel_path}")
    df_in = pdh.read_file(input_parcel_path)

    # If training_cross_pred_model_indexes is not None, only keep the parcels that have
    # one of the indexes specified
    if training_query is not None:
        logger.info(f"Filter parcels with {training_query=}")
        df_in = df_in.query(training_query)

    logger.debug(f"Read input file ready, shape: {df_in.shape}")

    # Init some many-used variables from config
    class_balancing_column = conf.columns["class_balancing"]
    class_column = conf.columns["class"]

    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        count_per_class = df_in.groupby(class_balancing_column, as_index=False).size()
        logger.info(
            f"Number of elements per classname in input dataset:\n{count_per_class}"
        )

    # The test dataset should be as representative as possible for the entire dataset,
    # so create this first as a 20% sample of each class without any additional checks.
    # Remark: group_keys=False evades that apply creates an extra index-level of the
    #     groups above the data and evades having to do
    #     .reset_index(level=class_balancing_column_NAME, drop=True)
    #     to get rid of the group level
    test_df = (
        df_in.groupby(class_balancing_column)
        .apply(pd.DataFrame.sample, frac=0.20, include_groups=False)
        .reset_index()
    )
    if not test_df.empty:
        test_df.set_index("level_1", inplace=True)
        test_df.index.name = None

    logger.debug(
        f"df_test after sampling 20% of data per class, shape: {test_df.shape}"
    )

    # The candidate parcel for training are all non-test parcel
    train_base_df = df_in[~df_in.index.isin(test_df.index)]
    logger.debug(f"df_train_base after isin\n{train_base_df}")

    # Remove parcels with too few pixels from the train sample
    min_pixcount = conf.marker.getfloat("min_nb_pixels_train")
    train_base_df = train_base_df[
        train_base_df[conf.columns["pixcount_s1s2"]] >= min_pixcount
    ]
    logger.debug(
        "Number of parcels in df_train_base after filter on pixcount >= "
        f"{min_pixcount}: {len(train_base_df)}"
    )

    # Some classes shouldn't be used for training... so remove them!
    logger.info(
        "Remove 'classes_to_ignore_for_train' from train sample (= where "
        f"{class_column} is in: {conf.marker.getlist('classes_to_ignore_for_train')}"
    )
    train_base_df = train_base_df[
        ~train_base_df[class_column].isin(
            conf.marker.getlist("classes_to_ignore_for_train")
        )
    ]

    # All classes_to_ignore aren't meant for training either...
    logger.info(
        f"Remove 'classes_to_ignore' from train sample (= where {class_column} is in: "
        f"{conf.marker.getlist('classes_to_ignore')}"
    )
    train_base_df = train_base_df[
        ~train_base_df[class_column].isin(conf.marker.getlist("classes_to_ignore"))
    ]

    # There shouldn't be any classes left that start with IGNORE now
    train_ignore_df = train_base_df[
        train_base_df[class_column].str.startswith("IGNORE")
    ]
    if len(train_ignore_df) > 0:
        raise ValueError(
            "There are still classes that start with IGNORE, after removing filtering "
            "all classes in classes_to_ignore and classes_to_ignore, this must be a "
            f"config error: {train_ignore_df[class_column].unique()}"
        )

    # Only keep parcels that are not to be ignored for training
    if "ignore_for_training" in train_base_df.columns:
        logger.info("Remove parcels from train sample where ignore_for_training == 1")
        train_base_df = train_base_df[train_base_df["ignore_for_training"] == 0]

    # Print the train base result before applying any balancing
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        count_per_class = train_base_df.groupby(
            class_balancing_column, as_index=False
        ).size()
        logger.info(
            "Number of elements per classname for train dataset, before balancing:\n"
            f"{count_per_class}"
        )

    # Depending on the balancing_strategy, use different way to get a training sample
    train_df = pd.DataFrame().reindex_like(train_base_df)
    if balancing_strategy == "BALANCING_STRATEGY_NONE":
        # Just use 25% of all non-test data as train data -> 25% of 80% of data -> 20%
        # of all data will be training date
        # Remark: - this is very unbalanced, eg. classes with 10.000 times the input
        #           size than other classes
        #         - this results in a relatively high accuracy in overall numbers, but
        #           the small classes are not detected at all
        train_df = (
            train_base_df.groupby(class_balancing_column)
            .apply(pd.DataFrame.sample, frac=0.25, include_groups=False)
            .reset_index()
        )
        if not train_df.empty:
            train_df.set_index("level_1", inplace=True)
            train_df.index.name = None

    elif balancing_strategy == "BALANCING_STRATEGY_MEDIUM":
        # Balance the train data, but still use some larger samples for the classes
        # that have a lot of members in the input dataset
        # Remark: with the upper limit of 10.000 this gives still OK results overall,
        #     and also the smaller classes give some results with upper limit of 4000
        #     results significantly less good.

        # For the larger classes, favor them by leaving the samples larger but cap at
        # upper_limit
        upper_limit = 10000
        lower_limit = 1000
        logger.info(
            f"Cap over {upper_limit}, keep the full number of training sample till "
            f"{lower_limit}, samples smaller than that are oversampled"
        )
        train_df = (
            train_base_df.groupby(class_balancing_column)
            .filter(lambda x: len(x) >= upper_limit)
            .groupby(class_balancing_column)
            .apply(pd.DataFrame.sample, upper_limit, include_groups=False)
            .reset_index()
        )
        if not train_df.empty:
            train_df.set_index("level_1", inplace=True)
            train_df.index.name = None

        # Middle classes use the number as they are
        train_df = pd.concat(
            [
                train_df,
                train_base_df.groupby(class_balancing_column)
                .filter(lambda x: len(x) < upper_limit)
                .groupby(class_balancing_column)
                .filter(lambda x: len(x) >= lower_limit),
            ]
        )
        # For smaller classes, oversample...
        train_base_df = (
            train_base_df.groupby(class_balancing_column)
            .filter(lambda x: len(x) < lower_limit)
            .groupby(class_balancing_column)
            .apply(
                pd.DataFrame.sample,
                lower_limit,
                replace=True,
                include_groups=False,
            )
            .reset_index()
        )
        if not train_base_df.empty:
            train_base_df.set_index("level_1", inplace=True)
            train_base_df.index.name = None
        train_df = pd.concat([train_df, train_base_df])

    elif balancing_strategy == "BALANCING_STRATEGY_MEDIUM2":
        # Balance the train data, but still use some larger samples for the classes
        # that have a lot of members in the input dataset
        # Remark: with the upper limit of 10.000 this gives still OK results overall,
        #     and also the smaller classes give some results with upper limit of 4000
        #     results significantly less good.

        # For the larger classes, leave the samples larger but cap
        # Cap 1
        cap_count_limit1 = 100000
        cap_train_limit1 = 30000
        logger.info(
            f"Cap balancing classes over {cap_count_limit1} to {cap_train_limit1}"
        )
        train_capped_df = (
            train_base_df.groupby(class_balancing_column)
            .filter(lambda x: len(x) >= cap_count_limit1)
            .groupby(class_balancing_column)
        )
        if len(train_capped_df) > 0:
            train_df = pd.concat(
                [
                    train_df,
                    train_capped_df.apply(
                        pd.DataFrame.sample,
                        cap_train_limit1,
                        include_groups=False,
                    )
                    .reset_index()
                    .set_index("level_1"),
                ]
            )

        # Cap 2
        cap_count_limit2 = 50000
        cap_train_limit2 = 20000
        logger.info(
            f"Cap balancing classes between {cap_count_limit2} and {cap_count_limit1} "
            f"to {cap_train_limit2}"
        )
        train_capped_df = (
            train_base_df.groupby(class_balancing_column)
            .filter(lambda x: len(x) < cap_count_limit1)
            .groupby(class_balancing_column)
            .filter(lambda x: len(x) >= cap_count_limit2)
            .groupby(class_balancing_column)
        )
        if len(train_capped_df) > 0:
            train_df = pd.concat(
                [
                    train_df,
                    train_capped_df.apply(
                        pd.DataFrame.sample,
                        cap_train_limit2,
                        include_groups=False,
                    )
                    .reset_index()
                    .set_index("level_1"),
                ]
            )

        # Cap 3
        cap_count_limit3 = 20000
        cap_train_limit3 = 10000
        logger.info(
            f"Cap balancing classes between {cap_count_limit3} and {cap_count_limit2} "
            f"to {cap_train_limit3}"
        )
        train_capped_df = (
            train_base_df.groupby(class_balancing_column)
            .filter(lambda x: len(x) < cap_count_limit2)
            .groupby(class_balancing_column)
            .filter(lambda x: len(x) >= cap_count_limit3)
            .groupby(class_balancing_column)
        )
        if len(train_capped_df) > 0:
            train_df = pd.concat(
                [
                    train_df,
                    train_capped_df.apply(
                        pd.DataFrame.sample,
                        cap_train_limit3,
                        include_groups=False,
                    )
                    .reset_index()
                    .set_index("level_1"),
                ]
            )

        # Cap 4
        cap_count_limit4 = 10000
        cap_train_limit4 = 10000
        logger.info(
            f"Cap balancing classes between {cap_count_limit4} and {cap_count_limit3} "
            f"to {cap_train_limit4}"
        )
        train_capped_df = (
            train_base_df.groupby(class_balancing_column)
            .filter(lambda x: len(x) < cap_count_limit3)
            .groupby(class_balancing_column)
            .filter(lambda x: len(x) >= cap_count_limit4)
            .groupby(class_balancing_column)
        )
        if len(train_capped_df) > 0:
            train_df = pd.concat(
                [
                    train_df,
                    train_capped_df.apply(
                        pd.DataFrame.sample,
                        cap_train_limit4,
                        include_groups=False,
                    )
                    .reset_index()
                    .set_index("level_1"),
                ]
            )

        # Middle classes use the number as they are, smaller classes are oversampled
        oversample_count = 1000
        logger.info(
            f"For classes between {cap_count_limit4} and {oversample_count}, just use "
            "all samples"
        )
        train_df = pd.concat(
            [
                train_df,
                train_base_df.groupby(class_balancing_column)
                .filter(lambda x: len(x) < cap_count_limit4)
                .groupby(class_balancing_column)
                .filter(lambda x: len(x) >= oversample_count),
            ]
        )
        # For smaller classes, oversample...
        logger.info(
            f"For classes smaller than {oversample_count}, oversample to "
            f"{oversample_count}"
        )
        train_capped_df = (
            train_base_df.groupby(class_balancing_column)
            .filter(lambda x: len(x) < oversample_count)
            .groupby(class_balancing_column)
        )
        if len(train_capped_df) > 0:
            train_df = pd.concat(
                [
                    train_df,
                    train_capped_df.apply(
                        pd.DataFrame.sample,
                        oversample_count,
                        replace=True,
                        include_groups=False,
                    )
                    .reset_index()
                    .set_index("level_1"),
                ]
            )

    elif balancing_strategy == "BALANCING_STRATEGY_PROPORTIONAL_GROUPS":
        # Balance the train data, but still use some larger samples for the classes
        # that have a lot of members in the input dataset
        # Remark: with the upper limit of 10.000 this gives still OK results overall,
        #     and also the smaller classes give some results with upper limit of 4000
        #     results significantly less good.

        # For the larger classes, leave the samples larger but cap
        train_df = pd.DataFrame()
        upper_count_limit1 = 100000
        upper_train_limit1 = 30000
        logger.info(
            f"Cap balancing classes over {upper_count_limit1} to {upper_train_limit1}"
        )
        train_df = (
            train_base_df.groupby(class_balancing_column)
            .filter(lambda x: len(x) >= upper_count_limit1)
            .groupby(class_balancing_column)
            .apply(pd.DataFrame.sample, upper_train_limit1, include_groups=False)
            .reset_index()
        )
        if not train_df.empty:
            train_df.set_index("level_1", inplace=True)
            train_df.index.name = None

        upper_count_limit2 = 50000
        upper_train_limit2 = 20000
        logger.info(
            f"Cap balancing classes between {upper_count_limit2} and "
            f"{upper_count_limit1} to {upper_train_limit2}"
        )
        train_limit2_df = (
            train_base_df.groupby(class_balancing_column)
            .filter(lambda x: len(x) < upper_count_limit1)
            .groupby(class_balancing_column)
            .filter(lambda x: len(x) >= upper_count_limit2)
            .groupby(class_balancing_column)
        )
        if len(train_limit2_df) > 0:
            train_df = pd.concat(
                [
                    train_df,
                    train_limit2_df.apply(
                        pd.DataFrame.sample,
                        upper_train_limit2,
                        include_groups=False,
                    )
                    .reset_index()
                    .set_index("level_1"),
                ]
            )
        upper_count_limit3 = 20000
        upper_train_limit3 = 10000
        logger.info(
            f"Cap balancing classes between {upper_count_limit3} and "
            f"{upper_count_limit2} to {upper_train_limit3}"
        )
        train_limit3_df = (
            train_base_df.groupby(class_balancing_column)
            .filter(lambda x: len(x) < upper_count_limit2)
            .groupby(class_balancing_column)
            .filter(lambda x: len(x) >= upper_count_limit3)
            .groupby(class_balancing_column)
        )
        if len(train_limit3_df) > 0:
            train_df = pd.concat(
                [
                    train_df,
                    train_limit3_df.apply(
                        pd.DataFrame.sample,
                        upper_train_limit3,
                        include_groups=False,
                    )
                    .reset_index()
                    .set_index("level_1"),
                ]
            )
        upper_count_limit4 = 10000
        upper_train_limit4 = 5000
        logger.info(
            f"Cap balancing classes between {upper_count_limit4} and "
            f"{upper_count_limit3} to {upper_train_limit4}"
        )
        train_limit4_df = (
            train_base_df.groupby(class_balancing_column)
            .filter(lambda x: len(x) < upper_count_limit3)
            .groupby(class_balancing_column)
            .filter(lambda x: len(x) >= upper_count_limit4)
            .groupby(class_balancing_column)
        )
        if len(train_limit4_df) > 0:
            train_df = pd.concat(
                [
                    train_df,
                    train_limit4_df.apply(
                        pd.DataFrame.sample,
                        upper_train_limit4,
                        include_groups=False,
                    )
                    .reset_index()
                    .set_index("level_1"),
                ]
            )
        # For smaller balancing classes, just use all samples
        train_df = pd.concat(
            [
                train_df,
                train_base_df.groupby(class_balancing_column).filter(
                    lambda x: len(x) < upper_count_limit4
                ),
            ]
        )

    elif balancing_strategy == "BALANCING_STRATEGY_UPPER_LIMIT":
        # Balance the train data, but still use some larger samples for the classes
        # that have a lot of members in the input dataset
        # Remark: with the upper limit of 10.000 this gives still OK results overall,
        #     and also the smaller classes give some results with upper limit of 4000
        #     results significantly less good.

        # For the larger classes, favor them by leaving the samples larger but cap at
        # upper_limit
        upper_limit = 10000
        logger.info(f"Cap over {upper_limit}...")
        train_df = (
            train_base_df.groupby(class_balancing_column)
            .filter(lambda x: len(x) >= upper_limit)
            .groupby(class_balancing_column)
            .apply(pd.DataFrame.sample, upper_limit, include_groups=False)
            .reset_index()
        )
        if not train_df.empty:
            train_df.set_index("level_1", inplace=True)
            train_df.index.name = None

        # For smaller classes, just use all samples
        train_df = pd.concat(
            [
                train_df,
                train_base_df.groupby(class_balancing_column).filter(
                    lambda x: len(x) < upper_limit
                ),
            ]
        )

    elif balancing_strategy == "BALANCING_STRATEGY_EQUAL":
        # In theory the most logical way to balance: make sure all classes have the
        # same amount of training data by undersampling the largest classes and
        # oversampling the small classes.
        train_df = (
            train_base_df.groupby(class_balancing_column)
            .apply(pd.DataFrame.sample, 2000, replace=True, include_groups=False)
            .reset_index()
        )
        if not train_df.empty:
            train_df.set_index("level_1", inplace=True)
            train_df.index.name = None

    else:
        raise Exception(f"Unknown balancing strategy, STOP!: {balancing_strategy}")

    # Log the resulting numbers per class in the train sample
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        count_per_class = train_df.groupby(
            class_balancing_column, as_index=False
        ).size()
        logger.info(
            "Number of elements per class_balancing_column in train dataset:\n"
            f"{count_per_class}"
        )
        if class_balancing_column != class_column:
            count_per_class = train_df.groupby(class_column, as_index=False).size()
            logger.info(
                "Number of elements per class_column in train dataset:\n"
                f"{count_per_class}"
            )

    # Log the resulting numbers per class in the test sample
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        count_per_class = test_df.groupby(class_balancing_column, as_index=False).size()
        logger.info(
            "Number of elements per class_balancing_column in test dataset:\n"
            f"{count_per_class}"
        )
        if class_balancing_column != class_column:
            count_per_class = test_df.groupby(class_column, as_index=False).size()
            logger.info(
                "Number of elements per class_column in test dataset:\n"
                f"{count_per_class}"
            )

    # Write to output files
    logger.info("Write the output files")
    train_df.set_index(conf.columns["id"], inplace=True)
    test_df.set_index(conf.columns["id"], inplace=True)
    pdh.to_file(train_df, output_parcel_train_path)  # The ID column is the index...
    pdh.to_file(test_df, output_parcel_test_path)  # The ID column is the index...
