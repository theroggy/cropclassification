"""Module with helper functions regarding (keras) models."""

import glob
import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

# -------------------------------------------------------------
# First define/init some general variables/constants
# -------------------------------------------------------------
# Get a logger...
logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)

# -------------------------------------------------------------
# The real work
# -------------------------------------------------------------


def get_max_data_version(model_dir: Path) -> int:
    """Get the maximum data version a model exists for in the model_dir.

    Args:
        model_dir: the dir to search models in
    """
    models_df = get_models(model_dir)
    if models_df is not None and len(models_df.index) > 0:
        train_data_version_max = models_df["train_data_version"].max()
        return int(train_data_version_max)
    else:
        return -1


def format_model_base_filename(
    segment_subject: str, train_data_version: int, model_architecture: str
) -> str:
    """Format the parameters into a model_base_filename.

    Args:
        segment_subject: the segment subject
        train_data_version: the version of the data used to train the model
        model_architecture: the architecture of the model
    """
    retval = f"{segment_subject}_{train_data_version:02}_{model_architecture}"
    return retval


def format_model_filename(
    segment_subject: str,
    train_data_version: int,
    model_architecture: str,
    acc_train: float,
    acc_val: float,
    acc_combined: float,
    epoch: int,
) -> str:
    """Format the parameters into a model_filename.

    Args:
        segment_subject: the segment subject
        train_data_version: the version of the data used to train the model
        model_architecture: the architecture of the model
        acc_train: the accuracy reached for the training dataset
        acc_val: the accuracy reached for the validation dataset
        acc_combined: the average of the train and validation accuracy
        epoch: the epoch during training that reached these model weights
    """
    base_filename = format_model_base_filename(
        segment_subject=segment_subject,
        train_data_version=train_data_version,
        model_architecture=model_architecture,
    )
    filename = format_model_filename2(
        model_base_filename=base_filename,
        acc_train=acc_train,
        acc_val=acc_val,
        acc_combined=acc_combined,
        epoch=epoch,
    )
    return filename


def format_model_filename2(
    model_base_filename: str,
    acc_train: float,
    acc_val: float,
    acc_combined: float,
    epoch: int,
) -> str:
    """Format the parameters into a model_filename.

    Args:
        model_base_filename: the base filename of the model
        acc_train: the accuracy reached for the training dataset
        acc_val: the accuracy reached for the validation dataset
        acc_combined: the average of the train and validation accuracy
        epoch: the epoch during training that reached these model weights
    """
    res = (
        f"{model_base_filename}_{acc_combined:.5f}_{acc_train:.5f}_{acc_val:.5f}_"
        f"{epoch}.hdf5"
    )
    return res


def parse_model_filename(path: Path) -> dict:
    """Parse a model_filename to a dict containing the properties of the model.

    Result:
        * segment_subject: the segment subject
        * train_data_version: the version of the data used to train the model
        * model_architecture: the architecture of the model
        * acc_train: the accuracy reached for the training dataset
        * acc_val: the accuracy reached for the validation dataset
        * acc_combined: the average of the train and validation accuracy
        * epoch: the epoch during training that reached these model weights

    Args:
        path: the path to the model file
    """
    # Prepare path to extract info
    param_values = path.stem.split("_")

    # Now extract fields...
    segment_subject = param_values[0]
    train_data_version = int(param_values[1])
    model_architecture = param_values[2]
    if len(param_values) > 3:
        acc_combined = float(param_values[3])
        acc_train = float(param_values[4])
        acc_val = float(param_values[5])
        epoch = int(param_values[6])
    else:
        acc_combined = 0.0
        acc_train = 0.0
        acc_val = 0.0
        epoch = 0

    return {
        "path": path,
        "filename": path.name,
        "segment_subject": segment_subject,
        "train_data_version": train_data_version,
        "model_architecture": model_architecture,
        "acc_combined": acc_combined,
        "acc_train": acc_train,
        "acc_val": acc_val,
        "epoch": epoch,
    }


def get_models(
    model_dir: Path, model_base_filename: Optional[str] = None
) -> pd.DataFrame:
    """Return the list of models in the model_dir passed.

    It is returned as a dataframe with the columns as returned in parse_model_filename.

    Args:
        model_dir: dir containing the models
        model_base_filename: optional, if passed, only the models with this
            base filename will be returned
    """
    # glob search string
    if model_base_filename is not None:
        model_weight_paths = glob.glob(
            f"{model_dir!s}{os.sep}{model_base_filename}_*.hdf5"
        )
    else:
        model_weight_paths = glob.glob(f"{model_dir!s}{os.sep}*.hdf5")

    # Loop through all models and extract necessary info...
    model_info_list = []
    for path in model_weight_paths:
        model_info_list.append(parse_model_filename(Path(path)))

    return pd.DataFrame(model_info_list)


def get_best_model(
    model_dir: Path, acc_metric_mode: str, model_base_filename: Optional[str] = None
) -> Optional[dict]:
    """Get model with the highest accuracy for the highest traindata version in the dir.

    Args:
        model_dir: dir containing the models
        acc_metric_mode: use 'min' if the accuracy metrics should be as low as possible,
            'max' if a higher values is better.
        model_base_filename: optional, if passed, only the models with this
            base filename will be taken in account
    """
    # Check validaty of input
    acc_metric_mode_values = ["min", "max"]
    if acc_metric_mode not in acc_metric_mode_values:
        raise Exception(
            f"Invalid value for mode: {acc_metric_mode}, should be one of "
            f"{acc_metric_mode_values}"
        )

    # Get list of existing models for this train dataset
    model_info_df = get_models(
        model_dir=model_dir, model_base_filename=model_base_filename
    )

    # If no model_base_filename provided, take highest data version
    if model_base_filename is None:
        max_data_version = get_max_data_version(model_dir)
        if max_data_version == -1:
            return None
        model_info_df = model_info_df.loc[
            model_info_df["train_data_version"] == max_data_version
        ]

    if len(model_info_df) > 0:
        if acc_metric_mode == "max":
            return model_info_df.loc[model_info_df["acc_combined"].values.argmax()]  # type: ignore
        else:
            return model_info_df.loc[model_info_df["acc_combined"].values.argmin()]  # type: ignore
    else:
        return None


def save_and_clean_models(
    model_save_dir: Path,
    model_save_base_filename: str,
    acc_metric_mode: str,
    new_model=None,
    new_model_acc_train: Optional[float] = None,
    new_model_acc_val: Optional[float] = None,
    new_model_epoch: Optional[int] = None,
    save_weights_only: bool = False,
    verbose: bool = True,
    debug: bool = False,
    only_report: bool = False,
):
    """Save the new model if it is good enough.

    Existing models are removed if they are worse than the new or other existing models.

    Args:
        model_save_dir: dir containing the models
        model_save_base_filename: base filename that will be used
        acc_metric_mode (str): use 'min' if the accuracy metrics should be
                as low as possible, 'max' if a higher values is better.
        new_model: optional, the keras model object that will be saved
        new_model_acc_train: optional: the accuracy on the train dataset
        new_model_acc_val: optional: the accuracy on the validation dataset
        new_model_epoch: optional: the epoch in the training
        save_weights_only: optional: only save the weights of the model
        verbose: report the best model after save and cleanup
        debug: write debug logging
        only_report: optional: only report which models would be cleaned up
    """
    # Check validaty of input
    acc_metric_mode_values = ["min", "max"]
    if acc_metric_mode not in acc_metric_mode_values:
        raise Exception(
            f"Invalid value for mode: {acc_metric_mode}, should be one of "
            f"{acc_metric_mode_values}"
        )

    # Get a list of all existing models
    model_info_df = get_models(
        model_dir=model_save_dir, model_base_filename=model_save_base_filename
    )

    # If there is a new model passed as param, add it to the list
    new_model_path = None
    if (
        new_model is not None
        and new_model_acc_train is not None
        and new_model_acc_val is not None
        and new_model_epoch is not None
    ):
        # Calculate combined accuracy
        new_model_acc_combined = (new_model_acc_train + new_model_acc_val) / 2

        # Build save path
        new_model_filename = format_model_filename2(
            model_base_filename=model_save_base_filename,
            acc_combined=new_model_acc_combined,
            acc_train=new_model_acc_train,
            acc_val=new_model_acc_val,
            epoch=new_model_epoch,
        )
        new_model_path = model_save_dir / new_model_filename

        # Append model to the retrieved models...
        model_info_df = pd.concat(
            [
                model_info_df,
                pd.DataFrame(
                    [
                        {
                            "path": new_model_path,
                            "filename": new_model_filename,
                            "acc_combined": new_model_acc_combined,
                            "acc_train": new_model_acc_train,
                            "acc_val": new_model_acc_val,
                            "epoch": new_model_epoch,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    # For each model found, check if there is one with ALL parameters
    # higher than itself. If one is found: delete model
    # Remark: the list is sorted before iterating it, this way the logging
    # is sorted from "worst to best"
    if acc_metric_mode == "max":
        model_info_sorted_df = model_info_df.sort_values(by="acc_combined")
    else:
        model_info_sorted_df = model_info_df.sort_values(
            by="acc_combined", ascending=False
        )
    for _, model_info in model_info_sorted_df.iterrows():
        if acc_metric_mode == "max":
            better_ones_df = model_info_df[
                (model_info_df["path"] != model_info["path"])
                & (model_info_df["acc_combined"] >= model_info["acc_combined"])
                & (model_info_df["acc_train"] >= model_info["acc_train"])
                & (model_info_df["acc_val"] >= model_info["acc_val"])
            ]
        else:
            better_ones_df = model_info_df[
                (model_info_df["path"] != model_info["path"])
                & (model_info_df["acc_combined"] <= model_info["acc_combined"])
                & (model_info_df["acc_train"] <= model_info["acc_train"])
                & (model_info_df["acc_val"] <= model_info["acc_val"])
            ]

        # If one or more better ones are found, no use in keeping it...
        if len(better_ones_df) > 0:
            if only_report:
                logger.debug(f"DELETE {model_info['filename']}")
            elif model_info["path"].exists():
                logger.debug(f"DELETE {model_info['filename']}")
                os.remove(model_info["path"])

            if debug:
                print(f"Better one(s) found for{model_info['filename']}:")
                for _, better_one in better_ones_df.iterrows():
                    print(f"  {better_one['filename']}")
        else:
            # No better one found, so keep it
            logger.debug(f"KEEP {model_info['filename']}")

            # If it is the new model that needs to be kept, save to disk
            if (
                new_model_path is not None
                and new_model is not None
                and only_report is not True
                and model_info["path"] == new_model_path
                and not new_model_path.exists()
            ):
                if save_weights_only:
                    new_model.save_weights(new_model_path)
                else:
                    new_model.save(new_model_path)

    if verbose is True or debug is True:
        best_model = get_best_model(
            model_save_dir, acc_metric_mode, model_save_base_filename
        )
        if best_model is not None:
            print(
                f"BEST MODEL: acc_combined: {best_model['acc_combined']}, acc_train: "
                f"{best_model['acc_train']}, acc_val: {best_model['acc_val']}, epoch: "
                f"{best_model['epoch']}"
            )


if __name__ == "__main__":
    # raise Exception("Not implemented")

    # General inits
    segment_subject = "greenhouses"
    base_dir = Path("X:/PerPersoon/PIEROG/Taken/2018/2018-08-12_AutoSegmentation")
    traindata_version = 17
    model_architecture = "inceptionresnetv2+linknet"

    project_dir = base_dir / segment_subject

    # Init logging
    from . import log_helper

    log_dir = project_dir / "log"
    logger = log_helper.main_log_init(log_dir, __name__)

    """
    print(get_models(model_dir="",
                     model_basename=""))
    """
    # Test the clean_models function (without new model)
    # Build save dir and model base filename
    model_save_dir = project_dir / "models"
    model_save_base_filename = format_model_base_filename(
        segment_subject, traindata_version, model_architecture
    )

    # Clean the models (only report)
    save_and_clean_models(
        model_save_dir=model_save_dir,
        model_save_base_filename=model_save_base_filename,
        acc_metric_mode="max",
        only_report=True,
    )
