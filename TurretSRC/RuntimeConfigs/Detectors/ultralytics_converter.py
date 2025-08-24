from __future__ import annotations

from pathlib import Path
from ultralytics import YOLO
import os
import torch.nn as nn


def convert_to_proper_format(save_directory: Path, ultralytics_model_key: Path | str, hyperparams: dict = None) -> Path:
    """
    This function converts the model that you passed in to the format specified in the hyperparams.
    If you do not specify a format, it will default to pytorch.
    Formats can be found here:
    https://docs.ultralytics.com/modes/export/#export-formats
    Args:
        save_directory: The directory where you want to save the models. Preferably somewhere you will not commit
            to git.
        ultralytics_model_key: The model key, this can be a string to download from ultralytics servers or your own
            pretrained model or whatever.
        hyperparams: The hyperparameters that we want to use.
    Returns:
        The path to the new converted model
    """
    if hyperparams is None:
        hyperparams: dict = {}

    model_name: str
    if isinstance(ultralytics_model_key, Path):
        model_name = ultralytics_model_key.name
    else:
        model_name = ultralytics_model_key

    model: YOLO = YOLO(ultralytics_model_key)

    if not isinstance(model.model, nn.Module):
        raise ValueError("ERROR, You passed a non-pytorch type to convert. Perhaps it's already been converted?")

    # Check the cache if the directory exists.
    # TODO: fix the nesting of cache.
    if "format" in hyperparams:
        if hyperparams["format"] == "edgetpu":
            exported_model_name: str = f"{model_name}_full_integer_quant_edgetpu.tflite"
            if (save_directory / f"{model_name}_saved_model" / exported_model_name).exists():
                return save_directory / f"{model_name}_saved_model" / exported_model_name
        elif hyperparams["format"] == "engine":
            return Path("") #TODO add wherever the path for engine will be.
    else:
        if (save_directory / f"{model_name}.torchscript").exists():
            return save_directory / f"{model_name}.torchscript"

    # The directory does not exist, so we have to generate it.
    print("Cache miss: recompiling the format...")
    old_dir: Path = Path.cwd()
    os.chdir(save_directory)

    converted_model_path = save_directory / Path(model.export(**hyperparams))

    os.chdir(old_dir)

    return converted_model_path
