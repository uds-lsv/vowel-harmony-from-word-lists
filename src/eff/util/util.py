from collections import defaultdict
import dill
import errno
import os
from pathlib import Path
from typing import Tuple
import torch


def _serialize_object(path: Path, obj: object, file_prefix: str) -> None:
    file = path / (file_prefix + '.dill')
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    else:
        if file.exists():
            file.unlink()
    with file.open('wb') as f:
        dill.dump(obj, f)


def _deserialize_object(path: Path, file_prefix: str) -> object:
    file = path / (file_prefix + '.dill')
    if not file.exists():
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), file.absolute())
    obj = dill.load(file.open('rb'))
    return obj


def save_results(base_path: Path, lang_id: str, datasets: defaultdict, result: defaultdict, \
    criterion: torch.nn.Module, model: torch.nn.Module) -> None:
    """ Convenience function to jointly serialize the results and the data they are derived from."""
    out_path = base_path / lang_id
    _serialize_object(out_path, datasets, 'datasets')
    _serialize_object(out_path, result, 'results')
    _serialize_object(out_path, criterion, 'criterion')
    _serialize_object(out_path, model, 'model')


def load_results(base_path: Path, lang_id: str) -> Tuple[defaultdict, defaultdict]:
    """ Convenience function to deserialize results and data. """
    out_path = base_path / lang_id
    datasets = _deserialize_object(out_path, 'datasets')
    result = _deserialize_object(out_path, 'results')
    criterion = _deserialize_object(out_path, 'criterion')
    model = _deserialize_object(out_path, 'model')
    return datasets, result, criterion, model
