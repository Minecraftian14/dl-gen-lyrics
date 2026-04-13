import os, pickle, re
from collections import defaultdict
from typing import Callable

import numpy as np
import pandas as pd

import torch
from torch import nn


def _write_object(data, file, bone):
    match bone:
        case 'numpy': np.save(file, data)
        case 'pandas': data.to_parquet(file)
        case 'torch': torch.save(data, file)
        case _: pickle.dump(data, open(file, 'wb'))


def _read_object(file, bone):
    match bone:
        case 'numpy': return np.load(file)
        case 'pandas': return pd.read_parquet(file)
        case 'torch': return torch.load(file)
        case _: return pickle.load(open(file, 'rb'))


def _type_to_bone(data):
    match type(data):
        case np.ndarray: return 'numpy'
        case pd.DataFrame: return 'pandas'
        case pd.Series: return 'pandas'
        case nn.Module: return 'torch'
        case _: return 'pickle'


_bone_to_ext = defaultdict(lambda: 'pkl')
_bone_to_ext.update({'numpy': 'npy', 'pandas': 'parquet', 'torch': 'pth'})


def key_cached(key, data_supplier,
               bone=None, root='temp', group=None,
               saver: Callable[[any, str], None] = None, loader: Callable[[str], None] = None):
    #
    if group is not None: key = group + '.' + key

    bone_file = os.path.join(root, f"{key}.bone")

    if os.path.exists(bone_file):
        print("Loaded Cache for", key)
        if loader is not None: return loader(os.path.join(root, key))

        with open(bone_file, 'r') as f: bone = f.read().strip()
        return _read_object(os.path.join(root, f"{key}.{_bone_to_ext[bone]}"), bone)

    else:
        if saver is not None:
            with open(bone_file, 'w') as f: f.write("custom")
            return saver(data_supplier(), os.path.join(root, key))

        data = data_supplier()
        bone = _type_to_bone(type(data))
        _write_object(data, os.path.join(root, f"{key}.{_bone_to_ext[bone]}"), bone)
        with open(bone_file, 'w') as f: f.write(bone)
        return data


def get_function_identifier(function):
    return re.search(rf"\w+\.{function.__name__}", str(function)).group(0)


def cached(key='cached', root='temp', saver=None, loader=None):
    def decorator(function):
        group = get_function_identifier(function)

        def wrapper(*args, **kwargs):
            return key_cached(key, lambda: function(*args, **kwargs), root=root, group=group, saver=saver, loader=loader)

        return wrapper

    return decorator


class SSSS:
    @cached("SSSS")
    def lol(self):
        return 456


if __name__ == '__main__':
    print("globals()")
    print(SSSS().lol())
