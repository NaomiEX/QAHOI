# ------------------------------------------------------------------------
# QAHOI
# Copyright (c) 2021 Junwen Chen. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------

import torch.utils.data

from .hico import build as build_hico
from .vcoco import build as build_vcoco
from .hoia import build as build_hoia

def build_dataset(image_set, args):
    """
    Args:
        image_set (str): 'train' or 'val'
        args : input args

    Returns:
        Dataset: appropriate Dataset object
    """
    if args.dataset_file == 'hico':
        return build_hico(image_set, args)
    if args.dataset_file == 'vcoco':
        return build_vcoco(image_set, args)
    if args.dataset_file == 'hoia':
        return build_hoia(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
