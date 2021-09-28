#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Fastext Common Crawl vectors, e.g. use with filename
"models:fasttext_fi_vectors/crawl-300d-fi.vec".
"""

import os
import torchtext.vocab as vocab

URL = 'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fi.300.vec.gz'


def download(datapath):
    return vocab.Vectors(
        name='cc.fi.300.vec.gz',
        url=URL,
        cache=os.path.join(datapath, 'models', 'fasttext_fi_vectors'),
    )
