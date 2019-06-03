#!/usr/bin/env python
import logging
import os
import sys
import argparse

parser = argparse.ArgumentParser(description='test code')
parser.add_argument('path', type=str, help='path to test file/dir')
parser.add_argument('-s', action='store_true', default=False, help='If specified, output will be logged.')
args = parser.parse_args()

LEVEL = logging.INFO
from allennlp.common.util import import_submodules

extra_libraries = ["lib"]
for library in extra_libraries:
    import_submodules(library)

import pytest

argv = ['-x', args.path]
if args.s:
	argv.append('-s')

pytest.main(argv)
