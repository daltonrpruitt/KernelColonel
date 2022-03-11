#!/apps/python-3.9.2/bin/python3
"""
General plotting functionality for GPGPU Microbenchmarks
"""

import itertools
import os
import sys
import datetime as dt
import types
import matplotlib
from matplotlib.transforms import Bbox
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from pandas.io.parsers import read_csv

import re
from icecream import ic

from kernel_postprocessing_info import *

debug = True 
if not debug: 
    ic.disable()

#assuming current folder only
base_folder = sys.argv[1] 

if "output" not in os.path.abspath(base_folder):
    print("Error: Base directory to process must be within the 'output' directory!")
    exit(-1)

