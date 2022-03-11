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
from collate_csvs import collate_csv

debug = True 
if not debug: 
    ic.disable()

#assuming current folder only
base_folder = sys.argv[1] 

if "output" not in os.path.abspath(base_folder):
    print("Error: Base directory to process must be within the 'output' directory!")
    exit(-1)

# Field names for reference
# kernel_type,array_size,tpb,occupancy,min,med,max,avg,stddev,throughput,fraction_of_max_bandwidth

date_time_str = base_folder[base_folder.find("2022"):]
ic(date_time_str)

kxc = kernel_extra_configs

uncoal = "uncoalesced_reuse_gen_single_ilp"

plot_configs = [
    ["uncoalesced_reuse_gen_single_ilp", "occupancy", "fraction_of_max_bandwidth", 
            kxc[uncoal][:-1], kxc[uncoal][-1], "uncoal_ilp", "Uncoalesced BW vs Occup."]
]

def plot_general(all_data, kernel_name, x_field, y_field, fields_to_keep_constant, 
                field_for_multiplotting, filename_base, plot_title_base):
    data = all_data.loc[all_data["kernel_type"] == kernel_type_names[kernel_name]]

    uniques = []
    for f in fields_to_keep_constant: 
        uniques.append(data[f].unique())
    product = itertools.product(*uniques)
    
    for element in product:
        ic(element)

for p in plot_configs:
    kernel = p[0]
    collated_file = collate_csv(base_folder, kernel)
    if collated_file is None:
        print("Could not collate", kernel, "in" ,base_folder, "!")
        exit(1)
    data = read_csv(collated_file)
    plot_general(data, p[0], p[1], p[2], p[3], p[4], p[5], p[6])
