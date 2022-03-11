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

images_dir = os.path.join(base_folder, "images")
if not os.path.exists(images_dir):
    os.makedirs(images_dir)


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

def get_config_combos(d, fields):
    uniques = []
    for f in fields: 
        uniques.append(data[f].unique())
    product = itertools.product(*uniques)
    ic(product)
    return product


def plot_general(all_data, kernel_name, x_field, y_field, fields_to_keep_constant, 
                field_for_multiplotting, filename_base, plot_title_base):
    data = all_data.loc[all_data["kernel_type"] == kernel_class_names[kernel_name]]
    
    if len(data) < 5:
        print(kernel_name, "does not have enough data to make useful plots!")
        return

    unique_combos = get_config_combos(data, fields_to_keep_constant)
    if len(unique_combos) > 32: 
        print("Warning: Plotting", kernel_name, "with current configuration is about to generate", len(unique_combos), "unique plots...")
        print("Continue?...(y/n)", end="")
        if input().lower() != "y": 
            print("Stopping current plotting...")
            return


for p in plot_configs:
    kernel = p[0]
    collated_file = collate_csv(base_folder, kernel)
    if collated_file is None:
        print("Could not collate", kernel, "in" ,base_folder, "!")
        exit(1)
    data = read_csv(collated_file)
    plot_general(data, p[0], p[1], p[2], p[3], p[4], p[5], p[6])
