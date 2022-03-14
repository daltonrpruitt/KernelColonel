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
field_strings = {
    "tpb": "Threads/Block",
    "occupancy": "Occupancy Fraction",
    "fraction_of_max_bandwidth": "Bandwidth Fraction",
    "min": "Min Exec Time (ms)",
    "throughput": "Bandwidth (GB/s)"
}

field_bounds = {
    "occupancy": [0.0, 1.1], 
    "fraction_of_max_bandwidth": [0.0, 1.1], 
    # "min": [0, -1],
    # "throughput": [0, -1]
}

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
    product_list = [i for i in product]
    ic(product_list)
    return product_list


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

    multi_plot_field_uniques = data[field_for_multiplotting].unique()

    colors = plt.cm.rainbow(np.linspace(0, 1, len(multi_plot_field_uniques)))

    for i, constants in enumerate(unique_combos):
        title_configs = ""
        filename_configs = ""
        for j in range(len(fields_to_keep_constant)):
            title_configs += fields_to_keep_constant[j]+"="+str(constants[j]) + (", " if j < len(fields_to_keep_constant) - 1 else "")
            filename_configs += fields_to_keep_constant[j]+"-"+str(constants[j]) + ("_" if j < len(fields_to_keep_constant) - 1 else "")
        filename = filename_base + "_" + filename_configs
        ic(title_configs)

        plt.close("all")
        
        # Filter data to only that matching the corrent constants
        multi_local_data = data.copy(deep=False)
        for j in range(len(fields_to_keep_constant)):
            multi_local_data = multi_local_data.loc[multi_local_data[fields_to_keep_constant[j]] == constants[j]]
    
        # Work through and plot each unique multiplot trend
        for j, individual_val in enumerate(multi_plot_field_uniques): 
            print(individual_val)

            fig = plt.figure()
            fig.suptitle(f"{plot_title_base} ({date_time_str})\n{title_configs}", fontsize=11)
            plt.xlabel(field_strings[x_field] if x_field in field_strings.keys() else x_field)
            plt.ylabel(field_strings[y_field] if y_field in field_strings.keys() else y_field)

            if x_field in field_bounds: plt.xlim(field_bounds[x_field])
            if y_field in field_bounds: plt.ylim(field_bounds[y_field])

            plot_local_data = multi_local_data.loc[multi_local_data[field_for_multiplotting] == individual_val]
            
            ic(plot_local_data)
            plt.plot(plot_local_data[x_field], 
                     plot_local_data[y_field], 
                     c=colors[j], 
                     marker="o",
                     label=(
                        (field_strings[field_for_multiplotting] if field_for_multiplotting in field_strings.keys() else field_for_multiplotting )
                        +"="+str(individual_val)
                        )
            )
    
        plt.legend(loc="best")
        plt.savefig(os.path.join(images_dir, filename) + ".png")
        plt.close()

for p in plot_configs:
    kernel = p[0]
    collated_file = collate_csv(base_folder, kernel)
    if collated_file is None:
        print("Could not collate", kernel, "in" ,base_folder, "!")
        exit(1)
    data = read_csv(collated_file)
    plot_general(data, p[0], p[1], p[2], p[3], p[4], p[5], p[6])
