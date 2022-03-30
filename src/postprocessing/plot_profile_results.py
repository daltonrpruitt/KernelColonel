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
from node_postprocessing_info import *
from collate_csvs import collate_csv

debug = True 
if not debug: 
    ic.disable()

#assuming current folder only
data_file = sys.argv[1] 

if "output" not in os.path.abspath(data_file):
    print("Error: File to process must be within the 'output' directory!")
    exit(-1)

parent_dir = os.path.dirname(data_file)
ic(parent_dir)

images_dir = os.path.join(parent_dir, "images")
if not os.path.exists(images_dir):
    os.makedirs(images_dir)


date_time_str = data_file[data_file.find("2022"):]
ic(date_time_str)

node_start_idx = data_file.find("output")+len("output/")
node_str = data_file[node_start_idx:node_start_idx+data_file[node_start_idx:].find(os.path.sep)]
architecture_str = node_gpu_info[node_str]
ic(architecture_str)


# Field names for reference
# kernel_type,array_size,tpb,occupancy,min,med,max,avg,stddev,throughput,fraction_of_max_bandwidth
field_strings = {
}

field_bounds = {
}

fields_for_logscale = [
]


# Reference for inputs...
# kernel_name, x_field, y_field, 
#   fields_to_keep_constant, field_for_multiplotting, filename_base, plot_title_base
plot_configs = [
    ["indirect_copy", "occupancy", "fraction_of_max_bandwidth",
            ["ILP","shuffle_size"], "access_pattern", "indirect", "Indirect BW vs Occup."],
    
]

def get_config_combos(d, fields):
    uniques = []
    for f in fields: 
        uniques.append(data[f].unique())
    product = itertools.product(*uniques)
    product_list = [i for i in product]
    ic(fields,product_list)
    return product_list


def plot_general(all_data, kernel_name, x_field, y_field, fields_to_keep_constant, 
                fields_for_multiplotting, filename_base, plot_title_base):
    data = all_data.loc[all_data["kernel_type"] == kernel_class_names[kernel_name]]
    
    if type(fields_to_keep_constant) is str: fields_to_keep_constant = [fields_to_keep_constant] 
    if type(fields_for_multiplotting) is str: fields_for_multiplotting = [fields_for_multiplotting] 

    if len(data) < 5:
        print(kernel_name, "does not have enough data to make useful plots!")
        return

    unique_combos = None
    # if type(fields_to_keep_constant) is list: 
    unique_combos = get_config_combos(data, fields_to_keep_constant)
    # elif type(fields_to_keep_constant) is str:
    #     unique_combos = data[fields_to_keep_constant].unique()
    # else:                                
    #     raise TypeError("Invalid type for fields to keep constant: "+str(type(fields_to_keep_constant)))

    if len(unique_combos) > 100: 
        print("Error: Plotting", kernel_name, "for combos of", fields_to_keep_constant, "would generate", len(unique_combos), "unique plots.")
        print("    Not continuing!")
        return
        
    elif len(unique_combos) > 32: 
        print("Warning: Plotting", kernel_name, "with current configuration is about to generate", len(unique_combos), "unique plots...")
        print("Continue?...(y/n)", end="")
        if input().lower() != "y": 
            print("Stopping current plotting...")
            return

    kernel_class_name = kernel_class_names[kernel_name]
    config_names = kernel_extra_configs[kernel_name]



    for i, constants in enumerate(unique_combos):
        title_configs = ""
        filename_configs = ""
        if len(constants) == 1 and len(unique_combos) == 1:
            ic("Only need 1 chart; not adding config info to title or filename.")
        else:
            for j in range(len(fields_to_keep_constant)):
                title_configs += fields_to_keep_constant[j]+"="+str(constants[j]) + (", " if j < len(fields_to_keep_constant) - 1 else "")
                filename_configs += fields_to_keep_constant[j]+"-"+str(constants[j]) + ("_" if j < len(fields_to_keep_constant) - 1 else "")
        filename = f"{architecture_str}_{filename_base}" + (f"_{filename_configs}" if filename_configs != "" else "")
        ic(title_configs)
        
        plt.close("all")
        
        # Filter data to only that matching the corrent constants
        multi_local_data = data.copy(deep=False)
        for j in range(len(fields_to_keep_constant)):
            multi_local_data = multi_local_data.loc[multi_local_data[fields_to_keep_constant[j]] == constants[j]]
        if len(multi_local_data) == 0: 
            print(f"Configuration of {title_configs} does not have enough data to plot!")
            continue
        elif len(multi_local_data) < 2:
            print(f"Configuration of {title_configs} only has {len(multi_local_data)} values to plot!")
            print("Plot anyway? (y/n)")
            inpt = input()
            if inpt.lower() != "y": continue


        multi_plot_field_unique_combos = None
        
        # if type(fields_for_multiplotting) is list:
        if len(fields_for_multiplotting) == 1:
            multi_local_data.sort_values(fields_for_multiplotting[0], inplace=True, kind="stable")
            tmp_list = multi_local_data[fields_for_multiplotting[0]].unique()
            multi_plot_field_unique_combos = [ [i] for i in tmp_list]
        else: 
            multi_plot_field_unique_combos = get_config_combos(multi_local_data, fields_for_multiplotting)
        # elif type(fields_for_multiplotting) is str:
        #     multi_plot_field_unique_combos = multi_local_data[fields_for_multiplotting].unique()
        # else:                                
        #     raise TypeError("Invalid type for multiplot fields: "+str(type(fields_for_multiplotting)))

        colors = plt.cm.rainbow(np.linspace(0, 1, len(multi_plot_field_unique_combos)))

        plt.figure()
        subtitle = f"Arch={architecture_str} | {date_time_str} \n{title_configs}"
        plt.suptitle(plot_title_base, fontsize=11)
        plt.title(subtitle, fontsize=9, y=1)
        plt.xlabel(field_strings[x_field] if x_field in field_strings.keys() else x_field)
        plt.ylabel(field_strings[y_field] if y_field in field_strings.keys() else y_field)

        if x_field in field_bounds: plt.xlim(field_bounds[x_field])
        if y_field in field_bounds: plt.ylim(field_bounds[y_field])

        if x_field in fields_for_logscale: plt.xscale("log", basex=2)
        if y_field in fields_for_logscale: plt.yscale("log", basey=2)

        # Work through and plot each unique multiplot trend
        for j, multi_plot_cur_vals in enumerate(multi_plot_field_unique_combos): 
            ic(multi_plot_cur_vals)


            # plot_local_data = multi_local_data.loc[multi_local_data[fields_for_multiplotting[]] == multi_plot_cur_vals]
            plot_local_data = multi_local_data.copy(deep=False)
            label_str = ""
            for k, field in enumerate(fields_for_multiplotting):
                val = multi_plot_cur_vals[k]
                plot_local_data = (plot_local_data.loc[plot_local_data[field] == val]).sort_values(x_field, kind="stable")
                label_str += field_strings[field] if field in field_strings else field
                label_str += "="
                t = type(val)
                if t in (bool, np.bool_):
                    label_str += "T" if val else "F"
                elif t in (int, np.int32, np.int64, np.float32, np.float64, str):
                    label_str += str(val)
                else: 
                    raise TypeError("Invalid type for multiplot value: "+str(t))
                if k < len(fields_for_multiplotting) -1: label_str += " | "

            ic(plot_local_data)
            plt.plot(plot_local_data[x_field], 
                     plot_local_data[y_field], 
                     c=colors[j], 
                     marker="o",
                     label=label_str
            )
    
        plt.legend(loc="best", fontsize=6)
        plt.savefig(os.path.join(images_dir, filename) + ".png")
        plt.close()

for p in plot_configs:
    kernel = p[0]
    data = read_csv(data_file)
    ic(data)
    break
    plot_general(data, p[0], p[1], p[2], p[3], p[4], p[5], p[6])
