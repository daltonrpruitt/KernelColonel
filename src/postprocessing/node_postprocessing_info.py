'''
node_post_processing_info.py

Simple file to contain current list of node names with 

'''
from fileinput import filename
import pandas as pd

import os
from icecream import ic

debug = True
if not debug: ic.disable()

s = os.getcwd().find('gpgpu_microbenchmarks')

posprocessing_dir = os.path.join(os.getcwd()[:s], "gpgpu_microbenchmarks","src","postprocessing")

node_gpu_info = {}

# Reference: node_name,architecture
node_arch_filename = os.path.join(posprocessing_dir, "node_arch_translation.csv")
try:
    data = pd.read_csv(node_arch_filename)
    data = data.reset_index()

    for index, row in data.iterrows():    
        node_gpu_info[row["node_name"]] = row["architecture"]

    ic(node_gpu_info)
except FileNotFoundError as e:
    print("Node/Architecture CSV file does not exist! :", e)
    print("The file should be similar to this:")
    print("#"*(20+len(node_arch_filename)))
    print("    Filename:", node_arch_filename)
    print("-"*(20+len(node_arch_filename)))
    print("\tnode_name,architecture")
    print("\tscout,K20m")
    print("\tyour_desktop_node_name,your_desktop_gpu_architecture")
    print("\tother_node_name,other_node_gpu_architecture")
    print("#"*(20+len(node_arch_filename)))
    exit(-1)
