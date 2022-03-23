import os
import sys

import pandas as pd
import numpy as np

from pandas.io.parsers import read_csv

import re
# from icecream import ic

from kernel_postprocessing_info import *


def collate_csv(base_folder, kernel):
    
    main_df = pd.DataFrame()

    configs = kernel_extra_configs[kernel]

    # First pass for collated files
    for filename in os.listdir(base_folder):
        if ".csv" not in filename: continue
        kernel_type = get_specific_kernel_type(filename)
        if kernel_type != kernel: continue
        if "collated" in filename:
            print("Collated file for", kernel,"already exists in",base_folder,"!")
            collated = os.path.join(base_folder, filename)
            return collated

    for filename in os.listdir(base_folder):
        if ".csv" not in filename: continue
        kernel_type = get_specific_kernel_type(filename)
        if kernel_type != kernel: continue
        if "collated" in filename: continue
        post_fix = "_driver.csv"
        data = pd.read_csv(base_folder + "/" + filename,header=0)
    
        has_configs = True
        for cfg in configs:
            if cfg not in data.columns:
                has_configs = False
                break
        
        if not has_configs:
            print(filename, "does not have correct configs in contents!\nProcessing filenames for configs...")
            
            kernel_and_configs = filename[:filename.find(post_fix)]
            # print(filename[len(kernel_type) + len(post_fix)])

            has_extra_config = len(kernel_and_configs) - len(kernel) > 1 # filename[len(kernel_type)-1].isdigit()
            # print(kernel_and_configs, filename[filename.find(post_fix):])
            # ic(filename[len(kernel_and_configs)-1])
            if(has_extra_config):
                search_start = len(kernel)
                possible_vals = [v[1:] for v in re.findall(r'(_\d+|_[^_]+)', kernel_and_configs[search_start:])]
                # print(possible_vals)
                # exit()
                assert(len(possible_vals) == len(configs))
                for i, cfg in enumerate(configs):
                    val = None
                    if possible_vals[i] == "true": val = True
                    elif possible_vals[i] == "false": val = False
                    elif possible_vals[i].isdigit(): val = int(possible_vals[i])
                    data[cfg] = val
                # kernel_and_configs = kernel_and_configs[:-int(np.log10(value))-2]
                
        main_df = main_df.append(data)

    if(len(main_df) == 0):
        print("Could not find any data for ", kernel,"!", sep="")
        return ""

    # print(main_df)
    new_filename = os.path.join(base_folder, kernel+"_collated.csv")
    main_df.to_csv(new_filename, index=False)
    return new_filename

def main():
    base_folder = sys.argv[-1]
    if not os.path.exists(base_folder):
        ic(f"{base_folder} does not exist!")
    collate_csv(base_folder, "interleaved_copy_full_life")
    collate_csv(base_folder, "uncoalesced_reuse_general_single")

if __name__ == '__main__':
    main()