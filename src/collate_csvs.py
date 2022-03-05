import os
import sys

import pandas as pd
import numpy as np

from pandas.io.parsers import read_csv

import re
# from icecream import ic

kernel_extra_configs = {
    "copy": [""],  "direct": [""], "indirect": [""], 
    "overlapped": ["degree"], 
    "computational_intensity": ["comp-intens"], 
    "interleaved_copy": ["block_life","elements"],
    "interleaved_copy_full_life": ["elements"],
    "uncoalesced_reuse_general_single": ["preload", "avoid_bank_conflicts", "shuffle_size"]
    }

kernel_type_names = {
    "copy": "ArrayCopy",  
    "direct": "SimpleIndirectionTest_Direct", 
    "indirect":"SimpleIndirectionTest_Indirect", 
    "overlapped": "OverlappedIdxDataAccessKernel", 
    "computational_intensity": "ComputationalIntensity", 
    "interleaved_copy": "InterleavedCopy",
    "interleaved_copy_full_life": "InterleavedCopyFullLife",
    "uncoalesced_reuse_general_single": "UncoalescedReuseGeneralSingleElement"
    }





def collate_interleaved_copy(base_folder):
    
    main_df = pd.DataFrame()
    kernel = "interleaved_copy"

    configs = kernel_extra_configs[kernel]

    for filename in os.listdir(base_folder):
        if ".csv" not in filename: continue 
        if kernel not in filename: continue 
        post_fix = "_driver"
        data = pd.read_csv(base_folder + "/" + filename,header=0)
    

        kernel_type = filename[:filename.find(post_fix)]
        # print(filename[len(kernel_type) + len(post_fix)])

        has_extra_config = filename[len(kernel_type)-1].isdigit()

        # ic(filename[len(kernel_type)-1])
        if(has_extra_config):
            search_start = len(kernel_type) - 7
            possible_vals = [int(v[1:]) for v in re.findall(r'_\d+', filename[search_start:])]
            # print(possible_vals)
            # exit()
            assert(len(possible_vals) == len(configs))
            for i, cfg in enumerate(configs):
                data[cfg] = possible_vals[i]
            # kernel_type = kernel_type[:-int(np.log10(value))-2]
    
        main_df = main_df.append(data)
        
    # print(main_df)
    main_df.to_csv(os.path.join(base_folder, kernel+"_collated.csv"))

def main():
    base_folder = sys.argv[-1]
    if not os.path.exists(base_folder):
        ic(f"{base_folder} does not exist!")
    collate_interleaved_copy(base_folder)

if __name__ == '__main__':
    main()