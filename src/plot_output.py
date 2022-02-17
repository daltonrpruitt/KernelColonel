#!/apps/python-3.9.2/bin/python3
"""
Plot data output from benchmark_cache_reuse.cu
"""

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

#assuming current folder only
base_folder = sys.argv[1] 

# if len(sys.argv) == 1: 
#     print("Error: Must provide a benchmark version name!")
#     exit(-1)
# bm_version_name = sys.argv[-1]

if "output" not in os.path.abspath(base_folder):
    print("Error: Base directory to process must be within the 'output' directory!")
    exit(-1)

kernel_extra_configs = {"copy": "",  "direct": "", "indirect":"", "overlapped": "degree", "computational_intensity": "comp-intens"}

data_headers   = ["kernel_type", "array_size", "tpb", "occupancy", "min", "med", "max", "avg", "stddev"]

if False:
    data_types = [int, int, int, int, 
              int, int, float, int, 
              float, str, str, int, 
              float, float, float, float, 
              float, float]
    data_types_dict = dict(zip(data_headers, data_types))


kernel_types = []
main_df = pd.DataFrame()
for filename in os.listdir(base_folder):
    post_fix = "_kernel_output"

    data = pd.read_csv(base_folder + "/" + filename,header=0)
    kernel_type = filename[:filename.find(post_fix)]
    # print(filename[len(kernel_type) + len(post_fix)])

    has_extra_config = filename[len(kernel_type)-1].isdigit()
    # ic(filename[len(kernel_type)-1])
    if(has_extra_config):
        search_start = len(kernel_type) - 5
        possible_vals = [int(v[1:-1]) for v in re.findall(r'_\d+_', filename[search_start:search_start+10])]
        value = possible_vals[0]
        kernel_type = kernel_type[:-int(np.log10(value))-2]
        data[kernel_extra_configs[kernel_type]] = value
    
    # ic(data.head())
    # if main_df is None: main_df = data
    # else: 
    main_df = main_df.append(data)
    # ic(main_df)

ic(main_df)

filename = base_folder + ("" if base_folder[-1]=="/" else "/") +  "overlapped_kernel_output_1.csv"
try:
    main_df = pd.read_csv(filename,header=0)
    print(main_df.head())
except Exception as e:
    print(e)
    exit(-1)
    small_dfs = [] # https://stackoverflow.com/a/56746204 Following this advice not because I need performance, but it may be good?

    one_processed = False

    for subdir, dirs, files in os.walk(base_folder):
        for file in files:
            if ".csv" in file and not "collated" in file:
                # try:
                if len(open(os.path.join(subdir, file),'r').readlines()) < 3:
                    print(os.path.join(subdir, file),"recorded an error (more than likely...)")    
                    continue
                # except FileNotFoundError:
                #     print(os.path.join(subdir, file), "does not exist!")
                print("Processing:", os.path.join(subdir, file))
                
                current_df = pd.read_csv(os.path.join(subdir, file), header=0, dtype=data_types_dict)
                
                        
                # print(current_df.values.tolist()[:10])
                # print(type(current_df.values.tolist()))
                if len(small_dfs) == 0:
                    small_dfs = current_df.values.tolist()
                else:
                    small_dfs += current_df.values.tolist()
                # print(current_df.iloc[:])
        #         one_processed = True
        #         break 
        # if one_processed:
        #     break

    # print(small_dfs)
    main_df = pd.DataFrame(small_dfs, columns=data_headers)
    main_df.to_csv("collated_data.csv", index=False)

print("Entries in", filename,"=", len(main_df))
# exit(0)

# Plotting 
plt.close("all")

occupancy = main_df["occupancy"]
print(occupancy)

fig = plt.figure(num=0) 
fig.suptitle(filename, fontsize=11)

plt.plot(main_df["occupancy"], main_df["min"], c='b', marker="o", label="min")
plt.show()

exit(0)
independent_vars = ['bwss', 'tpb', 'num_repeat']
dependent_vars = ['fraction_of_l2_used_per_block', 'min', 'avg', 'stddev', 'achieved_throughput']
show_versions_on_same = ['kernel_type']
copies_to_make = ['shuffle_type']

kernel_types = ['Direct-Same', 'Indirect-Same','Direct-Shift', 'Indirect-Shift']
shuffle_types = ['same_indices', 'random_indices']

shuffle_kernel_combos = {"same_indices": kernel_types, "random_indices": kernel_types[1::2]}

for shuffle_kernel_combo_idx in range(len(shuffle_kernel_combos)):
    fig_num = 1
    shuffle = list(shuffle_kernel_combos.keys())[shuffle_kernel_combo_idx]
    for indep_var in ["bwss"]: # independent_vars:
        indep_var_idx = independent_vars.index(indep_var)
        other_independent_vars = independent_vars.copy()
        other_independent_vars.remove(indep_var)

        unique_other_vars_vals = []
        for i in range(2):
            unique_other_vars_vals.append(list(main_df[other_independent_vars[i]].unique())) # hack; will need to remove the indexing here
            unique_other_vars_vals[i].sort()

        for unique_other_var_1_val in unique_other_vars_vals[0]:
            for unique_other_var_2_val in unique_other_vars_vals[1]:
                print(other_independent_vars[0],"=",unique_other_var_1_val,other_independent_vars[1],"=",unique_other_var_2_val)
                data = main_df.loc[( (main_df["shuffle_type"]==shuffle) & 
                                     (main_df[other_independent_vars[0]]==unique_other_var_1_val) &
                                     (main_df[other_independent_vars[1]]==unique_other_var_2_val),
                                    ["shuffle_type","kernel_type",indep_var]+other_independent_vars+dependent_vars)]
                data.sort_values([indep_var], inplace=True)

                title = "Cache Reuse Benchmark (" +bm_version_name+ "): execution time as a function of \n" + \
                        indep_var + " for shuffle=" + shuffle + " with tpb="+str(unique_other_var_1_val) + \
                        " and repeat="+str(unique_other_var_2_val)

                print(title)
                fig , ax1 = plt.subplots(num=fig_num) 
                fig.suptitle(title, fontsize=11)

                max_time = data["avg"].max()
                max_throughput = data["achieved_throughput"].max()

                ax1.set_xscale("log",base=2)
                if indep_var == "bwss":
                    ax1.set_xlim(2**9, 2**18)
                else:
                    ax1.set_xlim(1/2, 64)
                # ax1.margins(0.2, 0.2)
                ax1.set_ylim(0, round(max_time +1,ndigits=0))
                ax1.set_yscale("linear")

                ax1.set_xlabel(indep_var, fontsize=7)
                ax1.set_ylabel("avg"+" exec time (ms)", fontsize=8)

                ax2 = ax1.twinx()
                ax2.set_yscale("linear")
                ax2.set_ylim(1,max(round(max_throughput+5),100))
                _, max_ = ax2.get_ylim()
                ax2.yaxis.set_ticks(np.arange(0, max_, 25.0))
                ax2.set_ylabel('throughtput(GB/s), %L2 used total')
                lns = []
                # This could be made more abstract, but doing so was not worth it currently (with only two kernels considered)
                colors = plt.cm.rainbow(np.linspace(0, 1, len(shuffle_kernel_combos[shuffle])))

                for k_idx in range(len(shuffle_kernel_combos[shuffle])):
                    kernel = shuffle_kernel_combos[shuffle][k_idx]
                    kernel_data = data.loc[(data["kernel_type"] == kernel)]
                    lns  += ax1.plot(kernel_data[indep_var], kernel_data["avg"], c=colors[k_idx], marker="o", label=kernel + " avg")
                    # fig.legend(loc="upper right")
                    # secondary y axis for blocks_per_sm values
                    #lns += 
                    ax2.plot(kernel_data[indep_var], kernel_data["fraction_of_l2_used_per_block"]*100*num_blocks, c=colors[k_idx], marker="x", label=kernel+" %L2 total")
                    #lns += 
                    ax2.plot(kernel_data[indep_var], kernel_data["achieved_throughput"], c=colors[k_idx], marker="+", label=kernel+" throughput")

                # https://stackoverflow.com/questions/5484922/secondary-axis-with-twinx-how-to-add-to-legend
                labs = [l.get_label() for l in lns]
                ax1.legend(lns, labs, loc=0, prop={"size":10}) 
                ax1.text(ax1.get_xlim()[0]* 1.25, ax1.get_ylim()[1]/10,"x - %L2\n+ - Throughput",fontsize=10,bbox={'facecolor': 'white', 'alpha': 0.1, 'pad': 10})
                fig.savefig(base_folder+"/"+"shuffle-"+shuffle+"_"+other_independent_vars[0]+"-"+str(unique_other_var_1_val)+
                            "_"+other_independent_vars[1]+"-"+str(unique_other_var_2_val)+ "_figure.png")  
                plt.close(fig_num)
                fig_num += 1
                print("-----------------------------------")

    # https://stackoverflow.com/questions/9834452/how-do-i-make-a-single-legend-for-many-subplots-with-matplotlib
    # lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    # lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

    # fig.legend(lines[:2], labels[:2], loc="lower left")


# plt.show()
