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

kernel_extra_configs = {"copy": "",  "direct": "", "indirect":"",
                        "overlapped": "degree", 
                        "computational_intensity": "comp-intens",
                        "interleaved_copy": ["block_life", "elements"] 
}

kernel_type_names = {"copy": "ArrayCopy",  
                    "direct": "SimpleIndirectionTest_Direct", 
                    "indirect":"SimpleIndirectionTest_Indirect", 
                    "overlapped": "OverlappedIdxDataAccessKernel", 
                    "computational_intensity": "ComputationalIntensity", 
                    "interleaved_copy":"InterleavedCopy"}


data_headers   = ["kernel_type", "array_size", "tpb", "occupancy", "min", "med", "max", "avg", "stddev"]

date_time_str = base_folder[base_folder.find("2022"):]
ic(date_time_str)

kernel_types = []
main_df = pd.DataFrame()
for filename in os.listdir(base_folder):
    if ".csv" not in filename: continue # ignore images and/or directories
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
main_df.reset_index(drop=True, inplace=True)  # make sure indexes pair with number of rows

ic(main_df)

images_dir = base_folder + "/images"
if not os.path.exists(images_dir):
    os.makedirs(images_dir)

full_filename = base_folder + ("" if base_folder[-1]=="/" else "/") +  "overlapped_kernel_output_1.csv"


print("Entries in all files in ", base_folder,"=", len(main_df))


def check_data(data):
    if len(data) == 0: 
        return False 
    elif len(data) < 5:
        ic("Not enough data!")
        return False
    

def process_comp_intens(df):
    kernel_name = kernel_type_names["computational_intensity"]
    data = df[df["kernel_type"]==kernel_name]
    config_name = kernel_extra_configs["computational_intensity"]
    
    if not check_data(data): return
    
    for intens in data[config_name].unique(): 
        print(intens)
        plt.close("all")

        fig = plt.figure() 
        fig.suptitle(kernel_name + " intensity=" +str(int(intens)) + " " + date_time_str, fontsize=11)
        plt.xlabel("Max Occupancy")
        plt.ylabel("Fraction of max Bandwidth")

        local_data = data[data[config_name] == intens]
        plt.plot(local_data["occupancy"], local_data["fraction_of_max_bandwidth"], c='b', marker="o", label="bandwidth fraction")
        # plt.show()
        plt.savefig(images_dir+"/"+kernel_name+"_"+config_name+"-"+str(int(intens))+".png")

def process_overlapped_access(df):
    kernel_name = kernel_type_names["overlapped"]
    data = df[df["kernel_type"]==kernel_name]
    config_name = kernel_extra_configs["overlapped"]

    if not check_data(data): return

    uniques = data[config_name].unique()
    colors = plt.cm.rainbow(np.linspace(0, 1, len(uniques)))

    plt.close("all")
    combined_fig = plt.figure(1)
    combined_fig.suptitle(kernel_name +" Combined " + date_time_str, fontsize=11)
    plt.xlabel("Max Occupancy")
    plt.ylabel("Fraction of max Bandwidth")
    
    for i, degree in enumerate(uniques): 
        print(degree)

        fig = plt.figure() 
        fig.suptitle(kernel_name + " intensity=" +str(int(degree)) + " " + date_time_str, fontsize=11)
        plt.xlabel("Max Occupancy")
        plt.ylabel("Fraction of max Bandwidth")

        local_data = data[data[config_name] == degree]
        plt.plot(local_data["occupancy"], local_data["fraction_of_max_bandwidth"], c='b', marker="o", label="bandwidth fraction")
        # plt.show()
        plt.savefig(images_dir+"/"+kernel_name+"_"+config_name+"-"+str(int(degree))+".png")
        plt.close()
        combined_fig = plt.figure(1)
        plt.plot(local_data["occupancy"], local_data["fraction_of_max_bandwidth"], c=colors[i], marker="o", label="degree="+str(int(degree)))
    combined_fig = plt.figure(1)
    plt.legend(loc="best")

    plt.savefig(images_dir+"/"+kernel_name+"_combined.png")

def process_interleaved_kernel(df):
    kernel_name = kernel_type_names["interleaved"]
    data = df[df["kernel_type"]==kernel_name]
    config_names = kernel_extra_configs["interleaved"]

    if not check_data(data): return

    assert(len(config_names) == 2)
    for i, config in enumerate(config_names):
        uniques = data[config].unique()
        other_config_counts = {}
        other_uniques = data[config_names[1-i]].unique()
        

        colors = plt.cm.rainbow(np.linspace(0, 1, len(uniques)))

        plt.close("all")
        combined_fig = plt.figure(1)
        combined_fig.suptitle(kernel_name +" Combined " + date_time_str, fontsize=11)
        plt.xlabel("Max Occupancy")
        plt.ylabel("Fraction of max Bandwidth")
    
        for i, degree in enumerate(uniques): 
            print(degree)

            fig = plt.figure() 
            fig.suptitle(kernel_name + " intensity=" +str(int(degree)) + " " + date_time_str, fontsize=11)
            plt.xlabel("Max Occupancy")
            plt.ylabel("Fraction of max Bandwidth")
            
            local_data = data[data[config_name] == degree]
            plt.plot(local_data["occupancy"], local_data["fraction_of_max_bandwidth"], c='b', marker="o", label="bandwidth fraction")
            # plt.show()
            plt.savefig(images_dir+"/"+kernel_name+"_"+config_name+"-"+str(int(degree))+".png")
            plt.close()
            combined_fig = plt.figure(1)
            plt.plot(local_data["occupancy"], local_data["fraction_of_max_bandwidth"], c=colors[i], marker="o", label="degree="+str(int(degree)))
            combined_fig = plt.figure(1)
            plt.legend(loc="best")
        
        plt.savefig(images_dir+"/"+kernel_name+"_combined.png")


process_comp_intens(main_df)
process_overlapped_access(main_df)
process_interleaved_kernel(main_df)

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
