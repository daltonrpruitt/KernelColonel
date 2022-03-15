'''
kernel_processing_info.py

Simple file to contain current list of active test kernels in one place. May
combine collating and plotting files, or distribute their functionality further,
but this single location for the info is better than keeping multiple places
current.

'''

debug = False

try:
    from icecream import ic
except Exception as e: 
    print("Could not import 'icecream':", e)
    def ic(*args, **keyword_args):
        print(**locals())

kernel_config_info = [
    ["copy",                                "ArrayCopy"],  
    ["direct",                              "SimpleIndirectionTest_Direct"], 
    ["indirect",                            "SimpleIndirectionTest_Indirect"], 
    # ["overlapped",                          "OverlappedIdxDataAccessKernel", "degree"], 
    # ["computational_intensity",             "ComputationalIntensity", "comp-intens"], 
    # ]"interleaved_copy",                    "InterleavedCopy", "block_life","elements"],
    # ]"interleaved_copy_full_life",          "InterleavedCopyFullLife", "elements"],
    ["interleaved_fl_ilp",                  "InterleavedFullLifeILP", "elements", "ILP"],
    ["uncoalesced_reuse_general_single",    "UncoalescedReuseGeneralSingleElement", 
                                                "preload", "avoid_bank_conflicts", "shuffle_size"],
    ["uncoalesced_reuse_gen_single_ilp",    "UncoalescedReuseGenSingleILP", 
                                                "preload", "avoid_bank_conflicts", "shuffle_size", "ILP"]
]

kernel_class_names = {k[0]: k[1] for k in kernel_config_info}
kernel_extra_configs = {k[0]: "" for k in kernel_config_info}
for i, k in enumerate(kernel_extra_configs.keys()):
    assert k == kernel_config_info[i][0]
    extras = kernel_config_info[i][2:]
    if len(extras) > 0:
        kernel_extra_configs[k] = extras
ic(kernel_class_names)
ic(kernel_extra_configs)


data_field_plot_strings = {
    "occupancy": "Occup",
    "fraction_of_max_bandwidth": "BW Frac"
}

def sort_dict(d):
    sorted_ = {}
    for k in sorted(d, key=len, reverse=True):
        sorted_[k] = d[k]
    return sorted_

sorted_kernel_type_names = sort_dict(kernel_class_names)

def get_specific_kernel_type(full_string):
    for name in sorted_kernel_type_names.keys():
        if name in full_string:
            if debug: ic(full_string, name)
            return name
    return None