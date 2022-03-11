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
    if not debug: 
        ic.disable()
except Exception as e: 
    print("Could not import 'icecream':", e)
    def ic(*args, **keyword_args):
        if debug:
            print(**locals())

kernel_extra_configs = {
    "copy": [""],  "direct": [""], "indirect": [""], 
    "overlapped": ["degree"], 
    "computational_intensity": ["comp-intens"], 
    "interleaved_copy": ["block_life","elements"],
    "interleaved_copy_full_life": ["elements"],
    "interleaved_fl_ilp": ["elements", "ILP"],
    "uncoalesced_reuse_general_single": ["preload", "avoid_bank_conflicts", "shuffle_size"],
    "uncoalesced_reuse_gen_single_ilp": ["preload", "avoid_bank_conflicts", "shuffle_size", "ILP"]
    }

kernel_type_names = {
    "copy": "ArrayCopy",  
    "direct": "SimpleIndirectionTest_Direct", 
    "indirect":"SimpleIndirectionTest_Indirect", 
    "overlapped": "OverlappedIdxDataAccessKernel", 
    "computational_intensity": "ComputationalIntensity", 
    "interleaved_copy": "InterleavedCopy",
    "interleaved_copy_full_life": "InterleavedCopyFullLife",
    "interleaved_fl_ilp": "InterleavedFullLifeILP",
    "uncoalesced_reuse_general_single": "UncoalescedReuseGeneralSingleElement",
    "uncoalesced_reuse_gen_single_ilp": "UncoalescedReuseGenSingleILP"
    }

data_field_plot_strings = {
    "occupancy": "Occup",
    "fraction_of_max_bandwidth": "BW Frac"
}

def sort_dict(d):
    sorted_ = {}
    for k in sorted(d, key=len, reverse=True):
        sorted_[k] = d[k]

sorted_kernel_type_names = sort_dict(kernel_type_names)

def get_specific_kernel_type(full_string):
    for name in sorted_kernel_type_names.keys():
        if name in full_string:
            ic(full_string, name)
            return name