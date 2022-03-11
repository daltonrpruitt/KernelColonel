'''
kernel_processing_info.py

Simple file to contain current list of active test kernels in one place. May
combine collating and plotting files, or distribute their functionality further,
but this single location for the info is better than keeping multiple places
current.

'''

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
