import os, sys
import itertools

os.chdir(os.path.dirname(sys.argv[0]))

kernels = ["ArrayCopyContext", "SimpleIndirectionKernel"]
files = ["copy.cu", "simple_indirection.cu"]

value_types = ["float", "double", "int"]
index_types = ["int", "short"]
bool_types = ["true", "false"]

template_args_config = {
    "ArrayCopyContext" : {
        "vt": value_types,
        "it": index_types 
        },
    "SimpleIndirectionKernel": {
        "vt": value_types,
        "it": index_types,
        "is_indirect": bool_types
    }
}



with open("kernel_types.h", "w") as file:
    file.write("#pragma once\n")
    for f in files:
        file.write("#include <"+f+">\n")

    for k in kernels:
        template_args = [val_list for val_list in template_args_config[k].values()]

        possible_combinations = itertools.product(*template_args)
        for vals in possible_combinations:
            template_str = k+"<"
            type_name_str = k+"_"
            for i, v in enumerate(vals):
                template_str += v
                type_name_str += v
                if i < len(vals) - 1 :
                    template_str += ", "
                    type_name_str += "_"
            template_str += ">"
            type_name_str += "_t"

            file.write("typedef "+template_str + " " + type_name_str + ";\n")

