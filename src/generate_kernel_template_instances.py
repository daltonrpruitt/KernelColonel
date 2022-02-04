import os, sys

os.chdir(os.path.dirname(sys.argv[0]))

kernels = ["ArrayCopyContext"]
files = ["copy.cu"]



value_types = ["float", "double", "int"]
index_types = ["int", "short"]


with open("kernel_types.h", "w") as file:
    file.write("#pragma once\n")
    for f in files:
        file.write("#include <"+f+">\n")

    for k in kernels:
        for vt in value_types:
            for it in index_types:
                file.write("typedef "+k+"<"+vt+", "+it+"> "+k+"_"+vt+"_"+it+"_t;\n")
