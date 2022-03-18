### nvprof for older systems?
nvprof --csv --kernels "*" --analysis-metrics --log-file nvprof_output.csv $(./build/debug/main &> debug.txt)



### Nsight Compute for V100/A100
ncu --list-sets > ncu_sets_avail.txt
ncu --list-sections > ncu_sections_avail.txt

Not sure if will work or not!
ncu --set detailed --csv -o ncu_profile.csv --kernel-id ::*:"3|4|5" ./build/debug/main &> debug.txt