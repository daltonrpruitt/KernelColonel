### nvprof for older systems?
$ nvprof --csv --kernels "*" --analysis-metrics --log-file nvprof_output.csv $(./build/debug/main &> debug.txt)



### Nsight Compute for V100/A100
$ ncu --list-sets > ncu_sets_avail.txt
$ ncu --list-sections > ncu_sections_avail.txt

This command profiles every kernel (be carefule) and  generates the ncu-rep type file for output
$ ncu --set detailed -o ncu_profile ./build/debug/main &> debug.txt