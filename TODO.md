## TODO.txt

- [ ] Figure out way to test for memory leaks for KernelData tests (need way to confirm destructor was correct?)
    - use some kind of dynamic "size of" thing?







Done: 

- [x] fix debugging symbols not being in outputted code
    - made an attempt at copying the set build type into the `tests` subfolder directly, but did not help
    - solution: use the actual visual studio debugger built into VS Code, I think
