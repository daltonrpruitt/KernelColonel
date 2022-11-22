## TODO.txt

- [ ] setup development environment on Windows to use WSL2 and containers(?) to use linux native stuff 
    - instead of trying to do cross-platform support for no reason other than my edification...
- [ ] add more tests for IKernelData, such as:
    - [ ] passing to actual kernel
    - [ ] 

- [ ] Figure out way to test for memory leaks for KernelData tests (need way to confirm destructor was correct?)
    - use some kind of dynamic "size of" thing?
    - or just use address sanitizer when on Linux!!






Done: 

- [x] fix debugging symbols not being in outputted code
    - made an attempt at copying the set build type into the `tests` subfolder directly, but did not help
    - solution: use the actual visual studio debugger built into VS Code, I think
