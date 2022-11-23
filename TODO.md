## TODO.txt

* [ ] separate IKernelData into two files: header and implementation
  * [ ] Figure out how to make the library of the implemenation via the CMakeLists.txt 
    * (temporarily comment out old stuff)
  * [ ] Get new library linking into tests


- [ ] add more tests for IKernelData, such as:
    - [x] passing to actual kernel
    - [ ] Initing with device that doesn't exist (using cudaGetDeviceCount...)

- [ ] Figure out way to test for memory leaks for KernelData tests (need way to confirm destructor was correct?)
    - use some kind of dynamic "size of" thing?
    - or just use address sanitizer when on Linux!!


* [ ] Start implementing the design of IKernelContext
  * [ ] Add tests first
* [ ] Start implementing the design of IKernelDriver
  * [ ] Add tests first



Done: 

- [x] setup development environment on Windows to use WSL2 and containers(?) to use linux native stuff 
    - instead of trying to do cross-platform support for no reason other than my edification...


- [x] fix debugging symbols not being in outputted code
    - made an attempt at copying the set build type into the `tests` subfolder directly, but did not help
    - solution: use the actual visual studio debugger built into VS Code, I think
