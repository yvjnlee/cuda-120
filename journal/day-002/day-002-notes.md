#### day 2

##### project organization
- important to reduce tech debt as much as possible
    - Make or CMake files for easier builds
    - clear folder structure
        - src/ : main.cu (host code + kernel invocations), kernels.cu (device kernels), utils.cu
        - include/ : declrations for kernels, so header files?
        - Makefile or CMakeLists.txt
        - docs/ : documentation
        - README.md

##### device linking
- for projects with multiple CUDA files u might wanna compile with `-dlink`
    - this combines multple compiled object files into a single executable

##### debugging
- cuda-gdb
    - gdb based debugger that lets u set breakpoints, inspect thread-local variables and step thru kernel instructions
- `printf` in kernels (classic)
- cuda-memcheck
    - checks for out of bounds access, misaligned mem usage and other gpu mem errors

##### profiling and performance tools
- nsight systems
    - timeline-based profiler that shows how cpu functions and gpu kernels overlap
    - helps identifiy concurrency or synchronization bottlenecks
- nsight compute
    - kernel level metrics like occupancy, instruction throughput, warp diverence, memory transactions
    - useful for performance tuning
- start with nsight systems (bigger picture) -> nsight compute (optimizations)

##### summary
- learn about general project folder structure and good high level practices
- also looked at debugging and tools available for devloping kernels

