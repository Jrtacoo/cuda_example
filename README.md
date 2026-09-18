# cuda_example
some examples I made during the learning of CUDA

nsys profile -o result .\tile_mine.exe
nsys stats result.nsys-rep     
nsys-ui result.nsys-rep       


ncu -o profile_result .\tile_demo.exe ;采集指定 kernel 的详细指标

ncu -k matmul_tile_reg -o profile_result .\tile_demo.exe ;只分析某个 kernel（按名字过滤，避免采集全部）

ncu --import profile_result.ncu-rep ;命令行直接看结果

ncu-ui profile_result.ncu-rep ;或 GUI 打开
