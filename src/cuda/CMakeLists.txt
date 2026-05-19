cmake_minimum_required(VERSION 3.18)
project(TransformerCUDA LANGUAGES CXX CUDA)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Export compile commands for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Create an executable target named 'test_exec' from test.cu
add_executable(test_exec test.cu)

# Enable separable compilation for CUDA
set_target_properties(test_exec PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Optional: Add any required CUDA architectures (e.g., native)
# set_target_properties(test_exec PROPERTIES CUDA_ARCHITECTURES "native")
