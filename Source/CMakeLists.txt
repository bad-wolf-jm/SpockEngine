cmake_minimum_required(VERSION 3.20)

project(LTSimulationEngine LANGUAGES CXX CUDA C)

message(STATUS ${SHADER_EMBEDDED_FILES})

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

add_library(LTSimulationEngineRuntime SHARED
    Core/Logging.cpp

    Core/CUDA/Cuda.cpp
    Core/CUDA/Array/MultiTensor.cu
    Core/CUDA/Array/MemoryPool.cu
    Core/CUDA/Texture/Texture2D.cu
    Core/CUDA/Texture/Conversion.cpp
    Core/CUDA/Texture/TextureCubeMap.cu
    Core/CUDA/Texture/ColorFormat.cpp
    Core/CUDA/Texture/TextureData.cpp


    TensorOps/Implementation/KernelLaunchers.cu
    TensorOps/NodeControllers.cpp
    TensorOps/ScalarTypes.cpp
    TensorOps/Scope.cpp
)

target_link_libraries(LTSimulationEngineRuntime PRIVATE
    fmt
    winmm
    version
    Bcrypt
    ${CUDA_LIBRARIES}
    ${CUDA_CUDA_LIBRARY}
    CUDA::curand    
)

target_compile_definitions(LTSimulationEngineRuntime PUBLIC LTSimulationEngineRuntime )
target_compile_options(LTSimulationEngineRuntime PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr -Xcudafe "--display_error_number --diag_suppress=\"20012,20044\"">
    $<$<COMPILE_LANGUAGE:CXX>:/bigobj /wd4068>)