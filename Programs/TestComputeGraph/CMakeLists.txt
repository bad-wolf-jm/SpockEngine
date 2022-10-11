cmake_minimum_required(VERSION 3.20)

add_executable(test_compute_graph WIN32
    Source/Main.cpp
    # Source/WaveformGenerator.cpp
    # Source/WaveformGeneratorKernels.cu
)

target_include_directories(test_compute_graph PRIVATE Source )

target_compile_definitions(test_compute_graph PRIVATE VK_USE_PLATFORM_WIN32_KHR)
target_compile_options( test_compute_graph PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr -Xcudafe "--display_error_number --diag_suppress=20012"> )
target_link_libraries( test_compute_graph LTSimulationEngineRuntime )
set_target_properties( test_compute_graph PROPERTIES LINK_FLAGS /SUBSYSTEM:CONSOLE )
