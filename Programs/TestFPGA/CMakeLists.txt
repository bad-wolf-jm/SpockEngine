cmake_minimum_required(VERSION 3.20)

add_executable(test_fpga_model WIN32
    Source/Main.cpp
)

target_include_directories(test_fpga_model PRIVATE Source )

target_compile_definitions(test_fpga_model PRIVATE VK_USE_PLATFORM_WIN32_KHR)
target_compile_options( test_fpga_model PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr -Xcudafe "--display_error_number --diag_suppress=20012"> )
target_link_libraries( test_fpga_model LTSimulationEngineRuntime )
set_target_properties( test_fpga_model PROPERTIES LINK_FLAGS /SUBSYSTEM:CONSOLE )