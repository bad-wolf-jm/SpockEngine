cmake_minimum_required(VERSION 3.20)

add_executable(test_sensor_load WIN32
    Source/Main.cpp
)

target_include_directories(test_sensor_load PRIVATE Source )

target_compile_definitions(test_sensor_load PRIVATE VK_USE_PLATFORM_WIN32_KHR)
target_compile_options( test_sensor_load PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr -Xcudafe "--display_error_number --diag_suppress=20012"> )
target_link_libraries( test_sensor_load LTSimulationEngineRuntime )
set_target_properties( test_sensor_load PROPERTIES LINK_FLAGS /SUBSYSTEM:CONSOLE )
