cmake_minimum_required(VERSION 3.20)

add_executable(ltse_raytracing WIN32
    Source/Main.cpp
    # Source/Device.cpp
    # Source/TestSensorModel.cpp
)

target_include_directories(ltse_raytracing PRIVATE Source )

target_compile_definitions(ltse_raytracing PRIVATE VK_USE_PLATFORM_WIN32_KHR)
target_compile_options(ltse_raytracing PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr -Xcudafe "--display_error_number --diag_suppress=20012">)
target_link_libraries( ltse_raytracing LTSimulationEngineRuntime )
set_target_properties( ltse_raytracing PROPERTIES LINK_FLAGS /SUBSYSTEM:CONSOLE )
