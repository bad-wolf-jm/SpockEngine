cmake_minimum_required(VERSION 3.20)

add_executable(test_mono_exfo WIN32
    Source/Main.cpp
)

target_include_directories(test_mono_exfo PRIVATE Source )

target_compile_definitions(test_mono_exfo PRIVATE VK_USE_PLATFORM_WIN32_KHR)
target_compile_options( test_mono_exfo PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr -Xcudafe "--display_error_number --diag_suppress=20012"> )
target_link_libraries( test_mono_exfo LTSimulationEngineRuntime )
set_target_properties( test_mono_exfo PROPERTIES LINK_FLAGS /SUBSYSTEM:CONSOLE )
