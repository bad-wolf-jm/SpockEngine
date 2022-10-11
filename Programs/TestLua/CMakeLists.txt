cmake_minimum_required(VERSION 3.20)

add_executable(test_lua WIN32
    Source/Main.cpp
)

target_include_directories(test_lua PRIVATE Source )

target_compile_definitions(test_lua PRIVATE VK_USE_PLATFORM_WIN32_KHR)
target_compile_options( test_lua PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr -Xcudafe "--display_error_number --diag_suppress=20012"> )
target_link_libraries( test_lua LTSimulationEngineRuntime lua::lua )
set_target_properties( test_lua PROPERTIES LINK_FLAGS /SUBSYSTEM:CONSOLE )
