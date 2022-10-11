cmake_minimum_required(VERSION 3.20)

add_executable(test_delaunay_triangulation WIN32
    Source/Delaunay.cpp
    Source/Main.cpp
)

target_include_directories(test_delaunay_triangulation PRIVATE Source )

target_compile_definitions(test_delaunay_triangulation PRIVATE VK_USE_PLATFORM_WIN32_KHR)
target_compile_options( test_delaunay_triangulation PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr -Xcudafe "--display_error_number --diag_suppress=20012"> )
target_link_libraries( test_delaunay_triangulation LTSimulationEngineRuntime )
set_target_properties( test_delaunay_triangulation PROPERTIES LINK_FLAGS /SUBSYSTEM:CONSOLE )
