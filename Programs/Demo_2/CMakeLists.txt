cmake_minimum_required(VERSION 3.20)

add_executable(ltse_demo_2 WIN32
    Source/Main.cpp
)

target_include_directories(ltse_demo_2 PRIVATE Source )

target_compile_definitions(ltse_demo_2 PRIVATE VK_USE_PLATFORM_WIN32_KHR)
target_link_libraries( ltse_demo_2 LTSimulationEngineRuntime )
set_target_properties( ltse_demo_2 PROPERTIES LINK_FLAGS /SUBSYSTEM:CONSOLE )

