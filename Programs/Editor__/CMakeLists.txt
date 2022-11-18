cmake_minimum_required(VERSION 3.20)

add_executable(ltse_editor WIN32
    Main.cpp
)

target_include_directories(ltse_editor PRIVATE Source ${SE_EXTRA_INCLUDES})
target_compile_definitions(ltse_editor PRIVATE VK_USE_PLATFORM_WIN32_KHR)
target_link_libraries( ltse_editor LTSimulationEngineRuntime lua::lua)
set_target_properties( ltse_editor PROPERTIES LINK_FLAGS /SUBSYSTEM:CONSOLE )
