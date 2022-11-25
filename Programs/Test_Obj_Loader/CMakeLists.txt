cmake_minimum_required(VERSION 3.20)

add_executable(ltse_obj_import WIN32
    Source/Main.cpp
)

target_include_directories(ltse_obj_import PRIVATE Source )

target_compile_definitions(ltse_obj_import PRIVATE VK_USE_PLATFORM_WIN32_KHR)
target_link_libraries( ltse_obj_import LTSimulationEngineRuntime )
set_target_properties( ltse_obj_import PROPERTIES LINK_FLAGS /SUBSYSTEM:CONSOLE )

