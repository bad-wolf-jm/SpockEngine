cmake_minimum_required(VERSION 3.20)

add_executable(blueprints_vk_demo WIN32
# Source/blueprints-example.cpp
Source/builders.cpp
Source/drawing.cpp
Source/widgets.cpp
Source/Main.cpp
)

target_include_directories(blueprints_vk_demo PRIVATE Source )

target_compile_definitions(blueprints_vk_demo PRIVATE VK_USE_PLATFORM_WIN32_KHR)
target_link_libraries( blueprints_vk_demo LTSimulationEngineRuntime )
set_target_properties( blueprints_vk_demo PROPERTIES LINK_FLAGS /SUBSYSTEM:CONSOLE )
