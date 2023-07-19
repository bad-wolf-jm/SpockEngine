cmake_minimum_required(VERSION 3.20)

project(LTSimulationEngine LANGUAGES CXX CUDA C)

add_executable( TestNewMaterials WIN32 Source/Main.cpp )
target_include_directories(TestNewMaterials PRIVATE Source ${SE_EXTRA_INCLUDES})
target_compile_definitions(TestNewMaterials PRIVATE VK_USE_PLATFORM_WIN32_KHR)
target_link_libraries( TestNewMaterials LTSimulationEngineRuntime)
set_target_properties( TestNewMaterials PROPERTIES LINK_FLAGS /SUBSYSTEM:CONSOLE )