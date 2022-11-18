cmake_minimum_required(VERSION 3.20)

add_executable(ltse_3d_engine_test WIN32
    Source/FileTree.cpp
    Source/Main.cpp
)

target_include_directories(ltse_3d_engine_test PRIVATE Source )
target_include_directories(ltse_3d_engine_test PRIVATE ${SE_EXTRA_INCLUDES} )

target_compile_definitions(ltse_3d_engine_test PRIVATE VK_USE_PLATFORM_WIN32_KHR)
target_link_libraries( ltse_3d_engine_test LTSimulationEngineRuntime tinygltf lua::lua )
set_target_properties( ltse_3d_engine_test PROPERTIES LINK_FLAGS /SUBSYSTEM:CONSOLE )
