cmake_minimum_required(VERSION 3.20)

add_executable(test_coreclr_embedding WIN32
    Source/nativehost.cpp
)

target_include_directories(test_coreclr_embedding PRIVATE Source )

target_compile_definitions(test_coreclr_embedding PRIVATE VK_USE_PLATFORM_WIN32_KHR)
target_link_libraries( test_coreclr_embedding LTSimulationEngineRuntime )
set_target_properties( test_coreclr_embedding PROPERTIES LINK_FLAGS /SUBSYSTEM:CONSOLE )
