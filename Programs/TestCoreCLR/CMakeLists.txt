cmake_minimum_required(VERSION 3.20)

add_executable(test_coreclr_embedding WIN32
    Source/nativehost.cpp
)

target_include_directories(test_coreclr_embedding PRIVATE 
    Source
    C:/GitLab/runtime/artifacts/bin/win-x64.Debug/corehost
     )

# target_compile_definitions(test_coreclr_embedding PRIVATE VK_USE_PLATFORM_WIN32_KHR)
target_link_libraries( test_coreclr_embedding 
    C:/GitLab/runtime/artifacts/bin/win-x64.Debug/corehost/hostfxr.lib
    C:/GitLab/runtime/artifacts/bin/win-x64.Debug/corehost/nethost.lib
)
set_target_properties( test_coreclr_embedding PROPERTIES LINK_FLAGS /SUBSYSTEM:CONSOLE )

add_custom_command(TARGET test_coreclr_embedding POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy "C:/GitLab/runtime/artifacts/bin/win-x64.Debug/corehost/hostfxr.dll" $<TARGET_FILE_DIR:test_coreclr_embedding>
  COMMAND ${CMAKE_COMMAND} -E copy "C:/GitLab/runtime/artifacts/bin/win-x64.Debug/corehost/nethost.dll" $<TARGET_FILE_DIR:test_coreclr_embedding>
)
