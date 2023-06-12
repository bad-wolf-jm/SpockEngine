cmake_minimum_required(VERSION 3.20)
project(SE_CoreScriping LANGUAGES CSharp)

set(LIBRARY_NAME MonoScriptingTest)

add_library(${LIBRARY_NAME} SHARED
    Source/Test_BaseLibrary.cs
    # Source/Test_Entities.cs
    # Source/Test_TensorOps.cs
)

set_property(TARGET ${LIBRARY_NAME} PROPERTY DOTNET_TARGET_FRAMEWORK_VERSION "v4.6.2")
set_property(TARGET ${LIBRARY_NAME} PROPERTY WIN32_EXECUTABLE FALSE)
set_property(TARGET ${LIBRARY_NAME} PROPERTY VS_CONFIGURATION_TYPE ClassLibrary)
set_property(TARGET ${LIBRARY_NAME} PROPERTY VS_DOTNET_REFERENCES "C:/GitLab/SpockEngine/Source/ScriptCore/Build/Debug/SE_Core.dll")
