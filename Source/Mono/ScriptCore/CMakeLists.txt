cmake_minimum_required(VERSION 3.20)
project(SE_CoreScriping CSharp)

add_executable("galapagos"
    Test.cs
)

set_property(TARGET "galapagos" PROPERTY DOTNET_TARGET_FRAMEWORK_VERSION "v4.6.1")
set_property(TARGET "galapagos" PROPERTY WIN32_EXECUTABLE FALSE)
set_property(TARGET "galapagos" PROPERTY VS_CONFIGURATION_TYPE ClassLibrary)
