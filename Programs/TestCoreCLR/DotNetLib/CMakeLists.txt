cmake_minimum_required(VERSION 3.20)
project(DotNetLib LANGUAGES CSharp)

set(LIBRARY_NAME DotNetLib)

# set(CMAKE_CSharp_FLAGS "/langversion:latest") 

set(DOTNET_SYSTEM_ASSEMBLIES
    System.dll
    System.Core.dll
    System.Linq.dll
    System.Windows.Forms.dll
    System.ComponentModel.Annotations.dll
    System.ComponentModel.DataAnnotations.dll
    System.Drawing.dll
    System.Runtime.dll
    System.Runtime.Serialization.dll
    System.Runtime.InteropServices.dll
    System.Runtime.InteropServices.RuntimeInformation.dll
    System.Xml.dll
    System.Xml.Serialization.dll
    System.Xml.Linq.dll
    System.Management.dll
)

add_library(${LIBRARY_NAME} SHARED
    Source/Lib.cs
)

set_property(TARGET ${LIBRARY_NAME} PROPERTY DOTNET_TARGET_FRAMEWORK_VERSION "v4.6.1")
set_property(TARGET ${LIBRARY_NAME} PROPERTY WIN32_EXECUTABLE FALSE)
set_property(TARGET ${LIBRARY_NAME} PROPERTY VS_CONFIGURATION_TYPE ClassLibrary)
set_property(TARGET ${LIBRARY_NAME} PROPERTY VS_DOTNET_REFERENCES 
    ${DOTNET_SYSTEM_ASSEMBLIES}
)
