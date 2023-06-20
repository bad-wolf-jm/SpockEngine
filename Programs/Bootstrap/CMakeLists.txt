cmake_minimum_required(VERSION 3.20)

project(SE_Bootstrap LANGUAGES CSharp)

set(LIBRARY_NAME SE_Bootstrap)

set(CMAKE_CSharp_FLAGS "/langversion:latest") 

set(DOTNET_SYSTEM_ASSEMBLIES
    System.dll
    System.Core.dll
    System.Linq.dll
    System.Windows.Forms.dll
    System.ComponentModel.Annotations.dll
    System.ComponentModel.DataAnnotations.dll
    System.Drawing.dll
    System.Xml.dll
    System.Xml.Serialization.dll
    System.Xml.Linq.dll
    System.Management.dll
    System.Reflection.dll
    System.Runtime.dll
    System.Runtime.Serialization.dll
    PresentationCore.dll
)

add_executable(${LIBRARY_NAME} 
    Source/Main.cs
)

set_property(TARGET ${LIBRARY_NAME} PROPERTY DOTNET_TARGET_FRAMEWORK "net462")
set_property(TARGET ${LIBRARY_NAME} PROPERTY DOTNET_SDK "Microsoft.NET.Sdk")
set_property(TARGET ${LIBRARY_NAME} PROPERTY WIN32_EXECUTABLE TRUE)
set_property(TARGET ${LIBRARY_NAME} PROPERTY VS_CONFIGURATION_TYPE ClassLibrary)
set_property(TARGET ${LIBRARY_NAME} PROPERTY VS_PACKAGE_REFERENCES Newtonsoft.Json_13.0.1 CommandLineParser_2.9.1)
set_property(TARGET ${LIBRARY_NAME} PROPERTY VS_DOTNET_REFERENCES 
    "C:\\GitLab\\SpockEngine\\Source\\ScriptCore\\Build\\Debug\\SE_Core.dll"
    ${DOTNET_SYSTEM_ASSEMBLIES}
)
