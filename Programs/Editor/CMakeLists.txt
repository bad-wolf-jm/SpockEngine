cmake_minimum_required(VERSION 3.20)

project(EXFO LANGUAGES CSharp)

set(LIBRARY_NAME SEEditor)

add_library(${LIBRARY_NAME} SHARED
    Source/Main.cs
    Source/MaterialEditor.cs
)

set(NUGET_PACKAGE_ROOT "D:/Work/.nuget/packages")

set(DOTNET_REFERENCE_PATH_TP
    ${NUGET_PACKAGE_ROOT}/system.diagnostics.performancecounter/7.0.0/lib/net462/System.Diagnostics.PerformanceCounter.dll
    ${NUGET_PACKAGE_ROOT}/microsoft.extensions.configuration/3.1.4/lib/netstandard2.0/Microsoft.Extensions.Configuration.dll
    ${NUGET_PACKAGE_ROOT}/microsoft.extensions.configuration.abstractions/3.1.4/lib/netstandard2.0/Microsoft.Extensions.Configuration.Abstractions.dll
    ${NUGET_PACKAGE_ROOT}/microsoft.extensions.configuration.binder/3.1.3/lib/netstandard2.0/Microsoft.Extensions.Configuration.Binder.dll
    ${NUGET_PACKAGE_ROOT}/microsoft.extensions.primitives/3.1.4/lib/netstandard2.0/Microsoft.Extensions.Primitives.dll
    ${NUGET_PACKAGE_ROOT}/newtonsoft.json/12.0.3/lib/net40/Newtonsoft.Json.dll
    ${NUGET_PACKAGE_ROOT}/newtonsoft.json.bson/1.0.2/lib/net45/Newtonsoft.Json.Bson.dll
    ${NUGET_PACKAGE_ROOT}/openmcdf/2.2.1.12/lib/net40/OpenMcdf.dll
    ${NUGET_PACKAGE_ROOT}/YamlDotNet/YamlDotNet.dll
)


set(DOTNET_SYSTEM_ASSEMBLIES
    System.dll
    System.Core.dll
    System.Windows.Forms.dll
    System.ComponentModel.Annotations.dll
    System.ComponentModel.DataAnnotations.dll
    System.Drawing.dll
    System.Runtime.Serialization.dll
    System.Xml.dll
    System.Xml.Serialization.dll
    System.Xml.Linq.dll
    System.Management.dll
    System.Numerics.dll
)

set(DOTNET_REFERENCE_PATH
    ${DOTNET_REFERENCE_PATH_TP}
    ${DOTNET_SYSTEM_ASSEMBLIES}
)


set_property(TARGET ${LIBRARY_NAME} PROPERTY DOTNET_TARGET_FRAMEWORK_VERSION "v4.6.2")
set_property(TARGET ${LIBRARY_NAME} PROPERTY WIN32_EXECUTABLE FALSE)
set_property(TARGET ${LIBRARY_NAME} PROPERTY VS_CONFIGURATION_TYPE ClassLibrary)
set_property(TARGET ${LIBRARY_NAME} PROPERTY VS_DOTNET_REFERENCES 
    "D:\\Personal\\Git\\SpockEngine\\Build\\CoreScripting\\Debug\\SE_Core.dll"

    ${DOTNET_REFERENCE_PATH}
)
