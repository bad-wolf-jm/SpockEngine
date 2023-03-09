cmake_minimum_required(VERSION 3.20)
project(SE_CoreScriping LANGUAGES CSharp)

set(LIBRARY_NAME Test)

add_library(${LIBRARY_NAME} SHARED
    Source/Application.cs
    Source/PowerMeter.cs
)

set(NUGET_PACKAGE_ROOT "D:/.nuget/packages")
set(DOTNET_SYSTEM_ASSEMBLIES
    System.dll
    System.Core.dll
    System.ComponentModel.Annotations.dll
    System.ComponentModel.DataAnnotations.dll
    System.Drawing.dll
    System.Runtime.Serialization.dll
    System.Xml.dll
    System.Xml.Serialization.dll
    System.Xml.Linq.dll
    System.Management.dll
)

set(EXFO_OPTICAL_PLATFORM ${NUGET_PACKAGE_ROOT}/exfo.optical.platform/1.6.19091/lib/net40/)
set(EXFO_OPTICAL_PLATFORM_ASSEMBLIES 
    ${EXFO_OPTICAL_PLATFORM}/Metrino.Kernos.dll
    ${EXFO_OPTICAL_PLATFORM}/Metrino.Kernos.Instrument.dll
    ${EXFO_OPTICAL_PLATFORM}/Metrino.Kernos.IO.ExfoBus.dll
    ${EXFO_OPTICAL_PLATFORM}/Metrino.Kernos.Platform.dll
    ${EXFO_OPTICAL_PLATFORM}/Metrino.Kernos.Windows.Forms.dll
    ${EXFO_OPTICAL_PLATFORM}/Metrino.Platform.Client.dll
    ${EXFO_OPTICAL_PLATFORM}/Metrino.Platform.Definitions.dll
    ${EXFO_OPTICAL_PLATFORM}/Metrino.EepromSections.dll
)

set(DOTNET_REFERENCE_PATH_NON_EXFO
    ${NUGET_PACKAGE_ROOT}/system.diagnostics.performancecounter/7.0.0/lib/net462/System.Diagnostics.PerformanceCounter.dll
    ${NUGET_PACKAGE_ROOT}/microsoft.extensions.configuration/3.1.4/lib/netstandard2.0/Microsoft.Extensions.Configuration.dll
    ${NUGET_PACKAGE_ROOT}/microsoft.extensions.configuration.abstractions/3.1.4/lib/netstandard2.0/Microsoft.Extensions.Configuration.Abstractions.dll
    ${NUGET_PACKAGE_ROOT}/microsoft.extensions.configuration.binder/3.1.3/lib/netstandard2.0/Microsoft.Extensions.Configuration.Binder.dll
    ${NUGET_PACKAGE_ROOT}/microsoft.extensions.primitives/3.1.4/lib/netstandard2.0/Microsoft.Extensions.Primitives.dll
    ${NUGET_PACKAGE_ROOT}/newtonsoft.json/12.0.3/lib/net40/Newtonsoft.Json.dll
    ${NUGET_PACKAGE_ROOT}/newtonsoft.json.bson/1.0.2/lib/net45/Newtonsoft.Json.Bson.dll
)

set(DOTNET_REFERENCE_PATH
    ${EXFO_OPTICAL_PLATFORM_ASSEMBLIES}
    ${DOTNET_REFERENCE_PATH_NON_EXFO}
    ${DOTNET_SYSTEM_ASSEMBLIES}
)

set(REFERENCE_PATH D:/Build/Lib/debug)

set_property(TARGET ${LIBRARY_NAME} PROPERTY DOTNET_TARGET_FRAMEWORK_VERSION "v4.6.2")
set_property(TARGET ${LIBRARY_NAME} PROPERTY WIN32_EXECUTABLE FALSE)
set_property(TARGET ${LIBRARY_NAME} PROPERTY VS_CONFIGURATION_TYPE ClassLibrary)
set_property(TARGET ${LIBRARY_NAME} PROPERTY VS_DOTNET_REFERENCES 
    "C:/GitLab/SpockEngine/Source/ScriptCore/Build/Debug/SE_Core.dll"
    ${REFERENCE_PATH}/Metrino.Mono/Metrino.Mono.dll
    ${REFERENCE_PATH}/Metrino.Otdr/Metrino.Otdr.dll
    ${REFERENCE_PATH}/Metrino.Otdr.SignalProcessing/Metrino.Otdr.SignalProcessing.dll
    ${REFERENCE_PATH}/Metrino.Otdr.Instrument/Metrino.Otdr.Instrument.dll
    ${DOTNET_REFERENCE_PATH}
)
