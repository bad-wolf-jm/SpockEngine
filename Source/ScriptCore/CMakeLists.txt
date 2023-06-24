cmake_minimum_required(VERSION 3.20)
project(SE_CoreScriping LANGUAGES CSharp)

set(LIBRARY_NAME SE_Core)

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
)

add_library(${LIBRARY_NAME} SHARED
    Source/Math/mat4.cs
    Source/Math/mat3.cs
    Source/Math/vec2.cs
    Source/Math/vec3.cs
    Source/Math/vec4.cs
    Source/Math/quaternion.cs
    Source/Math/functions.cs

    # Source/Scripting/DomainManager.cs
    # Source/Scripting/Sponsor.cs
    # Source/Scripting/PluginLoader.cs
    # Source/Scripting/Interfaces.cs

    Source/Colors.cs
    # Source/ScriptProxy.cs
    Source/CppCalls.cs
    Source/Interop.cs
    Source/Entity.cs
    Source/Component.cs
    Source/Utilities.cs

    Source/UI/Components/Component.cs
    Source/UI/Components/Button.cs
    Source/UI/Components/TextToggleButton.cs
    Source/UI/Components/Label.cs
    Source/UI/Components/BaseImage.cs
    Source/UI/Components/Image.cs
    Source/UI/Components/ImageButton.cs
    Source/UI/Components/ImageToggleButton.cs
    Source/UI/Components/PropertyValue.cs
    Source/UI/Components/ComboBox.cs
    Source/UI/Components/Table.cs
    Source/UI/Components/Plot.cs
    Source/UI/Components/Workspace.cs
    Source/UI/Components/Menu.cs
    Source/UI/Components/TextInput.cs
    Source/UI/Components/TextOverlay.cs
    Source/UI/Components/ProgressBar.cs
    Source/UI/Components/CheckBox.cs
    Source/UI/Components/DropdownButton.cs
    Source/UI/Components/TreeView.cs
    Source/UI/Components/ColorButton.cs
    Source/UI/Components/Slider.cs
    Source/UI/Components/VectorInput.cs

    Source/UI/Widgets/FileTree.cs
    Source/UI/Widgets/CheckboxWithLabel.cs
    Source/UI/Widgets/ComboboxWithLabel.cs

    Source/UI/Layouts/BoxLayout.cs
    Source/UI/Layouts/ZLayout.cs
    Source/UI/Layouts/Splitter.cs
    Source/UI/Layouts/StackLayout.cs
    Source/UI/Layouts/Container.cs
    
    Source/UI/Form.cs
    Source/UI/Dialog.cs
    Source/UI/Colors.cs
    Source/UI/Fonts.cs
    
    # Source/Script.cs
    Source/SEApplication.cs
)

set_property(TARGET ${LIBRARY_NAME} PROPERTY DOTNET_TARGET_FRAMEWORK "net462")
set_property(TARGET ${LIBRARY_NAME} PROPERTY DOTNET_SDK "Microsoft.NET.Sdk")
set_property(TARGET ${LIBRARY_NAME} PROPERTY WIN32_EXECUTABLE FALSE)
set_property(TARGET ${LIBRARY_NAME} PROPERTY VS_CONFIGURATION_TYPE ClassLibrary)
set_property(TARGET ${LIBRARY_NAME} PROPERTY VS_PACKAGE_REFERENCES Newtonsoft.Json_13.0.1 System.ValueTuple_4.5.0)
set_property(TARGET ${LIBRARY_NAME} PROPERTY VS_DOTNET_REFERENCES 
    ${DOTNET_SYSTEM_ASSEMBLIES}
)
