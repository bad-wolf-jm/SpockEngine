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
    System.Runtime.Serialization.dll
    System.Xml.dll
    System.Xml.Serialization.dll
    System.Xml.Linq.dll
    System.Management.dll
)


add_library(${LIBRARY_NAME} SHARED
    Source/Math/mat4.cs
    Source/Math/mat3.cs
    Source/Math/vec2.cs
    Source/Math/vec3.cs
    Source/Math/vec4.cs
    Source/Math/quaternion.cs
    Source/Math/functions.cs

    Source/Colors.cs
    Source/ScriptProxy.cs
    Source/CppCalls.cs
    Source/Entity.cs
    Source/Component.cs
    Source/Utilities.cs
    Source/TensorOps.cs

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

    Source/UI/Layouts/BoxLayout.cs
    Source/UI/Layouts/ZLayout.cs
    Source/UI/Layouts/Splitter.cs
    Source/UI/Layouts/StackLayout.cs
    Source/UI/Layouts/Container.cs
    
    Source/UI/Form.cs
    Source/UI/Dialog.cs
    Source/UI/Colors.cs
    Source/UI/Fonts.cs
    
    Source/Script.cs
    Source/SEApplication.cs
)

set_property(TARGET ${LIBRARY_NAME} PROPERTY DOTNET_TARGET_FRAMEWORK_VERSION "v4.6.1")
set_property(TARGET ${LIBRARY_NAME} PROPERTY WIN32_EXECUTABLE FALSE)
set_property(TARGET ${LIBRARY_NAME} PROPERTY VS_CONFIGURATION_TYPE ClassLibrary)
set_property(TARGET ${LIBRARY_NAME} PROPERTY VS_DOTNET_REFERENCES 
    ${DOTNET_SYSTEM_ASSEMBLIES}
)
