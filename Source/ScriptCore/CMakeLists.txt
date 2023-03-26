cmake_minimum_required(VERSION 3.20)
project(SE_CoreScriping LANGUAGES CSharp)

set(LIBRARY_NAME SE_Core)

add_library(${LIBRARY_NAME} SHARED
    Source/Math/mat4.cs
    Source/Math/mat3.cs
    Source/Math/vec2.cs
    Source/Math/vec3.cs
    Source/Math/vec4.cs
    Source/Math/quaternion.cs
    Source/Math/functions.cs

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
    Source/UI/Layouts/BoxLayout.cs
    Source/UI/Layouts/ZLayout.cs
    Source/UI/UI.cs
    Source/UI/Form.cs
    
    Source/CppCalls.cs
    Source/Component.cs
    Source/Entity.cs
    Source/Initialization.cs
    Source/ActorComponent.cs
    Source/Script.cs
    Source/HUDComponent.cs
    Source/TensorOps.cs
    Source/SEApplication.cs
)

set_property(TARGET ${LIBRARY_NAME} PROPERTY DOTNET_TARGET_FRAMEWORK_VERSION "v4.6.1")
set_property(TARGET ${LIBRARY_NAME} PROPERTY WIN32_EXECUTABLE FALSE)
set_property(TARGET ${LIBRARY_NAME} PROPERTY VS_CONFIGURATION_TYPE ClassLibrary)
