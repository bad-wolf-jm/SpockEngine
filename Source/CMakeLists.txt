cmake_minimum_required(VERSION 3.20)

project(LTSimulationEngine LANGUAGES CXX C)

add_library(LTSimulationEngineRuntime STATIC
    Core/File.cpp
    Core/Resource.cpp
    Core/Logging.cpp

    DotNet/Runtime.cpp
    DotNet/Class.cpp
    DotNet/Instance.cpp
    DotNet/Utils.cpp

    Core/CUDA/Texture/ColorFormat.cpp
    Core/CUDA/Texture/TextureData.cpp
    
    Engine/Engine.cpp
    
    UI/UI.cpp
    UI/TreeNode.cpp
    UI/Components/Component.cpp
    UI/Components/Button.cpp
    UI/Components/CheckBox.cpp
    UI/Components/Label.cpp
    UI/Components/TextOverlay.cpp
    UI/Components/BaseImage.cpp
    UI/Components/Image.cpp
    UI/Components/ImageButton.cpp
    UI/Components/ComboBox.cpp
    UI/Components/TextToggleButton.cpp
    UI/Components/ImageToggleButton.cpp
    UI/Components/Table.cpp
    UI/Components/Plot.cpp
    UI/Components/PropertyValue.cpp
    UI/Components/Workspace.cpp
    UI/Components/Menu.cpp
    UI/Components/TextInput.cpp
    UI/Components/ProgressBar.cpp
    UI/Layouts/BoxLayout.cpp
    UI/Layouts/ZLayout.cpp
    UI/Layouts/Splitter.cpp
    UI/Layouts/StackLayout.cpp
    UI/Form.cpp
    UI/Dialog.cpp
    UI/UIContext.cpp
    UI/UIWindow.cpp
    
    Graphics/API.cpp
    Graphics/Vulkan/DescriptorSet.cpp
    Graphics/Vulkan/GraphicsPipeline.cpp
    Graphics/Vulkan/ARenderContext.cpp
    Graphics/Vulkan/VkRenderTarget.cpp
    Graphics/Vulkan/SwapChain.cpp
    Graphics/Vulkan/VkGpuBuffer.cpp
    Graphics/Vulkan/VkTextureCubeMap.cpp
    Graphics/Vulkan/VkTexture2D.cpp
    Graphics/Vulkan/VkSampler2D.cpp
    Graphics/Vulkan/VkSamplerCubeMap.cpp
    Graphics/Vulkan/VkGraphicContext.cpp
    Graphics/Vulkan/VkRenderContext.cpp
    Graphics/Vulkan/VkGraphicsPipeline.cpp
    Graphics/Vulkan/VkPipeline.cpp
    Graphics/Vulkan/VkCommand.cpp
    Graphics/Vulkan/VkAbstractRenderPass.cpp

    Graphics/Interface/IWindow.cpp
    Graphics/Interface/ITexture.cpp
    Graphics/Interface/ITexture2D.cpp
    Graphics/Interface/ITextureCubeMap.cpp
    Graphics/Interface/ISampler2D.cpp
    Graphics/Interface/ISamplerCubeMap.cpp
    Graphics/Interface/IRenderTarget.cpp
    Graphics/Interface/IGraphicsPipeline.cpp
    Graphics/Interface/IRenderContext.cpp
    Graphics/Interface/IDescriptorSet.cpp

    OtdrEditor/OtdrWindow.cpp
    OtdrEditor/BaseOtdrApplication.cpp
)

function(add_shader TARGET SHADER)
    find_program(GLSLC glslc)

    set(CURRENT_SHADER_PATH ${CMAKE_CURRENT_SOURCE_DIR}/Scene/Renderer/Shaders/${SHADER})
    set(CURRENT_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../Resources/Shaders/${SHADER}.spv)

    # Add a custom command to compile GLSL to SPIR-V.
    get_filename_component(CURRENT_OUTPUT_DIR ${CURRENT_OUTPUT_PATH} DIRECTORY)
    file(MAKE_DIRECTORY ${CURRENT_OUTPUT_DIR})
    message(STATUS ${CURRENT_SHADER_PATH} " **************>> " ${CURRENT_OUTPUT_PATH})
    add_custom_command(
      OUTPUT ${CURRENT_OUTPUT_PATH}
      COMMAND ${GLSLC} -o ${CURRENT_OUTPUT_PATH} ${CURRENT_SHADER_PATH}
      DEPENDS ${CURRENT_SHADER_PATH}
      IMPLICIT_DEPENDS CXX ${CURRENT_SHADER_PATH}
      VERBATIM)

    # Make sure our build depends on this output.
    set_source_files_properties(${CURRENT_OUTPUT_PATH} PROPERTIES GENERATED TRUE)
    target_sources(${TARGET} PRIVATE ${CURRENT_OUTPUT_PATH})
endfunction(add_shader)

add_shader(LTSimulationEngineRuntime "ui_shader.vert")
add_shader(LTSimulationEngineRuntime "ui_shader.frag")

target_link_libraries(LTSimulationEngineRuntime PRIVATE
    Vulkan::Vulkan
    imgui
    imguizmo
    implot
    imgui-node-editor
    fmt
    winmm
    version
    Bcrypt
    "C:/GitLab/SpockEngine/ThirdParty/mono/lib/Debug/libmono-static-sgen.lib"
    "C:/GitLab/SpockEngine/ThirdParty/mono/lib/Debug/libmonoruntime-sgen.lib"
    "C:/GitLab/SpockEngine/ThirdParty/mono/lib/Debug/mono-2.0-sgen.lib"
    "C:/GitLab/SpockEngine/ThirdParty/mono/lib/Debug/MonoPosixHelper.lib"
    "C:/GitLab/SpockEngine/ThirdParty/mono/lib/Debug/libmonoutils.lib"
    "C:/GitLab/SpockEngine/ThirdParty/mono/lib/Debug/libmini-sgen.lib"
    "C:/GitLab/SpockEngine/ThirdParty/mono/lib/Debug/libgcmonosgen.lib"
    "C:/GitLab/SpockEngine/ThirdParty/mono/lib/Debug/eglib.lib"
    wsock32
    ws2_32
    yaml-cpp
    pugixml
    glfw)

target_compile_definitions(LTSimulationEngineRuntime PRIVATE)
target_compile_options(LTSimulationEngineRuntime PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr -Xcudafe "--display_error_number --diag_suppress=\"20012,20044\"">
    $<$<COMPILE_LANGUAGE:CXX>:/bigobj /wd4068>)

add_executable(OTDR_Run WIN32 Launch/OtdrMain.cpp)
target_include_directories(OTDR_Run PRIVATE Source ${SE_EXTRA_INCLUDES})
target_compile_definitions(OTDR_Run PRIVATE VK_USE_PLATFORM_WIN32_KHR)
target_link_libraries( OTDR_Run LTSimulationEngineRuntime lua::lua)
set_target_properties( OTDR_Run PROPERTIES LINK_FLAGS /SUBSYSTEM:CONSOLE )
