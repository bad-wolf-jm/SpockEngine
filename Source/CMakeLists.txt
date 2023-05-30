cmake_minimum_required(VERSION 3.20)

project(LTSimulationEngine LANGUAGES CXX CUDA C)

cuda_compile_and_embed(g_SensorModelEnvironmentSampler Scene/EnvironmentSampler/DevicePrograms.cu)
cuda_compile_and_embed(embedded_ptx_code Scene/Renderer/RayTracing/DevicePrograms.cu)





add_library(LTSimulationEngineRuntime STATIC
    Core/File.cpp
    Core/Resource.cpp
    Core/Logging.cpp

    Core/Entity/EditComponent.cpp

    DotNet/Runtime.cpp
    DotNet/Class.cpp
    DotNet/Instance.cpp
    DotNet/Utils.cpp
    DotNet/EntityRegistry.cpp
    DotNet/InternalCalls.cpp

    Core/CUDA/Cuda.cpp
    Core/CUDA/Array/MultiTensor.cu
    Core/CUDA/Array/MemoryPool.cu
    Core/CUDA/Texture/Texture2D.cu
    Core/CUDA/Texture/Conversion.cpp
    Core/CUDA/Texture/TextureCubeMap.cu
    Core/CUDA/Texture/ColorFormat.cpp
    Core/CUDA/Texture/TextureData.cpp

    Engine/Engine.cpp

    Shader/Compiler.cpp
    
    UI/UI.cpp
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
    UI/Components/DropdownButton.cpp
    UI/Components/TreeView.cpp
    UI/Widgets/FileTree.cpp
    UI/Components/ColorButton.cpp
    UI/Components/Slider.cpp
    UI/Layouts/BoxLayout.cpp
    UI/Layouts/ZLayout.cpp
    UI/Layouts/Splitter.cpp
    UI/Layouts/StackLayout.cpp
    UI/Layouts/Container.cpp
    UI/Form.cpp
    UI/Dialog.cpp
    UI/UIContext.cpp
    UI/UIWindow.cpp
    UI/Widgets.cpp
    UI/TreeNode.cpp
    
    Core/Optix/OptixContext.cpp
    Core/Optix/OptixModule.cpp
    Core/Optix/OptixPipeline.cpp
    Core/Optix/OptixProgramGroup.cpp
    Core/Optix/OptixAccelerationStructure.cpp

    Graphics/API.cpp
    Graphics/Vulkan/VkRenderTarget.cpp
    Graphics/Vulkan/VkSwapChain.cpp
    Graphics/Vulkan/VkGpuBuffer.cpp
    Graphics/Vulkan/VkTexture2D.cpp
    Graphics/Vulkan/VkSampler2D.cpp
    Graphics/Vulkan/VkSamplerCubeMap.cpp
    Graphics/Vulkan/VkGraphicContext.cpp
    Graphics/Vulkan/VkBaseRenderContext.cpp
    Graphics/Vulkan/VkRenderContext.cpp
    Graphics/Vulkan/VkSwapChainRenderContext.cpp
    Graphics/Vulkan/VkGraphicsPipeline.cpp
    Graphics/Vulkan/VkPipeline.cpp
    Graphics/Vulkan/VkCommand.cpp
    Graphics/Vulkan/VkRenderPass.cpp
    Graphics/Vulkan/VkDescriptorSet.cpp
    Graphics/Vulkan/VkDescriptorSetLayout.cpp
    Graphics/Vulkan/VkShaderProgram.cpp

    Graphics/Interface/IWindow.cpp
    Graphics/Interface/ITexture.cpp
    Graphics/Interface/ITexture2D.cpp
    Graphics/Interface/ISampler2D.cpp
    Graphics/Interface/ISamplerCubeMap.cpp
    Graphics/Interface/IRenderTarget.cpp
    Graphics/Interface/IGraphicsPipeline.cpp
    Graphics/Interface/IRenderContext.cpp
    Graphics/Interface/IDescriptorSet.cpp
    Graphics/Interface/ISwapChain.cpp
    Graphics/Interface/IRenderPass.cpp
    Graphics/Interface/ICommandBuffer.cpp
    Graphics/Interface/IDescriptorSet.cpp
    Graphics/Interface/IDescriptorSetLayout.cpp
    Graphics/Interface/IShaderProgram.cpp

    OtdrEditor/OtdrWindow.cpp
    OtdrEditor/BaseOtdrApplication.cpp

    Renderer/ASceneRenderer.cpp
    Renderer/SceneRenderData.cpp

    Scene/Primitives/Primitives.cpp
    Scene/Primitives/Arrow.cpp
    Scene/Visualizers/Wall2D.cpp
    Scene/Scene.cpp
    Scene/VertexTransform.cu
    Scene/MaterialSystem/MaterialSystem.cpp
    Scene/MaterialSystem/TextureManager.cpp
    Scene/Renderer/MeshRenderer.cpp
    Scene/Renderer/CoordinateGridRenderer.cpp
    Scene/Renderer/EffectProcessor.cpp
    Scene/Renderer/ParticleSystemRenderer.cpp
    Scene/Renderer/ForwardSceneRenderer.cpp
    Scene/Renderer/ShadowSceneRenderer.cpp
    Scene/Renderer/DeferredSceneRenderer.cpp
    Scene/Renderer/DeferredLightingRenderer.cpp
    Scene/Serialize/FileIO.cpp
    Scene/Serialize/AssetFile.cpp
    Scene/Serialize/SerializeComponents.cpp
    ${embedded_ptx_code}
    Scene/Renderer/RayTracing/ToneMap.cu
    Scene/Renderer/RayTracing/RayTracingRenderer.h
    Scene/Renderer/RayTracing/RayTracingRenderer.cpp

    Scene/Importer/glTFImporter.cpp
    Scene/Importer/ObjImporter.cpp

    Editor/PopupWindow.cpp
    Editor/EditorWindow.cpp
    Editor/ContentBrowser.cpp
    Editor/BaseEditorApplication.cpp
    Editor/SceneHierarchyPanel.cpp
    Editor/SceneElementEditor.cpp

    ProcessingNodes/HitRecordProcessorKernels.cu
    ProcessingNodes/HitRecordProcessor.cpp

    Scene/EnvironmentSampler/PointCloudVisualizer.cu
    Scene/EnvironmentSampler/EnvironmentSampler.cpp
    ${g_SensorModelEnvironmentSampler}

    TensorOps/Implementation/KernelLaunchers.cu
    TensorOps/NodeControllers.cpp
    TensorOps/ScalarTypes.cpp
    TensorOps/Scope.cpp
)


function(embed_shader TARGET output_var shader_file)
    set(shader_path ${CMAKE_CURRENT_SOURCE_DIR}/Scene/Renderer/Shaders/${shader_file})
    set(shader_output_path ${CMAKE_CURRENT_SOURCE_DIR}/Scene/Renderer/Shaders/Embedded)

    if (NOT EXISTS ${shader_output_path})
        file(MAKE_DIRECTORY ${shader_output_path})
    endif ()

    file(READ ${shader_path} content HEX)

    string(REGEX MATCHALL "([A-Fa-f0-9][A-Fa-f0-9])" SEPARATED_HEX ${content})

    set(counter 0)
    # Iterate through each of the bytes from the source file
    foreach (hex IN LISTS SEPARATED_HEX)
        # Write the hex string to the line with an 0x prefix and a , postfix to seperate the bytes of the file.
        string(APPEND output_c "0x${hex},")
        # Increment the element counter before the newline.
        math(EXPR counter "${counter}+1")
        if (counter GREATER 16)
            # Write a newline so that all of the array initializer gets spread across multiple lines.
            string(APPEND output_c "\n        ")
            set(counter 0)
        endif ()
    endforeach ()

    set(output_c "
#include \"${output_var}.h\"

namespace SE::Private::Shaders
{
    const std::vector<uint8_t> ${output_var}_data = {
        ${output_c}
    }\;
}")
    
    # Generate the contents that will be contained in the header file.
    set(output_h "
#pragma once

#include \"stdint.h\"
#include <vector>

namespace SE::Private::Shaders
{
    extern const std::vector<uint8_t> ${output_var}_data\;
}")

    file(WRITE ${shader_output_path}/${output_var}.h ${output_h})
    file(WRITE ${shader_output_path}/${output_var}.cpp ${output_c})

    set_source_files_properties(${shader_output_path}/${output_var}.h PROPERTIES GENERATED TRUE)
    set_source_files_properties(${shader_output_path}/${output_var}.cpp PROPERTIES GENERATED TRUE)

    target_sources(${TARGET} PRIVATE ${shader_path})
    target_sources(${TARGET} PRIVATE ${shader_output_path}/${output_var}.cpp)
endfunction(embed_shader)

embed_shader(LTSimulationEngineRuntime gUIVertexShader                       "ui_shader.vert")
embed_shader(LTSimulationEngineRuntime gUIFragmentShader                     "ui_shader.frag")
embed_shader(LTSimulationEngineRuntime gCopyFragmentShader                   "copy.frag")
embed_shader(LTSimulationEngineRuntime gFXAAFragmentShader                   "fxaa.frag")
embed_shader(LTSimulationEngineRuntime gFXAAVertexShader                     "fxaa.vert")
embed_shader(LTSimulationEngineRuntime gDirectionalShadowVertexShader        "shadow.vert")
embed_shader(LTSimulationEngineRuntime gOmniDirectionalShadowVertexShader    "OmniShadow.vert")
embed_shader(LTSimulationEngineRuntime gOmniDirectionalShadowFragmentShader  "OmniShadow.frag")
embed_shader(LTSimulationEngineRuntime gFiolterCubeVertexShader              "filtercube.vert")
embed_shader(LTSimulationEngineRuntime gIrradianceCubeFragmentShader         "irradiancecube.frag")
embed_shader(LTSimulationEngineRuntime gPrefilteredEnvironmentFragmentShader "prefilterenvmap.frag")
embed_shader(LTSimulationEngineRuntime gSkyboxVertexShader                   "skybox.vert")
embed_shader(LTSimulationEngineRuntime gSkyboxFragmentShader                 "skybox.frag")
embed_shader(LTSimulationEngineRuntime gCoordinateGridVertexShader           "coordinategrid.vert")
embed_shader(LTSimulationEngineRuntime gCoordinateGridFragmentShader         "coordinategrid.frag")
embed_shader(LTSimulationEngineRuntime gGenerateBrdfVertexShader             "genbrdflut.vert")
embed_shader(LTSimulationEngineRuntime gGenerateBrdfFragmentShader           "genbrdflut.frag")
embed_shader(LTSimulationEngineRuntime gPBRMeshVertexShader                  "PBRMeshShader.vert")
embed_shader(LTSimulationEngineRuntime gPBRMeshFragmentShader                "PBRMeshShader.frag")
embed_shader(LTSimulationEngineRuntime gParticleSystemVerrtexShader          "ParticleSystem.vert")
embed_shader(LTSimulationEngineRuntime gParticleSystemFragmentShader         "ParticleSystem.frag")
embed_shader(LTSimulationEngineRuntime gSurfaceVertexShader                  "SurfaceShader.vert")
embed_shader(LTSimulationEngineRuntime gSurfaceFragmentShader                "SurfaceShader.frag")
embed_shader(LTSimulationEngineRuntime gWireframeVertexShader                "WireframeShader.vert")
embed_shader(LTSimulationEngineRuntime gWirefrrameFragmentShader             "WireframeShader.frag")
embed_shader(LTSimulationEngineRuntime gUnlitVertexShader                    "Unlit.vert")
embed_shader(LTSimulationEngineRuntime gUnlitFragmentShader                  "Unlit.frag")
embed_shader(LTSimulationEngineRuntime gDeferredLightingMSAAVertexShader     "Deferred/DeferredLightingMSAA.vert")
embed_shader(LTSimulationEngineRuntime gDeferredLightingMSAAFragmentShader   "Deferred/DeferredLightingMSAA.frag")
embed_shader(LTSimulationEngineRuntime gDeferredGeometryPassVertexShader     "Deferred/MRT.vert")
embed_shader(LTSimulationEngineRuntime gDeferredGeometryPassFragmentShader   "Deferred/MRT.frag")


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

# add_shader(LTSimulationEngineRuntime "ui_shader.vert")
# add_shader(LTSimulationEngineRuntime "ui_shader.frag")

set(MONO_PATH "C:/GitLab/SpockEngine/ThirdParty/mono/lib/Debug")

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
    glslang
    SPIRV 
    glslang-default-resource-limits
    ${MONO_PATH}/libmono-static-sgen.lib
    ${MONO_PATH}/mono-2.0-sgen.lib
    ${MONO_PATH}/MonoPosixHelper.lib
    ${MONO_PATH}/libmonoruntime-sgen.lib
    ${MONO_PATH}/libmonoutils.lib
    ${MONO_PATH}/libmini-sgen.lib
    ${MONO_PATH}/libgcmonosgen.lib
    ${MONO_PATH}/eglib.lib
    ${optix_LIBRARY}
    ${CUDA_LIBRARIES}
    ${CUDA_CUDA_LIBRARY}
    CUDA::curand    
    wsock32
    ws2_32
    yaml-cpp
    pugixml
    tinygltf
    tinyobjloader
    glfw)

target_compile_definitions(LTSimulationEngineRuntime PRIVATE)
target_compile_options(LTSimulationEngineRuntime PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr -Xcudafe "--display_error_number --diag_suppress=\"20012,20044\"">
    $<$<COMPILE_LANGUAGE:CXX>:/bigobj /wd4068>)


add_executable( SE_Run WIN32 Launch/EditorMain.cpp )
target_include_directories(SE_Run PRIVATE Source ${SE_EXTRA_INCLUDES})
target_compile_definitions(SE_Run PRIVATE VK_USE_PLATFORM_WIN32_KHR)
target_link_libraries( SE_Run LTSimulationEngineRuntime lua::lua)
set_target_properties( SE_Run PROPERTIES LINK_FLAGS /SUBSYSTEM:CONSOLE )
    
add_executable(OTDR_Run WIN32 Launch/OtdrMain.cpp)
target_include_directories(OTDR_Run PRIVATE Source ${SE_EXTRA_INCLUDES})
target_compile_definitions(OTDR_Run PRIVATE VK_USE_PLATFORM_WIN32_KHR)
target_link_libraries( OTDR_Run LTSimulationEngineRuntime lua::lua)
set_target_properties( OTDR_Run PROPERTIES LINK_FLAGS /SUBSYSTEM:CONSOLE )
