cmake_minimum_required(VERSION 3.20)

project(LTSimulationEngine LANGUAGES CUDA CXX C)

cuda_compile_and_embed(g_SensorModelEnvironmentSampler Scene/EnvironmentSampler/DevicePrograms.cu)
cuda_compile_and_embed(embedded_ptx_code Scene/Renderer/RayTracing/DevicePrograms.cu)

add_library(LTSimulationEngineRuntime STATIC
    Core/File.cpp
    Core/Resource.cpp
    Core/Logging.cpp
    
    Mono/MonoscriptEngine.cpp
    Mono/MonoScriptClass.cpp
    Mono/MonoScriptInstance.cpp
    Mono/MonoScriptMethod.cpp
    Mono/MonoScriptUtils.cpp
    Mono/InternalCalls.cpp
    Mono/EntityRegistry.cpp
    
    Core/CUDA/Array/MultiTensor.cu
    Core/CUDA/Array/MemoryPool.cu
    Core/CUDA/Texture/Texture2D.cu
    Core/CUDA/Texture/Conversion.cpp
    Core/CUDA/Texture/TextureCubeMap.cu
    Core/CUDA/Texture/ColorFormat.cpp
    Core/CUDA/Texture/TextureData.cpp
    
    Engine/Engine.cpp
    
    UI/Widgets.cpp
    UI/UI.cpp
    UI/TreeNode.cpp
    UI/CanvasView.cpp
    
    Core/Optix/OptixContext.cpp
    Core/Optix/OptixModule.cpp
    Core/Optix/OptixPipeline.cpp
    Core/Optix/OptixProgramGroup.cpp
    Core/Optix/OptixAccelerationStructure.cpp
    
    Core/GraphicContext/UI/UIContext.cpp

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

    Renderer/ASceneRenderer.cpp
    Renderer/SceneRenderData.cpp

    Scene/Primitives/Primitives.cpp
    Scene/Primitives/Arrow.cpp
    Scene/Visualizers/Wall2D.cpp
    Scene/Scene.cpp
    Scene/VertexTransform.cu
    Scene/MaterialSystem/MaterialSystem.cpp
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

add_shader(LTSimulationEngineRuntime "copy.frag")
add_shader(LTSimulationEngineRuntime "fxaa.frag")
add_shader(LTSimulationEngineRuntime "fxaa.vert")

add_shader(LTSimulationEngineRuntime "shadow.vert")

add_shader(LTSimulationEngineRuntime "OmniShadow.vert")
add_shader(LTSimulationEngineRuntime "OmniShadow.frag")

add_shader(LTSimulationEngineRuntime "filtercube.vert")
add_shader(LTSimulationEngineRuntime "irradiancecube.frag")
add_shader(LTSimulationEngineRuntime "prefilterenvmap.frag")

add_shader(LTSimulationEngineRuntime "skybox.vert")
add_shader(LTSimulationEngineRuntime "skybox.frag")

add_shader(LTSimulationEngineRuntime "coordinategrid.vert")
add_shader(LTSimulationEngineRuntime "coordinategrid.frag")

add_shader(LTSimulationEngineRuntime "genbrdflut.vert")
add_shader(LTSimulationEngineRuntime "genbrdflut.frag")

add_shader(LTSimulationEngineRuntime "PBRMeshShader.vert")
add_shader(LTSimulationEngineRuntime "PBRMeshShader.frag")

add_shader(LTSimulationEngineRuntime "ui_shader.vert")
add_shader(LTSimulationEngineRuntime "ui_shader.frag")

add_shader(LTSimulationEngineRuntime "ParticleSystem.vert")
add_shader(LTSimulationEngineRuntime "ParticleSystem.frag")

add_shader(LTSimulationEngineRuntime "SurfaceShader.vert")
add_shader(LTSimulationEngineRuntime "SurfaceShader.frag")

add_shader(LTSimulationEngineRuntime "WireframeShader.vert")
add_shader(LTSimulationEngineRuntime "WireframeShader.frag")

add_shader(LTSimulationEngineRuntime "Unlit.vert")
add_shader(LTSimulationEngineRuntime "Unlit.frag")

add_shader(LTSimulationEngineRuntime "Deferred/DeferredLightingMSAA.vert")
add_shader(LTSimulationEngineRuntime "Deferred/DeferredLightingMSAA.frag")

add_shader(LTSimulationEngineRuntime "Deferred/MRT.vert")
add_shader(LTSimulationEngineRuntime "Deferred/MRT.frag")

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
    ${optix_LIBRARY}
    ${CUDA_LIBRARIES}
    ${CUDA_CUDA_LIBRARY}
    CUDA::curand
    wsock32
    ws2_32
    yaml-cpp
    tinygltf
    tinyobjloader
    lua::lua
    Zep
    ${ASSIMP_LIBRARIES}
    glfw)

target_compile_definitions(LTSimulationEngineRuntime PRIVATE)
target_compile_options(LTSimulationEngineRuntime PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr -Xcudafe "--display_error_number --diag_suppress=\"20012,20044\"">
    $<$<COMPILE_LANGUAGE:CXX>:/bigobj /wd4068>)



add_executable(SE_Run WIN32
    Launch/EditorMain.cpp
)

target_include_directories(SE_Run PRIVATE Source ${SE_EXTRA_INCLUDES})
target_compile_definitions(SE_Run PRIVATE VK_USE_PLATFORM_WIN32_KHR)
target_link_libraries( SE_Run LTSimulationEngineRuntime lua::lua)
set_target_properties( SE_Run PROPERTIES LINK_FLAGS /SUBSYSTEM:CONSOLE )
