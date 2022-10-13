cmake_minimum_required(VERSION 3.20)

project(LTSimulationEngine LANGUAGES CUDA CXX C)

cuda_compile_and_embed(g_SensorModelEnvironmentSampler Developer/EnvironmentSampler/DevicePrograms.cu)

add_library(LTSimulationEngineRuntime STATIC
    Core/File.cpp
    Core/Resource.cpp
    Core/ColorFormat.cpp
    Core/Logging.cpp
    Core/TextureData.cpp
    Core/AssetFile.cpp

    Scripting/ScriptingEngine.cpp
    Scripting/Core/Texture.cpp
    Scripting/Core/Vector.cpp
    Scripting/Entities/EntityRegistry.cpp
    Scripting/Cuda/Tensor.cpp
    Scripting/Cuda/Texture.cpp
    Scripting/Math/VectorTypes.cpp
    Scripting/Math/MatrixTypes.cpp
    # Scripting/SensorModel.cpp
    Scripting/ArrayTypes.cpp
    # Scripting/ScriptedSensorDevice.cpp

    Cuda/MultiTensor.cu
    Cuda/MemoryPool.cu
    Cuda/Texture2D.cu

    Developer/Platform/ViewportClient.cpp
    Developer/Platform/EngineLoop.cpp

    Developer/UI/Widgets.cpp
    Developer/UI/UI.cpp
    Developer/UI/CanvasView.cpp

    Developer/Core/Optix/OptixContext.cpp
    Developer/Core/Optix/OptixModule.cpp
    Developer/Core/Optix/OptixPipeline.cpp
    Developer/Core/Optix/OptixProgramGroup.cpp
    Developer/Core/Optix/OptixAccelerationStructure.cpp

    Developer/Core/Vulkan/VkContext.cpp
    Developer/Core/Vulkan/VkRenderPass.cpp
    Developer/Core/Vulkan/VkPipeline.cpp
    Developer/Core/Vulkan/VkImage.cpp
    Developer/Core/Vulkan/VkCommand.cpp

    Developer/GraphicContext/Buffer.cpp
    Developer/GraphicContext/Texture2D.cpp
    Developer/GraphicContext/TextureCubeMap.cpp
    Developer/GraphicContext/GraphicContext.cpp
    Developer/GraphicContext/DescriptorSet.cpp
    Developer/GraphicContext/GraphicsPipeline.cpp
    Developer/GraphicContext/RenderContext.cpp
    Developer/GraphicContext/RenderTarget.cpp
    Developer/GraphicContext/UI/UIContext.cpp

    Developer/Scene/Primitives/Primitives.cpp
    Developer/Scene/Primitives/Arrow.cpp
    Developer/Scene/Visualizers/Wall2D.cpp
    Developer/Scene/Scene.cpp
    Developer/Scene/VertexTransform.cu
    Developer/Scene/MaterialSystem/MaterialSystem.cpp
    Developer/Scene/Components/VisualHelpers.cpp
    Developer/Scene/Renderer/MeshRenderer.cpp
    Developer/Scene/Renderer/CoordinateGridRenderer.cpp
    Developer/Scene/Renderer/ParticleSystemRenderer.cpp
    Developer/Scene/Renderer/VisualHelperLineRenderer.cpp
    Developer/Scene/Renderer/SceneRenderer.cpp
    Developer/Scene/Renderer/VisualHelperRenderer.cpp
    Developer/Scene/Renderer/VisualHelperMeshRenderer.cpp

    Developer/Scene/Importer/glTFImporter.cpp

    Developer/Editor/PopupWindow.cpp
    Developer/Editor/EditorWindow.cpp
    Developer/Editor/ContentBrowser.cpp
    Developer/Editor/BaseEditorApplication.cpp
    Developer/Editor/SceneHierarchyPanel.cpp
    Developer/Editor/SceneElementEditor.cpp

    # LidarSensorModel/SensorModelBase.cpp
    # LidarSensorModel/EnvironmentSampler.cpp
    # LidarSensorModel/SensorControllerBase.cpp
    # LidarSensorModel/SensorDevicebase.cpp
    # LidarSensorModel/ModelArchive.cpp
    # LidarSensorModel/AcquisitionContext/AcquisitionContext.cpp

    ProcessingNodes/HitRecordProcessorKernels.cu
    ProcessingNodes/HitRecordProcessor.cpp

    Serialize/FileIO.cpp
    # Serialize/SensorAsset.cpp
    # Serialize/SensorComponents.cpp
    # Serialize/SensorDefinition.cpp

    Developer/EnvironmentSampler/PointCloudVisualizer.cu
    Developer/EnvironmentSampler/EnvironmentSampler.cpp
    ${g_SensorModelEnvironmentSampler}

    TensorOps/Implementation/KernelLaunchers.cu
    TensorOps/NodeControllers.cpp
    TensorOps/ScalarTypes.cpp
    TensorOps/Scope.cpp
)

function(add_shader TARGET SHADER)
    find_program(GLSLC glslc)

    set(CURRENT_SHADER_PATH ${CMAKE_CURRENT_SOURCE_DIR}/Developer/Scene/Renderer/Shaders/${SHADER})
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

target_link_libraries(LTSimulationEngineRuntime PRIVATE
    Vulkan::Vulkan
    imgui
    imguizmo
    implot
    imgui-node-editor
    fmt
    ${optix_LIBRARY}
    ${CUDA_LIBRARIES}
    ${CUDA_CUDA_LIBRARY}
    CUDA::curand
    wsock32
    ws2_32
    yaml-cpp
    tinygltf
    lua::lua
    Zep
    ${ASSIMP_LIBRARIES}
    glfw)

target_compile_definitions(LTSimulationEngineRuntime PRIVATE)
target_compile_options(LTSimulationEngineRuntime PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr -Xcudafe "--display_error_number --diag_suppress=20012">
    $<$<COMPILE_LANGUAGE:CXX>:/bigobj>)
