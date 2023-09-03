cmake_minimum_required(VERSION 3.20)

project(LTSimulationEngine LANGUAGES CXX CUDA C)

cuda_compile_and_embed(g_SensorModelEnvironmentSampler Scene/EnvironmentSampler/DevicePrograms.cu)
# cuda_compile_and_embed(embedded_ptx_code Scene/Renderer/RayTracing/DevicePrograms.cu)

message(STATUS ${SHADER_EMBEDDED_FILES})

add_library(LTSimulationEngineRuntime STATIC
    Core/File.cpp
    Core/Resource.cpp
    Core/Logging.cpp
    Core/Vector.cpp
    Core/String.cpp

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

    Scene/Primitives/Primitives.cpp
    Scene/Primitives/Arrow.cpp
    Scene/Visualizers/Wall2D.cpp
    Scene/Scene.cpp
    Scene/VertexTransform.cu
    Scene/Serialize/FileIO.cpp
    Scene/Serialize/AssetFile.cpp
    Scene/Serialize/SerializeComponents.cpp
    
    Renderer/ASceneRenderer.cpp
    Renderer/SceneRenderer.cpp
    Renderer/MaterialSystem.cpp
    Renderer/ShadowRenderer.cpp
    Renderer/GridRenderer.cpp

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


set(MONO_PATH "C:/GitLab/SpockEngine/ThirdParty/mono/lib/Debug")

target_link_libraries(LTSimulationEngineRuntime PRIVATE
    # LTSimulationEngineRuntimeShaders
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
    glfw
)

target_compile_definitions(LTSimulationEngineRuntime PRIVATE)
target_compile_options(LTSimulationEngineRuntime PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr -Xcudafe "--display_error_number --diag_suppress=\"20012,20044\"">
    $<$<COMPILE_LANGUAGE:CXX>:/bigobj /wd4068>)


add_executable( SE_Run WIN32 Launch/EditorMain.cpp )
target_include_directories(SE_Run PRIVATE Source ${SE_EXTRA_INCLUDES})
target_compile_definitions(SE_Run PRIVATE VK_USE_PLATFORM_WIN32_KHR)
target_link_libraries( SE_Run LTSimulationEngineRuntime lua::lua)
set_target_properties( SE_Run PROPERTIES LINK_FLAGS /SUBSYSTEM:CONSOLE )