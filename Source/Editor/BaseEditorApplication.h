#pragma once

#include <filesystem>

#include "Core/Math/Types.h"
#include "Core/Memory.h"

#include "UI/UI.h"

#include "Engine/Engine.h"
#include "UI/UIContext.h"

#include "Graphics/Vulkan/VkGraphicContext.h"
#include "Scene/EnvironmentSampler/EnvironmentSampler.h"
#include "Scene/EnvironmentSampler/PointCloudVisualizer.h"
#include "Scene/Renderer/DeferredSceneRenderer.h"
#include "Scene/Renderer/ForwardSceneRenderer.h"
#include "Scene/Renderer/RayTracing/RayTracingRenderer.h"
#include "Scene/Scene.h"

#include "Renderer2/NewSceneRenderer.h"

#include "TensorOps/Scope.h"

#include "EditorWindow.h"

namespace SE::Editor
{
    namespace fs = std::filesystem;

    using namespace SE::Core;
    using namespace SE::Graphics;
    using namespace SE::TensorOps;

    class BaseEditorApplication
    {
      public:
        EditorWindow mEditorWindow;
        Ref<Engine>  mEngineLoop             = nullptr;
        fs::path     ConfigurationRoot       = "";
        fs::path     ConfigurationFile       = "";
        fs::path     SensorConfigurationFile = "";
        std::string  ApplicationName         = "Sensor Model Editor";
        std::string  ImGuiIniFile            = "imgui_config.ini";
        math::ivec2  WindowSize              = { 1920, 1080 };
        math::ivec2  WindowPosition          = { 100, 100 };

      public:
        BaseEditorApplication() = default;

        ~BaseEditorApplication() = default;

        void Init();
        void Init( std::string aAppClass, fs::path aConfigurationPath );

        void RenderScene();
        void Update( Timestep ts );
        bool RenderUI( ImGuiIO &io );

      protected:
        void RebuildOutputFramebuffer();

      protected:
        uint32_t mNewViewportHeight        = 1;
        uint32_t mNewViewportWidth         = 1;
        bool     mShouldRebuildNewViewport = true;

        uint32_t   mViewportHeight        = 1;
        uint32_t   mViewportWidth         = 1;
        bool       mShouldRebuildViewport = true;
        Ref<Scene> mWorld                 = nullptr;
        Ref<Scene> mTestWorld             = nullptr;

        Ref<DeferredRenderer>     mDeferredRenderer   = nullptr;
        Ref<NewSceneRenderer>     mNewSceneRenderer   = nullptr;
        Ref<ForwardSceneRenderer> mForwardRenderer    = nullptr;
        Ref<RayTracingRenderer>   mRayTracingRenderer = nullptr;

        Ref<ISampler2D> mOffscreenRenderTargetTexture = nullptr;
        ImageHandle     mOffscreenRenderTargetDisplayHandle{};

        Ref<ISampler2D> mDeferredRenderTargetTexture = nullptr;
        ImageHandle     mDeferredRenderTargetDisplayHandle{};

        Ref<ISampler2D> mNewRenderTargetTexture = nullptr;
        ImageHandle     mNewRenderTargetDisplayHandle{};

        Entity m_SensorEntity{};

        Ref<DotNetInstance> mApplicationInstance = nullptr;
    };

} // namespace SE::Editor