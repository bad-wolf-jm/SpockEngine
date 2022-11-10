#pragma once

#include <filesystem>

#include "Core/Math/Types.h"
#include "Core/Memory.h"

#include "UI/UI.h"

#include "Core/GraphicContext//UI/UIContext.h"
#include "Core/Platform/EngineLoop.h"

#include "Core/GraphicContext//GraphicContext.h"
#include "Scene/EnvironmentSampler/EnvironmentSampler.h"
#include "Scene/EnvironmentSampler/PointCloudVisualizer.h"
#include "Scene/Renderer/DeferredLightingPass.h"
#include "Scene/Renderer/DeferredSceneRenderer.h"
#include "Scene/Renderer/SceneRenderer.h"
#include "Scene/Scene.h"

#include "TensorOps/Scope.h"

#include "EditorWindow.h"

namespace LTSE::Editor
{
    namespace fs = std::filesystem;

    using namespace LTSE::Core;
    using namespace LTSE::Graphics;
    using namespace LTSE::TensorOps;

    class BaseEditorApplication
    {
      public:
        EditorWindow    mEditorWindow;
        Ref<EngineLoop> mEngineLoop             = nullptr;
        fs::path        ConfigurationRoot       = "";
        fs::path        ConfigurationFile       = "";
        fs::path        SensorConfigurationFile = "";
        std::string     ApplicationName         = "Sensor Model Editor";
        std::string     ImGuiIniFile            = "imgui_config.ini";
        math::ivec2     WindowSize              = { 1920, 1080 };
        math::ivec2     WindowPosition          = { 100, 100 };

      public:
        BaseEditorApplication();
        ~BaseEditorApplication() = default;

        void Init();

        void         RenderScene();
        virtual void Update( Timestep ts ) = 0;
        bool         RenderUI( ImGuiIO &io );

        uint32_t Run();

        virtual void LoadSensorConfiguration() = 0;
        virtual void SaveSensorConfiguration() = 0;
        virtual void OnUI()                    = 0;

      protected:
        void LoadConfiguration();
        void SaveConfiguration();

      protected:
        void RebuildOutputFramebuffer();

      protected:
        uint32_t                   mViewportHeight           = 1;
        uint32_t                   mViewportWidth            = 1;
        bool                       mShouldRebuildViewport    = true;
        Ref<Scene>                 mWorld                    = nullptr;
        Ref<SceneRenderer>         mWorldRenderer            = nullptr;
        Ref<DeferredSceneRenderer> mDeferredWorldRenderer    = nullptr;
        Ref<DeferredLightingPass>  mDeferredLightingRenderer = nullptr;

        RenderContext                 mViewportRenderContext{};
        DeferredRenderContext         mDeferredRenderContext{};
        DeferredLightingRenderContext mDeferredLightingRenderContext{};
        Ref<OffscreenRenderTarget>    mOffscreenRenderTarget        = nullptr;
        Ref<Graphics::Texture2D>      mOffscreenRenderTargetTexture = nullptr;
        ImageHandle                   mOffscreenRenderTargetDisplayHandle{};

        Ref<Graphics::Texture2D>      mDeferredRenderTargetTexture = nullptr;
        ImageHandle                   mDeferredRenderTargetDisplayHandle{};


        Ref<DescriptorSet> mLightingPassInputs = nullptr;

        Ref<DeferredRenderTarget> mDeferredRenderTarget = nullptr;
        Ref<LightingRenderTarget> mLightingRenderTarget = nullptr;

        Entity m_SensorEntity{};
    };

} // namespace LTSE::Editor