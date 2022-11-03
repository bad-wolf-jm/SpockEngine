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
#include "Scene/Renderer/SceneRenderer.h"
#include "Scene/Scene.h"

// #include "LidarSensorModel/SensorDeviceBase.h"
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
        Ref<EngineLoop> mEngineLoop = nullptr;

        EditorWindow mEditorWindow;
        fs::path     ConfigurationRoot       = "";
        fs::path     ConfigurationFile       = "";
        fs::path     SensorConfigurationFile = "";
        std::string  ApplicationName         = "Sensor Model Editor";
        std::string  ImGuiIniFile            = "imgui_config.ini";
        math::ivec2  WindowSize              = { 1920, 1080 };
        math::ivec2  WindowPosition          = { 100, 100 };

      public:
        BaseEditorApplication() = default;
        BaseEditorApplication( Ref<EngineLoop> aEngineLoop );

        ~BaseEditorApplication() = default;

        void Init();

        void RenderScene();
        void Update( Timestep ts );
        bool RenderUI( ImGuiIO &io );

      protected:
        void RebuildOutputFramebuffer();

      protected:
        uint32_t           mViewportHeight        = 1;
        uint32_t           mViewportWidth         = 1;
        bool               mShouldRebuildViewport = true;
        Ref<Scene>         mWorld                 = nullptr;
        Ref<SceneRenderer> mWorldRenderer         = nullptr;

        RenderContext              mViewportRenderContext{};
        Ref<OffscreenRenderTarget> mOffscreenRenderTarget = nullptr;
        ImageHandle                mOffscreenRenderTargetDisplayHandle{};
        Ref<Graphics::Texture2D>   mOffscreenRenderTargetTexture = nullptr;
    };

} // namespace LTSE::Editor