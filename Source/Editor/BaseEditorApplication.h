#pragma once

#include <filesystem>

#include "Core/Math/Types.h"
#include "Core/Memory.h"

#include "UI/UI.h"

#include "Core/GraphicContext//UI/UIContext.h"
#include "Core/Platform/EngineLoop.h"


#include "EnvironmentSampler/EnvironmentSampler.h"
#include "EnvironmentSampler/PointCloudVisualizer.h"
#include "Core/GraphicContext//GraphicContext.h"
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
    // using namespace LTSE::SensorModel;
    // using namespace LTSE::SensorModel::Dev;
    using namespace LTSE::TensorOps;

    class BaseEditorApplication
    {
      public:
        EditorWindow mEditorWindow;
        Ref<EngineLoop> mEngineLoop      = nullptr;
        fs::path ConfigurationRoot       = "";
        fs::path ConfigurationFile       = "";
        fs::path SensorConfigurationFile = "";
        std::string ApplicationName      = "Sensor Model Editor";
        std::string ImGuiIniFile         = "imgui_config.ini";
        math::ivec2 WindowSize           = { 1920, 1080 };
        math::ivec2 WindowPosition       = { 100, 100 };

      public:
        BaseEditorApplication();
        ~BaseEditorApplication() = default;

        void Init();

        void RenderScene();
        virtual void Update( Timestep ts ) = 0;
        bool RenderUI( ImGuiIO &io );

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
        uint32_t m_ViewportHeight          = 1;
        uint32_t m_ViewportWidth           = 1;
        bool m_ShouldRebuildViewport       = true;
        Ref<Scene> m_World                 = nullptr;
        Ref<SceneRenderer> m_WorldRenderer = nullptr;

        RenderContext m_ViewportRenderContext{};
        Ref<OffscreenRenderTarget> m_OffscreenRenderTarget      = nullptr;
        Ref<Graphics::Texture2D> m_OffscreenRenderTargetTexture = nullptr;
        // Ref<SensorDeviceBase> m_SensorController                = nullptr;

        ImageHandle m_OffscreenRenderTargetDisplayHandle{};
        Entity m_SensorEntity{};
    };

} // namespace LTSE::Editor