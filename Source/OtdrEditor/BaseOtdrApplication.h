#pragma once

#include <filesystem>

#include "Core/Math/Types.h"
#include "Core/Memory.h"

#include "UI/UI.h"
#include "UI/UIContext.h"

#include "Engine/Engine.h"

#include "OtdrWindow.h"

namespace SE::OtdrEditor
{
    namespace fs = std::filesystem;

    using namespace SE::Core;
    using namespace SE::Graphics;

    class BaseOtdrApplication
    {
      public:
        OtdrWindow  mEditorWindow;
        Ref<Engine> mEngineLoop             = nullptr;
        path_t    ConfigurationRoot       = "";
        path_t    ConfigurationFile       = "";
        path_t    SensorConfigurationFile = "";
        string_t ApplicationName         = "Sensor Model Editor";
        string_t ImGuiIniFile            = "imgui_config.ini";
        math::ivec2 WindowSize              = { 1920, 1080 };
        math::ivec2 WindowPosition          = { 100, 100 };

      public:
        BaseOtdrApplication() = default;

        ~BaseOtdrApplication() = default;

        void Init();
        void Init(string_t aAppClass, path_t aConfigurationPath);
        void Shutdown(path_t aConfigurationPath);

        void RenderScene() {}
        void Update( Timestep ts );
        bool RenderUI( ImGuiIO &io );

      protected:
        uint32_t mViewportHeight        = 1;
        uint32_t mViewportWidth         = 1;
        bool     mShouldRebuildViewport = true;

        Ref<DotNetInstance> mApplicationInstance = nullptr;
    };

} // namespace SE::OtdrEditor