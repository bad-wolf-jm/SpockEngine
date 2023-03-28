#pragma once

#include <filesystem>

#include "Core/Math/Types.h"
#include "Core/Memory.h"

#include "UI/UI.h"

#include "Core/GraphicContext//UI/UIContext.h"
#include "Engine/Engine.h"

#include "TensorOps/Scope.h"

#include "OtdrWindow.h"

namespace SE::OtdrEditor
{
    namespace fs = std::filesystem;

    using namespace SE::Core;
    using namespace SE::Graphics;
    using namespace SE::TensorOps;

    class BaseOtdrApplication
    {
      public:
        OtdrWindow  mEditorWindow;
        Ref<Engine> mEngineLoop             = nullptr;
        fs::path    ConfigurationRoot       = "";
        fs::path    ConfigurationFile       = "";
        fs::path    SensorConfigurationFile = "";
        std::string ApplicationName         = "Sensor Model Editor";
        std::string ImGuiIniFile            = "imgui_config.ini";
        math::ivec2 WindowSize              = { 1920, 1080 };
        math::ivec2 WindowPosition          = { 100, 100 };

      public:
        BaseOtdrApplication() = default;

        ~BaseOtdrApplication() = default;

        void Init();
        void Init(std::string aAppClass);
        void Shutdown();

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