#pragma once

#include <filesystem>

#include "Core/Math/Types.h"
#include "Core/Memory.h"

#include "UI/UI.h"
#include "UI/UIContext.h"

#include "Engine/Engine.h"

#include "CoreCLRHost.h"
#include "OtdrWindow.h"

namespace SE::OtdrEditor
{
    namespace fs = std::filesystem;

    using namespace SE::Core;
    using namespace SE::Graphics;

    class Application
    {
      public:
        MainWindow mEditorWindow;
        math::ivec2 WindowSize     = { 1920, 1080 };
        math::ivec2 WindowPosition = { 100, 100 };

      public:
        Application( CoreCLRHost &aManaged )
            : mManaged{ &aManaged } {};

        ~Application() = default;

        void Init();
        void Init( path_t aConfigurationPath );
        void Shutdown( path_t aConfigurationPath );

        void RenderScene() {}
        void Update( Timestep ts );
        bool RenderUI( ImGuiIO &io );

      protected:
        uint32_t mViewportHeight        = 1;
        uint32_t mViewportWidth         = 1;
        bool     mShouldRebuildViewport = true;

        CoreCLRHost *mManaged = nullptr;
    };

} // namespace SE::OtdrEditor