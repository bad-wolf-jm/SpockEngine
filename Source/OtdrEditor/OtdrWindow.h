#pragma once

#include <functional>
#include <string>
#include <vector>

#include "Core/Math/Types.h"
#include "Core/Memory.h"

#include "Graphics/Interface/IGraphicContext.h"
#include "Graphics/Interface/ISampler2D.h"

#include "UI/Components/Button.h"
#include "UI/Components/Checkbox.h"
#include "UI/Components/Image.h"
#include "UI/Components/ImageToggleButton.h"
#include "UI/Components/Menu.h"
#include "UI/Components/Plot.h"
#include "UI/Components/Table.h"
#include "UI/Components/TextToggleButton.h"
#include "UI/Components/Workspace.h"
#include "UI/Dialog.h"
#include "UI/Form.h"
#include "UI/Layouts/BoxLayout.h"
#include "UI/UI.h"



namespace SE::OtdrEditor
{

    using namespace SE::Core;

    class OtdrWindow
    {
      public:
        std::string ApplicationIcon = "";
        std::string ApplicationName = "";

        Ref<Engine> mEngineLoop = nullptr;

        Ref<DotNetInstance> mApplicationInstance = nullptr;

      public:
        float HeaderHeight       = 31.0f;
        float StatusBarHeight    = 31.0f;
        float SeparatorThickness = 2.0f;
        float SideMenuWidth      = 45.0f;

        void Update( Timestep aTs );

      public:
        OtdrWindow()                     = default;
        OtdrWindow( OtdrWindow const & ) = default;
        OtdrWindow( Ref<IGraphicContext> aGraphicContext, Ref<UIContext> mUIOverlay );

        ~OtdrWindow() = default;

        bool        Display();
        bool        RenderMainMenu();
        math::ivec2 GetWorkspaceAreaSize();

        void UpdateFramerate( Timestep ts );

        void ConfigureUI();

        UIMenu mMainMenu;

      private:
        Ref<IGraphicContext> mGraphicContext;
        Ref<UIContext>        mUIOverlay;

        uint32_t mFrameCounter = 0;
        float    mFpsTimer     = 0.0f;
        uint32_t mLastFPS      = 0;

        math::ivec2 mWorkspaceAreaSize = { 0, 0 };
        bool        mRequestQuit       = false;
    };
} // namespace SE::OtdrEditor