#pragma once

#include <functional>
#include <string>
#include <vector>

#include "Core/Math/Types.h"
#include "Core/Memory.h"

#include "Graphics/Vulkan/VkGraphicContext.h"
#include "Graphics/Vulkan/VkSampler2D.h"

#include "OtdrScene/OtdrScene.h"

#include "UI/UI.h"
#include "UI/Components/Button.h"
#include "UI/Layouts/BoxLayout.h"

#include "Mono/MonoScriptInstance.h"
#include "Editor/ContentBrowser.h"
#include "SceneHierarchyPanel.h"

namespace SE::OtdrEditor
{

    class OtdrWindow
    {
      public:
        enum SimulationState
        {
            EDIT,
            RUN
        };

      public:
        std::string ApplicationIcon = "";
        std::string ApplicationName = "";
        std::string DocumentName    = "";

        fs::path mMaterialsPath = "";
        fs::path mModelsPath    = "";

        Ref<Engine> mEngineLoop = nullptr;

        Ref<OtdrScene> mWorld       = nullptr;
        Ref<OtdrScene> mActiveWorld = nullptr;

        Ref<MonoScriptInstance> mCurrentScript{};
        bool mCurrentScriptIsRunning{};

      public:
        float HeaderHeight       = 31.0f;
        float StatusBarHeight    = 31.0f;
        float SeparatorThickness = 2.0f;
        float SideMenuWidth      = 45.0f;

        entt::delegate<void( void )> OnBeginScenario{};
        entt::delegate<void( void )> OnEndScenario{};

        void Update(Timestep aTs);

      public:
        OtdrWindow() = default;
        OtdrWindow(OtdrWindow const&) = default;
        OtdrWindow( Ref<VkGraphicContext> aGraphicContext, Ref<UIContext> mUIOverlay );

        ~OtdrWindow() = default;

        bool        Display();
        bool        RenderMainMenu();
        math::ivec2 GetWorkspaceAreaSize();

        void Workspace( int32_t width, int32_t height );
        void Console( int32_t width, int32_t height );
        void UpdateFramerate( Timestep ts );
        
        void ConfigureUI();

        void LoadIOlmData(fs::path aPath);

      private:
        UIButton mTestButton0;
        UIButton mTestButton1;
        UIButton mTestButton2;

        BoxLayout mTestLayout;


      private:
        OtdrSceneHierarchyPanel mSceneHierarchyPanel;

      private:
        Ref<VkGraphicContext> mGraphicContext;
        Ref<UIContext>        mUIOverlay;

        uint32_t mFrameCounter = 0;
        float    mFpsTimer     = 0.0f;
        uint32_t mLastFPS      = 0;

        math::ivec2      mWorkspaceAreaSize = { 0, 0 };
        Ref<VkSampler2D> mPlayIcon;
        ImageHandle      mPlayIconHandle;
        Ref<VkSampler2D> mPauseIcon;
        ImageHandle      mPauseIconHandle;

        SimulationState mState       = SimulationState::EDIT;
        fs::path        mCurrentPath = "";
    };
} // namespace SE::OtdrEditor