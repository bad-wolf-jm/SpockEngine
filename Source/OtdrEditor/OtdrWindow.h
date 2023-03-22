#pragma once

#include <functional>
#include <string>
#include <vector>

#include "Core/Math/Types.h"
#include "Core/Memory.h"

#include "Graphics/Vulkan/VkGraphicContext.h"
#include "Graphics/Vulkan/VkSampler2D.h"

#include "OtdrScene/OtdrScene.h"

#include "UI/Components/Button.h"
#include "UI/Components/Checkbox.h"
#include "UI/Components/Image.h"
#include "UI/Components/ImageToggleButton.h"
#include "UI/Components/Plot.h"
#include "UI/Components/Table.h"
#include "UI/Components/TextToggleButton.h"
#include "UI/Form.h"
#include "UI/Layouts/BoxLayout.h"
#include "UI/UI.h"

#include "AcquisitionData.h"
#include "Editor/ContentBrowser.h"
#include "Enums.h"
#include "EventOverview.h"
#include "LinkElementTable.h"
#include "LinkElementTracePlot.h"
#include "MeasurementOverview.h"
#include "Mono/MonoScriptInstance.h"
#include "MultiPulseEventTable.h"
#include "TestFailResultTable.h"
#include "Workspace.h"
#include "MonoClassHierarchy.h"

namespace SE::OtdrEditor
{

    using namespace SE::Core;

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

        Ref<MonoScriptInstance> mCurrentScript{};
        bool                    mCurrentScriptIsRunning{};

      public:
        float HeaderHeight       = 31.0f;
        float StatusBarHeight    = 31.0f;
        float SeparatorThickness = 2.0f;
        float SideMenuWidth      = 45.0f;

        entt::delegate<void( void )> OnBeginScenario{};
        entt::delegate<void( void )> OnEndScenario{};

        void Update( Timestep aTs );

      public:
        OtdrWindow()                     = default;
        OtdrWindow( OtdrWindow const & ) = default;
        OtdrWindow( Ref<VkGraphicContext> aGraphicContext, Ref<UIContext> mUIOverlay );

        ~OtdrWindow() = default;

        bool        Display();
        bool        RenderMainMenu();
        math::ivec2 GetWorkspaceAreaSize();

        void Workspace( int32_t width, int32_t height );
        void Console( int32_t width, int32_t height );
        void UpdateFramerate( Timestep ts );

        void ConfigureUI();

        void LoadIOlmData( fs::path aPath, bool aReanalyse = false );
        void LoadTestReport( fs::path aPath );

        Ref<MonoScriptInstance> mDataInstance = nullptr;

        UILinkElementTracePlot mTracePlot;

        Ref<UIMultiPulseEventTable> mEventTable;
        Ref<UIMultiPulseEventTable> mEventTable1;
        Ref<UILinkElementTable>     mLinkElementTable;
        Ref<UILinkElementTable>     mLinkElementTable1;
        Ref<UITestFailResultTable>  mTestFailResultTable;

        MeasurementOverview mMeasurementOverview;
        EventOverview       mEventOverview;
        AcquisitionData     mAcquisitionDataOverview;

      private:
        OtdrWorkspaceWindow mWorkspaceArea;
        MonoClassHierarchy mMonoClasses;

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