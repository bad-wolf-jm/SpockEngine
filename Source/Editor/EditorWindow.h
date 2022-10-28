#pragma once

#include <functional>
#include <string>
#include <vector>

#include "Core/Math/Types.h"
#include "Core/Memory.h"

// #include "LidarSensorModel/SensorDeviceBase.h"

#include "Graphics/API/GraphicContext.h"
#include "Graphics/API/Texture2D.h"
#include "Scene/Renderer/SceneRenderer.h"
#include "Scene/Scene.h"

#include "UI/UI.h"

#include "Scene/EnvironmentSampler/EnvironmentSampler.h"
#include "Scene/EnvironmentSampler/PointCloudVisualizer.h"

#include "ContentBrowser.h"
#include "SceneElementEditor.h"
#include "SceneHierarchyPanel.h"
// #include "TileFlashEditor.h"
// #include "TileLayoutEditor.h"
// #include "PhotodetectorCellEditor.h"

namespace LTSE::Editor
{

    using namespace LTSE::Core;
    using namespace LTSE::Core::UI;
    using namespace LTSE::Graphics;

    struct MenuItem
    {
        std::string           Icon  = "";
        std::string           Title = "MENU_TITLE";
        std::function<bool()> Action;
    };

    class EditorWindow
    {
      public:
        enum SimulationState
        {
            EDIT,
            RUN
        };

        enum SidePanelID
        {
            SENSOR_CONFIGURATION,
            SCENE_HIERARCHY,
            SETTINGS
        };

        enum PropertyPanelID
        {
            NONE,
            SCENE_ELEMENT_EDITOR,
            TILE_PROPERTY_EDITOR,
            TILE_LAYOUT_EDITOR,
            SENSOR_COMPONENT_EDITOR,
            SENSOR_ASSET_EDITOR
        };

      public:
        std::string ApplicationIcon = "";
        std::string ApplicationName = "";
        std::string DocumentName    = "";

        Ref<EngineLoop> mEngineLoop = nullptr;
        // Ref<SensorDeviceBase> SensorModel = nullptr;
        Ref<Scene>         World         = nullptr;
        Ref<Scene>         ActiveWorld   = nullptr;
        Ref<SceneRenderer> WorldRenderer = nullptr;

        Entity         Sensor{};
        Entity         ActiveSensor{};
        GraphicContext GraphicContext{};

        PropertyPanelID CurrentPropertyPanel = PropertyPanelID::NONE;

        // TileLayoutEditor LayoutEditor;
        // FlashAttenuationBindingPopup FlashEditor;
        // PhotodetectorCellEditor PhotodetectorEditor{};

      public:
        float HeaderHeight       = 31.0f;
        float StatusBarHeight    = 31.0f;
        float SeparatorThickness = 2.0f;
        float SideMenuWidth      = 45.0f;

        entt::delegate<void( void )> OnBeginScenario{};
        entt::delegate<void( void )> OnEndScenario{};

      public:
        EditorWindow() = default;
        EditorWindow( LTSE::Graphics::GraphicContext &aGraphicContext, Ref<UIContext> mUIOverlay );
        ~EditorWindow() = default;

        bool        Display();
        bool        RenderMainMenu();
        math::ivec2 GetWorkspaceAreaSize();

        EditorWindow &AddMenuItem( std::string l_Icon, std::string l_Title, std::function<bool()> l_Action );

        void ClearScene();
        void LoadScenario( fs::path aPath );

        void Workspace( int32_t width, int32_t height );
        void Console( int32_t width, int32_t height );
        void UpdateFramerate( Timestep ts );

        void UpdateSceneViewport( ImageHandle a_SceneViewport );

      private:
        void ConfigureUI();

      private:
        LTSE::Graphics::GraphicContext mGraphicContext;
        Ref<UIContext>                 mUIOverlay;

        std::vector<MenuItem> m_MainMenuItems;

        uint32_t m_FrameCounter = 0;
        float    m_FpsTimer     = 0.0f;
        uint32_t m_LastFPS      = 0;

        math::ivec2                    m_WorkspaceAreaSize = { 0, 0 };
        Ref<LTSE::Graphics::Texture2D> m_PlayIcon;
        ImageHandle                    m_PlayIconHandle;
        Ref<LTSE::Graphics::Texture2D> m_PauseIcon;
        ImageHandle                    m_PauseIconHandle;
        Ref<LTSE::Graphics::Texture2D> m_CameraIcon;
        ImageHandle                    m_CameraIconHandle;

        Ref<LTSE::Graphics::Texture2D> m_DefaultTextureImage;
        ImageHandle                    m_DefaultTextureImageHandle;

        SimulationState mState         = SimulationState::EDIT;
        SidePanelID     m_CurrentPanel = SidePanelID::SENSOR_CONFIGURATION;

        SceneHierarchyPanel m_SceneHierarchyPanel;
        SceneElementEditor  m_SceneElementEditor;
        ContentBrowser      mContentBrowser;

      private:
        ImageHandle m_SceneViewport{};

        std::vector<uint8_t> mTestTile;
    };
} // namespace LTSE::Editor