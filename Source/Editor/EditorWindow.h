#pragma once

#include <functional>
#include <string>
#include <vector>

#include "Core/Math/Types.h"
#include "Core/Memory.h"

#include "Graphics/API.h"

#include "Renderer/SceneRenderer.h"

// #include "Scene/Renderer/DeferredSceneRenderer.h"
// #include "Scene/Renderer/ForwardSceneRenderer.h"
// #include "Scene/Renderer/RayTracing/RayTracingRenderer.h"
#include "Scene/Scene.h"

#include "UI/UI.h"

#include "Scene/EnvironmentSampler/EnvironmentSampler.h"
#include "Scene/EnvironmentSampler/PointCloudVisualizer.h"

#include "ContentBrowser.h"
#include "SceneElementEditor.h"
#include "SceneHierarchyPanel.h"

namespace SE::Editor
{

    struct MenuItem
    {
        string_t           Icon  = "";
        string_t           Title = "MENU_TITLE";
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
        string_t ApplicationIcon = "";
        string_t ApplicationName = "";
        string_t DocumentName    = "";

        fs::path mMaterialsPath = "";
        fs::path mModelsPath    = "";

        ref_t<Engine>        mEngineLoop   = nullptr;
        ref_t<Scene>         World         = nullptr;
        ref_t<Scene>         ActiveWorld   = nullptr;
        ref_t<SceneRenderer> WorldRenderer = nullptr;

        // ref_t<ForwardSceneRenderer> WorldRenderer = nullptr;
        Entity Sensor{};
        Entity ActiveSensor{};

        PropertyPanelID     CurrentPropertyPanel = PropertyPanelID::NONE;
        // ref_t<DotNetInstance> mApplicationInstance = nullptr;

      public:
        float HeaderHeight       = 31.0f;
        float StatusBarHeight    = 31.0f;
        float SeparatorThickness = 2.0f;
        float SideMenuWidth      = 45.0f;

        entt::delegate<void( void )> OnBeginScenario{};
        entt::delegate<void( void )> OnEndScenario{};

      public:
        EditorWindow() = default;
        EditorWindow( ref_t<IGraphicContext> aGraphicContext, ref_t<UIContext> mUIOverlay );

        ~EditorWindow() = default;

        bool        Display();
        bool        RenderMainMenu();
        math::ivec2 GetWorkspaceAreaSize();
        math::ivec2 GetNewWorkspaceAreaSize();

        EditorWindow &AddMenuItem( string_t l_Icon, string_t l_Title, std::function<bool()> l_Action );

        void ClearScene();
        void LoadScenario( fs::path aPath );
        void ImportModel( fs::path aPath );

        void Workspace( int32_t width, int32_t height );
        void WorkspaceNew( int32_t width, int32_t height );
        void Console( int32_t width, int32_t height );
        void UpdateFramerate( Timestep ts );

        void UpdateSceneViewport( ImageHandle a_SceneViewport );
        void UpdateNewSceneViewport( ImageHandle a_SceneViewport );
        void UpdateSceneViewport_deferred( ImageHandle a_SceneViewport );

      private:
        void ConfigureUI();

      private:
        ref_t<IGraphicContext> mGraphicContext;

        ref_t<UIContext> mUIOverlay;

        vector_t<MenuItem> m_MainMenuItems;

        uint32_t m_FrameCounter = 0;
        float    m_FpsTimer     = 0.0f;
        uint32_t m_LastFPS      = 0;

        math::ivec2     m_WorkspaceAreaSize    = { 0, 0 };
        math::ivec2     m_NewWorkspaceAreaSize = { 0, 0 };
        ref_t<ISampler2D> m_PlayIcon;
        ImageHandle     m_PlayIconHandle;
        ref_t<ISampler2D> m_PauseIcon;
        ImageHandle     m_PauseIconHandle;
        ref_t<ISampler2D> m_CameraIcon;
        ImageHandle     m_CameraIconHandle;

        ref_t<ISampler2D> m_DefaultTextureImage;
        ImageHandle     m_DefaultTextureImageHandle;

        SimulationState mState         = SimulationState::EDIT;
        SidePanelID     m_CurrentPanel = SidePanelID::SENSOR_CONFIGURATION;

        SceneHierarchyPanel m_SceneHierarchyPanel;
        SceneElementEditor  m_SceneElementEditor;
        ContentBrowser      mContentBrowser;

      private:
        ImageHandle m_SceneViewport{};
        ImageHandle m_SceneViewport_deferred{};
        ImageHandle m_SceneViewport_new{};

        vector_t<uint8_t> mTestTile;
    };
} // namespace SE::Editor