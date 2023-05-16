#pragma once

#include <functional>
#include <string>
#include <vector>

#include "Core/Math/Types.h"
#include "Core/Memory.h"

#include "Graphics/Vulkan/VkGraphicContext.h"

#include "Graphics/Vulkan/ISampler2D.h"

#include "Scene/Renderer/DeferredSceneRenderer.h"
#include "Scene/Renderer/ForwardSceneRenderer.h"
#include "Scene/Renderer/RayTracing/RayTracingRenderer.h"
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

        fs::path mMaterialsPath = "";
        fs::path mModelsPath    = "";

        Ref<Engine>           mEngineLoop   = nullptr;
        Ref<Scene>            World         = nullptr;
        Ref<Scene>            ActiveWorld   = nullptr;
        Ref<DeferredRenderer> WorldRenderer = nullptr;

        Entity Sensor{};
        Entity ActiveSensor{};

        PropertyPanelID CurrentPropertyPanel = PropertyPanelID::NONE;

      public:
        float HeaderHeight       = 31.0f;
        float StatusBarHeight    = 31.0f;
        float SeparatorThickness = 2.0f;
        float SideMenuWidth      = 45.0f;

        entt::delegate<void( void )> OnBeginScenario{};
        entt::delegate<void( void )> OnEndScenario{};

      public:
        EditorWindow() = default;
        EditorWindow( Ref<VkGraphicContext> aGraphicContext, Ref<UIContext> mUIOverlay );

        ~EditorWindow() = default;

        bool        Display();
        bool        RenderMainMenu();
        math::ivec2 GetWorkspaceAreaSize();

        EditorWindow &AddMenuItem( std::string l_Icon, std::string l_Title, std::function<bool()> l_Action );

        void ClearScene();
        void LoadScenario( fs::path aPath );
        void ImportModel( fs::path aPath );

        void Workspace( int32_t width, int32_t height );
        void Console( int32_t width, int32_t height );
        void UpdateFramerate( Timestep ts );

        void UpdateSceneViewport( ImageHandle a_SceneViewport );
        void UpdateSceneViewport_deferred( ImageHandle a_SceneViewport );

      private:
        void ConfigureUI();

      private:
        Ref<VkGraphicContext> mGraphicContext;

        Ref<UIContext> mUIOverlay;

        std::vector<MenuItem> m_MainMenuItems;

        uint32_t m_FrameCounter = 0;
        float    m_FpsTimer     = 0.0f;
        uint32_t m_LastFPS      = 0;

        math::ivec2     m_WorkspaceAreaSize = { 0, 0 };
        Ref<ISampler2D> m_PlayIcon;
        ImageHandle     m_PlayIconHandle;
        Ref<ISampler2D> m_PauseIcon;
        ImageHandle     m_PauseIconHandle;
        Ref<ISampler2D> m_CameraIcon;
        ImageHandle     m_CameraIconHandle;

        Ref<ISampler2D> m_DefaultTextureImage;
        ImageHandle     m_DefaultTextureImageHandle;

        SimulationState mState         = SimulationState::EDIT;
        SidePanelID     m_CurrentPanel = SidePanelID::SENSOR_CONFIGURATION;

        SceneHierarchyPanel m_SceneHierarchyPanel;
        SceneElementEditor  m_SceneElementEditor;
        ContentBrowser      mContentBrowser;

      private:
        ImageHandle m_SceneViewport{};
        ImageHandle m_SceneViewport_deferred{};

        std::vector<uint8_t> mTestTile;
    };
} // namespace SE::Editor