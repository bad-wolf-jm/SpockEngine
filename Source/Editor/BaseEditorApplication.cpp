#include "BaseEditorApplication.h"

#include "Core/File.h"
#include "yaml-cpp/yaml.h"
#include <fstream>

#include <direct.h>
#include <iostream>
#include <limits.h>
#include <string>

#include "UI/UI.h"
#include "UI/Widgets.h"

#include "Core/Cuda/CudaBuffer.h"
#include "Core/Cuda/MultiTensor.h"
#include "Core/Cuda/ExternalMemory.h"

#include "Scene/Components.h"
#include "Scene/Importer/glTFImporter.h"

namespace LTSE::Editor
{

    using namespace LTSE::Core;
    using namespace LTSE::Cuda;
    using namespace LTSE::Core::EntityComponentSystem::Components;

    BaseEditorApplication::BaseEditorApplication() { mEngineLoop = New<EngineLoop>(); }

    void BaseEditorApplication::LoadConfiguration()
    {
        YAML::Node l_RootNode   = YAML::LoadFile( ConfigurationFile.string() );
        YAML::Node &l_ImGuiInit = l_RootNode["application"]["imgui_initialization"];
        if( !l_ImGuiInit.IsNull() )
            ImGuiIniFile = l_ImGuiInit.as<std::string>();

        YAML::Node &l_WindowProperties = l_RootNode["application"]["window_properties"];
        if( !l_WindowProperties.IsNull() )
        {
            WindowPosition = { l_WindowProperties["x"].as<int>(), l_WindowProperties["y"].as<int>() };
            WindowSize     = { l_WindowProperties["width"].as<int>(), l_WindowProperties["height"].as<int>() };
        }
    }

    void BaseEditorApplication::SaveConfiguration()
    {
        YAML::Emitter out;
        out << YAML::BeginMap;
        {
            out << YAML::Key << "application" << YAML::Value;
            out << YAML::BeginMap;
            {
                if( ImGuiIniFile.empty() )
                    out << YAML::Key << "imgui_initialization" << YAML::Value << YAML::Null;
                else
                    out << YAML::Key << "imgui_initialization" << YAML::Value << ImGuiIniFile;

                out << YAML::Key << "window_properties" << YAML::Value << YAML::Flow;
                {
                    out << YAML::BeginMap;
                    out << YAML::Key << "width" << YAML::Value << WindowSize.x;
                    out << YAML::Key << "height" << YAML::Value << WindowSize.y;
                    out << YAML::Key << "x" << YAML::Value << WindowPosition.x;
                    out << YAML::Key << "y" << YAML::Value << WindowPosition.y;
                    out << YAML::EndMap;
                }
            }
            out << YAML::EndMap;
        }
        out << YAML::EndMap;
        std::ofstream fout( ConfigurationFile );
        fout << out.c_str();
    }

    void BaseEditorApplication::RenderScene()
    {
        m_ViewportRenderContext.BeginRender();

        if( m_ViewportRenderContext )
            m_WorldRenderer->Render( m_ViewportRenderContext );

        m_ViewportRenderContext.EndRender();
    }

    void BaseEditorApplication::Update( Timestep ts )
    {
        mEditorWindow.ActiveWorld->Update( ts );
        mEditorWindow.UpdateFramerate( ts );
    }

    void BaseEditorApplication::RebuildOutputFramebuffer()
    {
        if( m_ViewportWidth == 0 || m_ViewportHeight == 0 )
            return;

        if( !m_OffscreenRenderTarget )
        {
            OffscreenRenderTargetDescription l_RenderTargetCI{};
            l_RenderTargetCI.OutputSize  = { m_ViewportWidth, m_ViewportHeight };
            l_RenderTargetCI.SampleCount = 4;
            l_RenderTargetCI.Sampled     = true;
            m_OffscreenRenderTarget      = New<OffscreenRenderTarget>( mEngineLoop->GetGraphicContext(), l_RenderTargetCI );
            m_ViewportRenderContext      = LTSE::Graphics::RenderContext( mEngineLoop->GetGraphicContext(), m_OffscreenRenderTarget );
        }
        else
        {
            m_OffscreenRenderTarget->Resize( m_ViewportWidth, m_ViewportHeight );
        }

        m_OffscreenRenderTargetTexture = New<Graphics::Texture2D>( mEngineLoop->GetGraphicContext(), TextureDescription{}, m_OffscreenRenderTarget->GetOutputImage() );

        if( !m_OffscreenRenderTargetDisplayHandle.Handle )
        {
            m_OffscreenRenderTargetDisplayHandle = mEngineLoop->UIContext()->CreateTextureHandle( m_OffscreenRenderTargetTexture );
            mEditorWindow.UpdateSceneViewport( m_OffscreenRenderTargetDisplayHandle );
        }
        else
        {
            m_OffscreenRenderTargetDisplayHandle.Handle->Write( m_OffscreenRenderTargetTexture, 0 );
        }

        if( m_WorldRenderer )
        {
            m_WorldRenderer->View.Projection = math::Perspective( 90.0_degf, static_cast<float>( m_ViewportWidth ) / static_cast<float>( m_ViewportHeight ), 0.01f, 100000.0f );
            m_WorldRenderer->View.Projection[1][1] *= -1.0f;
        }
    }

    bool BaseEditorApplication::RenderUI( ImGuiIO &io )
    {
        bool o_RequestQuit = false;
        if( m_ShouldRebuildViewport )
        {
            RebuildOutputFramebuffer();
            m_ShouldRebuildViewport = false;
        }

        mEditorWindow.mEngineLoop = mEngineLoop;
        // mEditorWindow.SensorModel    = m_SensorController;
        mEditorWindow.WorldRenderer  = m_WorldRenderer;
        mEditorWindow.GraphicContext = mEngineLoop->GetGraphicContext();

        o_RequestQuit = mEditorWindow.Display();
        OnUI();

        auto l_WorkspaceAreaSize = mEditorWindow.GetWorkspaceAreaSize();
        if( ( m_ViewportWidth != l_WorkspaceAreaSize.x ) || ( m_ViewportHeight != l_WorkspaceAreaSize.y ) )
        {
            m_ViewportWidth         = l_WorkspaceAreaSize.x;
            m_ViewportHeight        = l_WorkspaceAreaSize.y;
            m_ShouldRebuildViewport = true;
        }

        if( o_RequestQuit )
            return true;

        return false;
    }

    void BaseEditorApplication::Init()
    {
        // m_SensorController = a_SensorToControl;

        mEngineLoop->SetApplicationName( ApplicationName );
        mEngineLoop->PreInit( 0, nullptr );

        // get cwd
        {
            char buff[MAX_PATH];
            _getcwd( buff, MAX_PATH );
            fs::path lCwd = std::string( buff );

            LTSE::Logging::Info( "Current working directory is: '{}'", lCwd.string() );

            // ConfigurationRoot = lCwd;
            // if( fs::exists( lCwd / "SensorConfiguration.yaml" ) )
            // {
            //     LTSE::Logging::Info( "Current working directory set to: '{}'", ConfigurationRoot.string() );
            // }
            // else
            // {
            //     while( !ConfigurationRoot.empty() && ( ConfigurationRoot != ConfigurationRoot.root_path() ) && !fs::exists( ConfigurationRoot / "SensorConfiguration.yaml" ) )
            //     {
            //         ConfigurationRoot = ConfigurationRoot.parent_path();
            //         LTSE::Logging::Info( "Looking for sensor configuration in: '{}'", ConfigurationRoot.string() );
            //     }

            //     if( fs::exists( ConfigurationRoot / "SensorConfiguration.yaml" ) )
            //     {
            //         LTSE::Logging::Info( "Current working directory set to: '{}'", ConfigurationRoot.string() );
            //     }
            //     else
            //     {
            //         ConfigurationRoot = lCwd;
            //         LTSE::Logging::Error( "The file 'SensorConfiguration.yaml' was not found in this folder or any of its parents." );
            //         exit( 2 );
            //     }
            // }
        }

        // Create Saved, Saved/Logs
        if( !fs::exists( ConfigurationRoot / "Saved" / "Logs" ) )
            fs::create_directories( ConfigurationRoot / "Saved" / "Logs" );

        // Configure logger to send messages to saved/logs/EditorLogs.txt
        LTSE::Logging::Info( "Log file will be written to '{}'", ( ConfigurationRoot / "Saved" / "Logs" / "EditorLogs.txt" ).string() );
        LTSE::Logging::SetLogOutputFile( ConfigurationRoot / "Saved" / "Logs" / "EditorLogs.txt" );

        ConfigurationFile = ConfigurationRoot / "EditorConfiguration.yaml";
        if( fs::exists( ConfigurationFile ) )
            LoadConfiguration();
        else
            SaveConfiguration();

        // SensorConfigurationFile = ConfigurationRoot / "SensorConfiguration.yaml";
        // if( fs::exists( SensorConfigurationFile ) )
        // {
        //     LTSE::Logging::Info( "Loading sensor configuration from '{}'", SensorConfigurationFile.string() );
        // }
        // else
        // {
        //     LTSE::Logging::Info( "Sensor configuration not found.", SensorConfigurationFile.string() );
        // }
        // LoadSensorConfiguration();

        mEngineLoop->SetInitialWindowSize( WindowSize );
        mEngineLoop->SetInitialWindowPosition( WindowPosition );
        mEngineLoop->SetImGuiConfigurationFile( ( ConfigurationRoot / "Saved" / "imgui.ini" ).string() );
        mEngineLoop->Init();

        mEngineLoop->RenderDelegate.connect<&BaseEditorApplication::RenderScene>( *this );
        mEngineLoop->UIDelegate.connect<&BaseEditorApplication::RenderUI>( *this );
        mEngineLoop->UpdateDelegate.connect<&BaseEditorApplication::Update>( *this );

        mEditorWindow                 = EditorWindow( mEngineLoop->GetGraphicContext(), mEngineLoop->UIContext() );
        mEditorWindow.ApplicationIcon = ICON_FA_CODEPEN;

        RebuildOutputFramebuffer();
        m_World         = New<Scene>( mEngineLoop->GetGraphicContext(), mEngineLoop->UIContext() );
        m_WorldRenderer = New<SceneRenderer>( m_World, m_ViewportRenderContext, m_OffscreenRenderTarget->GetRenderPass() );

        mEditorWindow.World       = m_World;
        mEditorWindow.ActiveWorld = m_World;

        // {
        //     // Add sensor entity to the scene
        //     mEditorWindow.Sensor = m_World->Create( "Sensor", m_World->Root );
        //     mEditorWindow.Sensor.Add<sLocalTransformComponent>();
        //     mEditorWindow.Sensor.Add<EnvironmentSampler::sCreateInfo>();

        //     AcquisitionSpecification lAcqCreateInfo{};
        //     lAcqCreateInfo.mBasePoints   = 100;
        //     lAcqCreateInfo.mOversampling = 1;
        //     mEditorWindow.Sensor.Add<AcquisitionSpecification>( lAcqCreateInfo );
        //     mEditorWindow.Sensor.Add<sBehaviourComponent>();

        //     // Add a particle system to the sensor to display the point cloud
        //     auto &l_SensorPointCloud = mEditorWindow.Sensor.Add<sParticleSystemComponent>();

        //     // Create particle renderer for the point cloud
        //     auto &l_SensorPointCloudRenderer    = mEditorWindow.Sensor.Add<RendererComponent>();
        //     l_SensorPointCloudRenderer.Material = m_World->CreateEntity( "ParticleSystemMaterial" );
        //     l_SensorPointCloudRenderer.Material.Add<sParticleShaderComponent>();

        //     mEditorWindow.ActiveSensor = mEditorWindow.Sensor;
        // }

        m_WorldRenderer->RenderCoordinateGrid = true;
        m_WorldRenderer->View.CameraPosition  = math::vec3( 0.0f, 1.0f, 7.5f );
        m_WorldRenderer->View.ModelFraming    = math::mat4( 0.5f );
        m_WorldRenderer->View.View            = math::Inverse( math::Translate( math::mat4( 1.0f ), m_WorldRenderer->View.CameraPosition ) );
    }

    uint32_t BaseEditorApplication::Run()
    {
        while( mEngineLoop->Tick() )
        {
        }

        SaveConfiguration();

        mEngineLoop->Shutdown();

        return 0;
    }

} // namespace LTSE::Editor
