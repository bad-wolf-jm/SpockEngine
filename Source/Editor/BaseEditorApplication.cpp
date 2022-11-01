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
#include "Core/Cuda/ExternalMemory.h"
#include "Core/Cuda/MultiTensor.h"

#include "Core/Textures/ColorFormat.h"
#include "Scene/Components.h"
#include "Scene/Importer/glTFImporter.h"

#include "Mono/Manager.h"

namespace LTSE::Editor
{

    using namespace LTSE::Core;
    using namespace LTSE::Cuda;
    using namespace LTSE::Core::EntityComponentSystem::Components;

    BaseEditorApplication::BaseEditorApplication() { mEngineLoop = New<EngineLoop>(); }

    void BaseEditorApplication::LoadConfiguration()
    {
        YAML::Node  l_RootNode  = YAML::LoadFile( ConfigurationFile.string() );
        YAML::Node &l_ImGuiInit = l_RootNode["application"]["imgui_initialization"];
        // if( !l_ImGuiInit.IsNull() ) ImGuiIniFile = l_ImGuiInit.as<std::string>();

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
                // if( ImGuiIniFile.empty() )
                    // out << YAML::Key << "imgui_initialization" << YAML::Value << YAML::Null;
                // else
                    // out << YAML::Key << "imgui_initialization" << YAML::Value << ImGuiIniFile;

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
        mDeferredRenderer->Render();
        mForwardRenderer->Render();
    }

    void BaseEditorApplication::Update( Timestep ts )
    {
        mEditorWindow.ActiveWorld->Update( ts );
        mEditorWindow.UpdateFramerate( ts );
        mDeferredRenderer->Update( mEditorWindow.ActiveWorld );
        mForwardRenderer->Update( mEditorWindow.ActiveWorld );
    }

    void BaseEditorApplication::RebuildOutputFramebuffer()
    {
        if( mViewportWidth == 0 || mViewportHeight == 0 ) return;

        mDeferredRenderer->ResizeOutput( mViewportWidth, mViewportHeight );
        mForwardRenderer->ResizeOutput( mViewportWidth, mViewportHeight );

        mOffscreenRenderTargetTexture =
            New<Graphics::Texture2D>( mEngineLoop->GetGraphicContext(), TextureDescription{}, mForwardRenderer->GetOutputImage() );

        if( !mOffscreenRenderTargetDisplayHandle.Handle )
        {
            mOffscreenRenderTargetDisplayHandle = mEngineLoop->UIContext()->CreateTextureHandle( mOffscreenRenderTargetTexture );
            mEditorWindow.UpdateSceneViewport( mOffscreenRenderTargetDisplayHandle );
        }
        else
        {
            mOffscreenRenderTargetDisplayHandle.Handle->Write( mOffscreenRenderTargetTexture, 0 );
        }

        mDeferredRenderTargetTexture =
            New<Graphics::Texture2D>( mEngineLoop->GetGraphicContext(), TextureDescription{}, mDeferredRenderer->GetOutputImage() );

        if( !mDeferredRenderTargetDisplayHandle.Handle )
        {
            mDeferredRenderTargetDisplayHandle = mEngineLoop->UIContext()->CreateTextureHandle( mDeferredRenderTargetTexture );
            mEditorWindow.UpdateSceneViewport_deferred( mDeferredRenderTargetDisplayHandle );
        }
        else
        {
            mDeferredRenderTargetDisplayHandle.Handle->Write( mDeferredRenderTargetTexture, 0 );
        }

        mDeferredRenderer->SetProjection( math::Perspective(
            90.0_degf, static_cast<float>( mViewportWidth ) / static_cast<float>( mViewportHeight ), 0.01f, 100000.0f ) );
        mForwardRenderer->SetProjection( math::Perspective(
            90.0_degf, static_cast<float>( mViewportWidth ) / static_cast<float>( mViewportHeight ), 0.01f, 100000.0f ) );
    }

    bool BaseEditorApplication::RenderUI( ImGuiIO &io )
    {
        bool o_RequestQuit = false;
        if( mShouldRebuildViewport )
        {
            RebuildOutputFramebuffer();
            mShouldRebuildViewport = false;
        }

        mEditorWindow.mEngineLoop = mEngineLoop;
        // mEditorWindow.SensorModel    = m_SensorController;
        mEditorWindow.WorldRenderer = mForwardRenderer;
        mEditorWindow.DefRenderer = mDeferredRenderer;
        mEditorWindow.GraphicContext = mEngineLoop->GetGraphicContext();

        o_RequestQuit = mEditorWindow.Display();
        OnUI();

        auto l_WorkspaceAreaSize = mEditorWindow.GetWorkspaceAreaSize();
        if( ( mViewportWidth != l_WorkspaceAreaSize.x ) || ( mViewportHeight != l_WorkspaceAreaSize.y ) )
        {
            mViewportWidth         = l_WorkspaceAreaSize.x;
            mViewportHeight        = l_WorkspaceAreaSize.y;
            mShouldRebuildViewport = true;
        }

        if( o_RequestQuit ) return true;

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
            //     while( !ConfigurationRoot.empty() && ( ConfigurationRoot != ConfigurationRoot.root_path() ) && !fs::exists(
            //     ConfigurationRoot / "SensorConfiguration.yaml" ) )
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
            //         LTSE::Logging::Error( "The file 'SensorConfiguration.yaml' was not found in this folder or any of its parents."
            //         ); exit( 2 );
            //     }
            // }
        }

        // Create Saved, Saved/Logs
        if( !fs::exists( ConfigurationRoot / "Saved" / "Logs" ) ) fs::create_directories( ConfigurationRoot / "Saved" / "Logs" );

        // Configure logger to send messages to saved/logs/EditorLogs.txt
        LTSE::Logging::Info(
            "Log file will be written to '{}'", ( ConfigurationRoot / "Saved" / "Logs" / "EditorLogs.txt" ).string() );
        LTSE::Logging::SetLogOutputFile( ConfigurationRoot / "Saved" / "Logs" / "EditorLogs.txt" );

        ConfigurationFile = ConfigurationRoot / "EditorConfiguration.yaml";
        if( fs::exists( ConfigurationFile ) )
            LoadConfiguration();
        else
            SaveConfiguration();

        ScriptManager::Initialize();

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

        mWorld            = New<Scene>( mEngineLoop->GetGraphicContext(), mEngineLoop->UIContext() );
        mDeferredRenderer = New<DeferredRenderer>( mEngineLoop->GetGraphicContext(), eColorFormat::RGBA8_UNORM, 1 );
        mForwardRenderer  = New<ForwardSceneRenderer>( mEngineLoop->GetGraphicContext(), eColorFormat::RGBA8_UNORM, 4 );
        RebuildOutputFramebuffer();

        mForwardRenderer->Update( mWorld );
        mDeferredRenderer->Update( mWorld );

        mEditorWindow.World       = mWorld;
        mEditorWindow.ActiveWorld = mWorld;

        {
            // Add sensor entity to the scene
            mEditorWindow.Sensor = mWorld->Create( "Sensor", mWorld->Root );
            mEditorWindow.Sensor.Add<sNodeTransformComponent>();
            mEditorWindow.Sensor.Add<sBehaviourComponent>();

            // Add a particle system to the sensor to display the point cloud
            auto &l_SensorPointCloud = mEditorWindow.Sensor.Add<sParticleSystemComponent>();

            // Create particle renderer for the point cloud
            auto &l_SensorPointCloudRenderer = mEditorWindow.Sensor.Add<sParticleShaderComponent>();

            mEditorWindow.ActiveSensor = mEditorWindow.Sensor;
        }

        mForwardRenderer->RenderCoordinateGrid = true;
        mForwardRenderer->View.CameraPosition  = math::vec3( 0.0f, 1.0f, 7.5f );
        mForwardRenderer->View.ModelFraming    = math::mat4( 0.5f );
        mForwardRenderer->View.View = math::Inverse( math::Translate( math::mat4( 1.0f ), mForwardRenderer->View.CameraPosition ) );
    }

    uint32_t BaseEditorApplication::Run()
    {
        while( mEngineLoop->Tick() )
        {
        }

        SaveConfiguration();

        mEngineLoop->Shutdown();
    
        ScriptManager::Shutdown();

        return 0;
    }

} // namespace LTSE::Editor
