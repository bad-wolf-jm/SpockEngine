#include <argparse/argparse.hpp>
#include <filesystem>
#include <fstream>

#include <direct.h>
#include <iostream>
#include <limits.h>
#include <string>

#include "Core/EntityRegistry/ScriptableEntity.h"
#include "Core/GraphicContext//UI/UIContext.h"
#include "Core/Logging.h"
#include "Core/Math/Types.h"
#include "Core/Memory.h"
#include "Core/Platform/EngineLoop.h"

#include "Core/GraphicContext//GraphicContext.h"
#include "Scene/EnvironmentSampler/EnvironmentSampler.h"
#include "Scene/EnvironmentSampler/PointCloudVisualizer.h"
#include "Scene/Renderer/SceneRenderer.h"
#include "Scene/Scene.h"

// #include "LidarSensorModel/SensorDeviceBase.h"
#include "TensorOps/Scope.h"

#include "Editor/EditorWindow.h"
#include "UI/UI.h"
#include "UI/Widgets.h"

using namespace LTSE::Core;
using namespace LTSE::Graphics;
using namespace LTSE::Editor;
using namespace LTSE::Core::UI;
using namespace LTSE::SensorModel;
using namespace LTSE::SensorModel::Dev;

namespace fs = std::filesystem;

void LoadConfiguration( fs::path ConfigurationFile )
{
    YAML::Node l_RootNode = YAML::LoadFile( ConfigurationFile.string() );

    YAML::Node &l_WindowProperties = l_RootNode["application"]["window_properties"];
    if( !l_WindowProperties.IsNull() )
    {
        // WindowPosition = { l_WindowProperties["x"].as<int>(), l_WindowProperties["y"].as<int>() };
        // WindowSize     = { l_WindowProperties["width"].as<int>(), l_WindowProperties["height"].as<int>() };
    }
}

void SaveConfiguration( fs::path ConfigurationFile )
{
    YAML::Emitter out;
    out << YAML::BeginMap;
    {
        out << YAML::Key << "application" << YAML::Value;
        out << YAML::BeginMap;
        {
            // if( ImGuiIniFile.empty() )
            //     out << YAML::Key << "imgui_initialization" << YAML::Value << YAML::Null;
            // else
            //     out << YAML::Key << "imgui_initialization" << YAML::Value << ImGuiIniFile;

            out << YAML::Key << "window_properties" << YAML::Value << YAML::Flow;
            {
                out << YAML::BeginMap;
                // out << YAML::Key << "width" << YAML::Value << WindowSize.x;
                // out << YAML::Key << "height" << YAML::Value << WindowSize.y;
                // out << YAML::Key << "x" << YAML::Value << WindowPosition.x;
                // out << YAML::Key << "y" << YAML::Value << WindowPosition.y;
                out << YAML::EndMap;
            }
        }
        out << YAML::EndMap;
    }
    out << YAML::EndMap;
    std::ofstream fout( ConfigurationFile );
    fout << out.c_str();
}

Ref<argparse::ArgumentParser> ParseCommandLine( int argc, char **argv )
{
    auto lProgramArguments = New<argparse::ArgumentParser>( "bin2ktx" );

    lProgramArguments->add_argument( "-p", "--project" ).help( "Specify input file" );
    lProgramArguments->add_argument( "-s", "--scenario" ).help( "Specify input file" );
    lProgramArguments->add_argument( "-x", "--res_x" ).help( "Specify output file" );
    lProgramArguments->add_argument( "-y", "--res_y" ).help( "Specify output file" );
    lProgramArguments->add_argument( "-M", "--mono_runtime" ).help( "Specify output file" );
    lProgramArguments->add_argument( "-L", "--log_level" ).help( "Specify output file" );

    try
    {
        lProgramArguments->parse_args( argc, argv );

        return lProgramArguments;
    }
    catch( const std::runtime_error &err )
    {
        std::cerr << err.what() << std::endl;
        std::cerr << lProgramArguments;

        return nullptr;
    }
}

int main( int argc, char **argv )
{
    Ref<EngineLoop> mEngineLoop = New<EngineLoop>();

    std::string ApplicationName = "Sensor Model Editor";
    mEngineLoop->SetApplicationName( ApplicationName );
    mEngineLoop->PreInit( 0, nullptr );

    // get cwd
    fs::path ConfigurationRoot = "";
    {
        char buff[MAX_PATH];
        _getcwd( buff, MAX_PATH );
        fs::path lCwd = std::string( buff );

        LTSE::Logging::Info( "Current working directory is: '{}'", lCwd.string() );

        ConfigurationRoot = lCwd;
        if( fs::exists( lCwd / "Simulation.yaml" ) )
            LTSE::Logging::Info( "Current working directory set to: '{}'", ConfigurationRoot.string() );

        // Create Saved, Saved/Logs
        if( !fs::exists( ConfigurationRoot / "Saved" / "Logs" ) ) fs::create_directories( ConfigurationRoot / "Saved" / "Logs" );

        // Configure logger to send messages to saved/logs/EditorLogs.txt
        auto lOutputLogFile = ConfigurationRoot / "Saved" / "Logs" / "EditorLogs.txt";
        LTSE::Logging::Info( "Log file will be written to '{}'", lOutputLogFile.string() );
        LTSE::Logging::SetLogOutputFile( ConfigurationRoot / "Saved" / "Logs" / "EditorLogs.txt" );
    }

    math::ivec2 WindowSize     = { 1920, 1080 };
    math::ivec2 WindowPosition = { 100, 100 };

    fs::path ConfigurationFile = ConfigurationRoot / "EditorConfiguration.yaml";
    if( fs::exists( ConfigurationFile ) )
        LoadConfiguration( ConfigurationFile );
    else
        SaveConfiguration( ConfigurationFile );

    mEngineLoop->SetInitialWindowSize( WindowSize );
    mEngineLoop->SetInitialWindowPosition( WindowPosition );
    mEngineLoop->SetImGuiConfigurationFile( ( ConfigurationRoot / "Saved" / "imgui.ini" ).string() );
    mEngineLoop->Init();

    ScriptManager::Initialize();

    while( mEngineLoop->Tick() )
    {
    }

    mEngineLoop->Shutdown();
    ScriptManager::Shutdown();

    return 0;
}
