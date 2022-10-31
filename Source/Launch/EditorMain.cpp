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

void LoadConfiguration( fs::path aConfigurationFile, math::ivec2 &aWindowSize, math::ivec2 &aWindowPosition )
{
    YAML::Node lRootNode = YAML::LoadFile( aConfigurationFile.string() );

    YAML::Node &lWindowProperties = lRootNode["application"]["window_properties"];
    if( !lWindowProperties.IsNull() )
    {
        aWindowSize     = math::ivec2{ lWindowProperties["width"].as<int>(), lWindowProperties["height"].as<int>() };
        aWindowPosition = math::ivec2{ lWindowProperties["x"].as<int>(), lWindowProperties["y"].as<int>() };
    }
}

void SaveConfiguration( fs::path ConfigurationFile, math::ivec2 const &aWindowSize, math::ivec2 const &aWindowPosition )
{
    YAML::Emitter lConfigurationOut;
    lConfigurationOut << YAML::BeginMap;
    {
        lConfigurationOut << YAML::Key << "application" << YAML::Value;
        lConfigurationOut << YAML::BeginMap;
        {
            lConfigurationOut << YAML::Key << "window_properties" << YAML::Value << YAML::Flow;
            {
                lConfigurationOut << YAML::BeginMap;
                lConfigurationOut << YAML::Key << "width" << YAML::Value << aWindowSize.x;
                lConfigurationOut << YAML::Key << "height" << YAML::Value << aWindowSize.y;
                lConfigurationOut << YAML::Key << "x" << YAML::Value << aWindowPosition.x;
                lConfigurationOut << YAML::Key << "y" << YAML::Value << aWindowPosition.y;
                lConfigurationOut << YAML::EndMap;
            }
        }
        lConfigurationOut << YAML::EndMap;
    }
    lConfigurationOut << YAML::EndMap;
    std::ofstream fout( ConfigurationFile );
    fout << lConfigurationOut.c_str();
}

Ref<argparse::ArgumentParser> ParseCommandLine( int argc, char **argv )
{
    auto lProgramArguments = New<argparse::ArgumentParser>( "bin2ktx" );

    lProgramArguments->add_argument( "-p", "--project" ).help( "Specify input file" );
    lProgramArguments->add_argument( "-s", "--scenario" ).help( "Specify input file" );
    lProgramArguments->add_argument( "-w", "--res_x" ).help( "Specify output file" );
    lProgramArguments->add_argument( "-h", "--res_y" ).help( "Specify output file" );
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
        LoadConfiguration( ConfigurationFile, WindowSize, WindowPosition );
    else
        SaveConfiguration( ConfigurationFile, WindowSize, WindowPosition );

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
