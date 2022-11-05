
#ifdef APIENTRY
#    undef APIENTRY
#endif
#include <chrono>
#include <cstdlib>
#include <shlobj.h>

#include <argparse/argparse.hpp>
#include <filesystem>
#include <fstream>
#include <yaml-cpp/yaml.h>

#include <direct.h>
#include <iostream>
#include <limits.h>
#include <string>

#include "Core/GraphicContext//GraphicContext.h"
#include "Core/Logging.h"
#include "Core/Math/Types.h"
#include "Core/Memory.h"
#include "Engine/Engine.h"

#include "Editor/BaseEditorApplication.h"

#include "Mono/Manager.h"

using namespace LTSE::Core;
using namespace LTSE::Graphics;
using namespace LTSE::Core::UI;

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

void SaveConfiguration( fs::path aConfigurationFile, math::ivec2 const &aWindowSize, math::ivec2 const &aWindowPosition )
{
    YAML::Emitter lConfigurationOut;
    lConfigurationOut << YAML::BeginMap;
    {
        lConfigurationOut << YAML::Key << "application" << YAML::Value;
        lConfigurationOut << YAML::BeginMap;
        {
            lConfigurationOut << YAML::Key << "window_properties" << YAML::Value << YAML::Flow;

            lConfigurationOut << YAML::BeginMap;
            lConfigurationOut << YAML::Key << "width" << YAML::Value << aWindowSize.x;
            lConfigurationOut << YAML::Key << "height" << YAML::Value << aWindowSize.y;
            lConfigurationOut << YAML::Key << "x" << YAML::Value << aWindowPosition.x;
            lConfigurationOut << YAML::Key << "y" << YAML::Value << aWindowPosition.y;
            lConfigurationOut << YAML::EndMap;
        }

        lConfigurationOut << YAML::EndMap;
    }
    lConfigurationOut << YAML::EndMap;
    std::ofstream fout( aConfigurationFile );
    fout << lConfigurationOut.c_str();
}

Ref<argparse::ArgumentParser> ParseCommandLine( int argc, char **argv )
{
    auto lProgramArguments = New<argparse::ArgumentParser>( "bin2ktx" );

    // clang-format off

    lProgramArguments->add_argument( "-p", "--project" )
        .help( "Specify input file" )
        .default_value( std::string{ "" } );

    lProgramArguments->add_argument( "-s", "--scenario" )
        .help( "Specify input file" )
        .default_value( std::string{ "" } );

    lProgramArguments->add_argument( "-x", "--pos_x" )
        .help( "Specify output file" )
        .scan<'i', int>();

    lProgramArguments->add_argument( "-y", "--pos_y" )
        .help( "Specify output file" )
        .scan<'i', int>();

    lProgramArguments->add_argument( "-w", "--res_x" )
        .help( "Specify output file" )
        .scan<'i', int>();

    lProgramArguments->add_argument( "-h", "--res_y" )
        .help( "Specify output file" )
        .scan<'i', int>();

    lProgramArguments->add_argument( "-M", "--mono_runtime" )
        .help( "Specify output file" );

    lProgramArguments->add_argument( "-L", "--log_level" )
        .help( "Specify output file" ).scan<'i', int>();

    // clang-format on

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

fs::path GetCwd()
{
    char buff[MAX_PATH];
    _getcwd( buff, MAX_PATH );
    fs::path lCwd = std::string( buff );

    return lCwd;
}

int main( int argc, char **argv )
{
    auto lProgramArguments = ParseCommandLine( argc, argv );

    fs::path lLocalConfigFolder = "";
    fs::path lUserHomeFolder    = "";
    {
        CHAR    aProfilePath[MAX_PATH];
        HRESULT result = SHGetFolderPathA( NULL, CSIDL_PROFILE, NULL, 0, aProfilePath );
        if( SUCCEEDED( result ) )
        {
            lUserHomeFolder = fs::path( aProfilePath );
        }

        CHAR aUserAppData[MAX_PATH];
        result = SHGetFolderPathA( NULL, CSIDL_LOCAL_APPDATA, NULL, 0, aUserAppData );
        if( SUCCEEDED( result ) )
        {
            lLocalConfigFolder = fs::path( aUserAppData );
        }
    }

    // Retrieve the Mono runtime
    fs::path    lMonoPath = "C:\\Program Files\\Mono\\lib\\mono\\4.5";
    const char *lPath     = std::getenv( "MONO_PATH" );
    if( lPath && fs::exists( lPath ) )
    {
        lMonoPath = lPath;
        if( auto lMonoPathOverride = lProgramArguments->present<std::string>( "--mono_runtime" ) )
        {
            if( fs ::exists( lMonoPathOverride.value() ) ) lMonoPath = lMonoPathOverride.value();
        }
    }

    // get cwd
    fs::path lProjectRoot = GetCwd();

    LTSE::Logging::Info( "Current working directory is: '{}'", lProjectRoot.string() );

    // Create Saved, Saved/Logs
    if( !fs::exists( lProjectRoot / "Saved" / "Logs" ) ) fs::create_directories( lProjectRoot / "Saved" / "Logs" );

    // Create Saved, Saved/Config
    if( !fs::exists( lProjectRoot / "Saved" / "Config" ) ) fs::create_directories( lProjectRoot / "Saved" / "Config" );

    // Configure logger to send messages to Saved/Logs/EditorLogs.txt
    auto lOutputLogFile = lProjectRoot / "Saved" / "Logs" / "EditorLogs.txt";
    LTSE::Logging::Info( "Log file will be written to '{}'", lOutputLogFile.string() );
    LTSE::Logging::SetLogOutputFile( lProjectRoot / "Saved" / "Logs" / "EditorLogs.txt" );

    math::ivec2 lWindowSize     = { 640, 480 };
    math::ivec2 lWindowPosition = { 100, 100 };

    fs::path lConfigurationFile = lProjectRoot / "Saved" / "Config" / "EditorConfiguration.yaml";
    if( fs::exists( lConfigurationFile ) )
        LoadConfiguration( lConfigurationFile, lWindowSize, lWindowPosition );
    else
        SaveConfiguration( lConfigurationFile, lWindowSize, lWindowPosition );

    if( auto lResXOverride = lProgramArguments->present<int>( "--res_x" ) ) lWindowSize.x = lResXOverride.value();
    if( auto lResYOverride = lProgramArguments->present<int>( "--res_y" ) ) lWindowSize.y = lResYOverride.value();
    if( auto lPosXOverride = lProgramArguments->present<int>( "--pos_x" ) ) lWindowPosition.x = lPosXOverride.value();
    if( auto lPosYOverride = lProgramArguments->present<int>( "--pos_y" ) ) lWindowPosition.y = lPosYOverride.value();

    auto     lProjectName              = lProgramArguments->get<std::string>( "--project" );
    fs::path lProjectConfigurationPath = lProjectRoot / fmt::format( "{}.yaml", lProjectName );
    if( !fs::exists( lProjectConfigurationPath ) )
    {
        LTSE::Logging::Info( "Project file '{}' does not exist", lProjectConfigurationPath.string() );

        std::exit( 1 );
    }

    auto lScenario = fs::path( lProgramArguments->get<std::string>( "--scenario" ) );
    if( !fs::exists( lScenario ) ) LTSE::Logging::Info( "Scenario file '{}' does not exist", lScenario.string() );

    LTSE::Core::Engine::Initialize( lWindowSize, lWindowPosition, lProjectRoot / "Saved" / "imgui.ini" );

    LTSE::Graphics::OptixDeviceContextObject::Initialize();
    ScriptManager::Initialize();

    LTSE::Editor::BaseEditorApplication lEditorApplication;
    lEditorApplication.Init();

    LTSE::Core::Engine::GetInstance()->UpdateDelegate.connect<&LTSE::Editor::BaseEditorApplication::Update>( lEditorApplication );
    LTSE::Core::Engine::GetInstance()->RenderDelegate.connect<&LTSE::Editor::BaseEditorApplication::RenderScene>( lEditorApplication );
    LTSE::Core::Engine::GetInstance()->UIDelegate.connect<&LTSE::Editor::BaseEditorApplication::RenderUI>( lEditorApplication );

    while( LTSE::Core::Engine::GetInstance()->Tick() )
    {
    }

    ScriptManager::Shutdown();
    LTSE::Core::Engine::Shutdown();

    return 0;
}
