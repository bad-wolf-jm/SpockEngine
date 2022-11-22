
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

using namespace SE::Core;
using namespace SE::Graphics;
using namespace SE::Core::UI;

namespace fs = std::filesystem;

// Ref<argparse::ArgumentParser> ParseCommandLine( int argc, char **argv )
// {
//     auto lProgramArguments = New<argparse::ArgumentParser>( "bin2ktx" );

//     // clang-format off

//     lProgramArguments->add_argument( "-p", "--project" )
//         .help( "Specify input file" )
//         .default_value( std::string{ "" } );

//     lProgramArguments->add_argument( "-s", "--scenario" )
//         .help( "Specify input file" )
//         .default_value( std::string{ "" } );

//     lProgramArguments->add_argument( "-x", "--pos_x" )
//         .help( "Specify output file" )
//         .scan<'i', int>();

//     lProgramArguments->add_argument( "-y", "--pos_y" )
//         .help( "Specify output file" )
//         .scan<'i', int>();

//     lProgramArguments->add_argument( "-w", "--res_x" )
//         .help( "Specify output file" )
//         .scan<'i', int>();

//     lProgramArguments->add_argument( "-h", "--res_y" )
//         .help( "Specify output file" )
//         .scan<'i', int>();

//     lProgramArguments->add_argument( "-M", "--mono_runtime" )
//         .help( "Specify output file" );

//     lProgramArguments->add_argument( "-S", "--script_core" )
//         .help( "Specify output file" );

//     lProgramArguments->add_argument( "-L", "--log_level" )
//         .help( "Specify output file" ).scan<'i', int>();

//     // clang-format on

//     try
//     {
//         lProgramArguments->parse_args( argc, argv );

//         return lProgramArguments;
//     }
//     catch( const std::runtime_error &err )
//     {
//         std::cerr << err.what() << std::endl;
//         std::cerr << lProgramArguments;

//         return nullptr;
//     }
// }

fs::path GetCwd()
{
    char buff[MAX_PATH];
    _getcwd( buff, MAX_PATH );
    fs::path lCwd = std::string( buff );

    return lCwd;
}

int main( int argc, char **argv )
{
    // auto lProgramArguments = ParseCommandLine( argc, argv );

    // fs::path lLocalConfigFolder = "";
    // fs::path lUserHomeFolder    = "";
    // {
    //     CHAR    aProfilePath[MAX_PATH];
    //     HRESULT result = SHGetFolderPathA( NULL, CSIDL_PROFILE, NULL, 0, aProfilePath );
    //     if( SUCCEEDED( result ) )
    //     {
    //         lUserHomeFolder = fs::path( aProfilePath );
    //     }

    //     CHAR aUserAppData[MAX_PATH];
    //     result = SHGetFolderPathA( NULL, CSIDL_LOCAL_APPDATA, NULL, 0, aUserAppData );
    //     if( SUCCEEDED( result ) )
    //     {
    //         lLocalConfigFolder = fs::path( aUserAppData );
    //     }
    // }

    // get cwd
    fs::path lProjectRoot = GetCwd();

    SE::Logging::Info( "Current working directory is: '{}'", lProjectRoot.string() );

    // // Create Saved, Saved/Logs
    // if( !fs::exists( lProjectRoot / "Saved" / "Logs" ) ) fs::create_directories( lProjectRoot / "Saved" / "Logs" );

    // // Create Saved, Saved/Config
    // if( !fs::exists( lProjectRoot / "Saved" / "Config" ) ) fs::create_directories( lProjectRoot / "Saved" / "Config" );

    // Configure logger to send messages to Saved/Logs/EditorLogs.txt
    // auto lOutputLogFile = lProjectRoot / "Saved" / "Logs" / "EditorLogs.txt";
    // SE::Logging::Info( "Log file will be written to '{}'", lOutputLogFile.string() );
    // SE::Logging::SetLogOutputFile( lProjectRoot / "Saved" / "Logs" / "EditorLogs.txt" );

    math::ivec2     lWindowSize     = { 640, 480 };
    math::ivec2     lWindowPosition = { 100, 100 };
    UIConfiguration lUIConfiguration{};
    fs::path lFontRoot = "C:\\Windows\\Fonts";
    lUIConfiguration.mFontSize       = 15;
    lUIConfiguration.mMainFont       = lFontRoot / "segoeui.ttf";
    lUIConfiguration.mBoldFont       = lFontRoot / "segoeuib.ttf";
    lUIConfiguration.mItalicFont     = lFontRoot / "segoeuii.ttf";
    lUIConfiguration.mBoldItalicFont = lFontRoot / "segoeui.ttf";
    lUIConfiguration.mMonoFont       = "C:\\GitLab\\SpockEngine\\Resources\\fonts\\dejavu-fonts-ttf-2.37\\ttf\\DejaVuSansMono.ttf";
    lUIConfiguration.mIconFont       = "C:\\GitLab\\SpockEngine\\Resources\\fonts\\fontawesome-webfont.ttf";

    // ui:
    //     font_size: 14
    //     icon_font: "C:\\GitLab\\SpockEngine\\Resources\\fonts\\fontawesome-webfont.ttf"
    //     main_font:
    //       regular: segoeui.ttf
    //       bold: segoeuib.ttf
    //       italic: segoeuii.ttf
    //       bold_italic: segoeui.ttf
    //     mono_font: "C:\\GitLab\\SpockEngine\\Resources\\fonts\\dejavu-fonts-ttf-2.37\\ttf\\DejaVuSansMono.ttf"

    // fs::path lConfigurationFile = lProjectRoot / "Saved" / "Config" / "EditorConfiguration.yaml";
    // if( fs::exists( lConfigurationFile ) )
    //     LoadConfiguration( lConfigurationFile, lWindowSize, lWindowPosition, lUIConfiguration );
    // else
    //     SaveConfiguration( lConfigurationFile, lWindowSize, lWindowPosition, lUIConfiguration );

    // if( auto lResXOverride = lProgramArguments->present<int>( "--res_x" ) ) lWindowSize.x = lResXOverride.value();
    // if( auto lResYOverride = lProgramArguments->present<int>( "--res_y" ) ) lWindowSize.y = lResYOverride.value();
    // if( auto lPosXOverride = lProgramArguments->present<int>( "--pos_x" ) ) lWindowPosition.x = lPosXOverride.value();
    // if( auto lPosYOverride = lProgramArguments->present<int>( "--pos_y" ) ) lWindowPosition.y = lPosYOverride.value();

    // auto     lProjectName              = lProgramArguments->get<std::string>( "--project" );
    // fs::path lProjectConfigurationPath = lProjectRoot / fmt::format( "{}.yaml", lProjectName );
    // if( !fs::exists( lProjectConfigurationPath ) )
    // {
    //     SE::Logging::Info( "Project file '{}' does not exist", lProjectConfigurationPath.string() );

    //     std::exit( 1 );
    // }

    SE::Core::Engine::Initialize( lWindowSize, lWindowPosition, lProjectRoot / "imgui.ini", lUIConfiguration );

    // auto lScenario = fs::path( lProgramArguments->get<std::string>( "--scenario" ) );
    // if( !fs::exists( lScenario ) ) SE::Logging::Info( "Scenario file '{}' does not exist", lScenario.string() );

    SE::Graphics::OptixDeviceContextObject::Initialize();

    // Retrieve the Mono runtime
    // fs::path    lMonoPath = "C:\\Program Files\\Mono\\lib\\mono\\4.5";
    // const char *lPath     = std::getenv( "MONO_PATH" );
    // if( lPath && fs::exists( lPath ) )
    // {
    //     lMonoPath = lPath;
    //     if( auto lMonoPathOverride = lProgramArguments->present<std::string>( "--mono_runtime" ) )
    //         if( fs ::exists( lMonoPathOverride.value() ) ) lMonoPath = lMonoPathOverride.value();
    // }

    // // Retrieve the Mono core assembly path
    // fs::path lCoreScriptingPath = "c:/GitLab/SpockEngine/Source/ScriptCore/Build/Debug/SE_Core.dll";
    // if( auto lCoreScriptingPathOverride = lProgramArguments->present<std::string>( "--script_core" ) )
    //     if( fs ::exists( lCoreScriptingPathOverride.value() ) ) lCoreScriptingPath = lCoreScriptingPathOverride.value();

    // ScriptManager::Initialize( lMonoPath, lCoreScriptingPath );

    SE::Editor::BaseEditorApplication lEditorApplication;
    lEditorApplication.Init();

    // SE::Core::Engine::GetInstance()->UpdateDelegate.connect<&SE::Editor::BaseEditorApplication::Update>( lEditorApplication );
    // SE::Core::Engine::GetInstance()->RenderDelegate.connect<&SE::Editor::BaseEditorApplication::RenderScene>( lEditorApplication );
    // SE::Core::Engine::GetInstance()->UIDelegate.connect<&SE::Editor::BaseEditorApplication::RenderUI>( lEditorApplication );

    // // Load the application assembly and the default scenario file
    // {
    //     YAML::Node lRootNode = YAML::LoadFile( lProjectConfigurationPath.string() );

    //     YAML::Node &lAssemblyPath = lRootNode["project"]["assembly_path"];
    //     if( !lAssemblyPath.IsNull() && fs::exists( lAssemblyPath.as<std::string>() ) )
    //         ScriptManager::SetAppAssemblyPath( lAssemblyPath.as<std::string>() );

    //     YAML::Node &lDefaultScenarioPath = lRootNode["project"]["default_scenario"];
    //     if( !lDefaultScenarioPath.IsNull() && fs::exists( lDefaultScenarioPath.as<std::string>() ) )
    //         lEditorApplication.mEditorWindow.World->LoadScenario( lDefaultScenarioPath.as<std::string>() );
    // }

    while( SE::Core::Engine::GetInstance()->Tick() )
    {
    }

    // SaveConfiguration( lConfigurationFile, lWindowSize, lWindowPosition, lUIConfiguration );

    // ScriptManager::Shutdown();
    SE::Core::Engine::Shutdown();

    return 0;
}
