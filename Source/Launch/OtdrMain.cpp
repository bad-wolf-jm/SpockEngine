
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

#include "Core/Logging.h"
#include "Core/Math/Types.h"
#include "Core/Memory.h"
#include "Engine/Engine.h"
#include "Graphics/Vulkan/VkGraphicContext.h"

#include "Delegates.h"

#include "CoreCLRHost.h"

using namespace SE::Core;
using namespace SE::Graphics;
using namespace SE::Core::UI;

namespace fs = std::filesystem;

void LoadConfiguration( path_t aConfigurationFile, math::ivec2 &aWindowSize, math::ivec2 &aWindowPosition,
                        UIConfiguration &aUIConfiguration )
{
    YAML::Node lRootNode = YAML::LoadFile( aConfigurationFile.string() );
    lRootNode            = lRootNode["application"];

    path_t lFontRoot = "C:\\Windows\\Fonts";

    YAML::Node lWindowProperties = lRootNode["window_properties"];
    if( !lWindowProperties.IsNull() )
    {
        aWindowSize     = math::ivec2{ lWindowProperties["width"].as<int>(), lWindowProperties["height"].as<int>() };
        aWindowPosition = math::ivec2{ lWindowProperties["x"].as<int>(), lWindowProperties["y"].as<int>() };
    }

    YAML::Node lUIProperties = lRootNode["ui"];
    if( !lUIProperties.IsNull() )
    {
        aUIConfiguration.mFontSize = lUIProperties["font_size"].as<int>();

        aUIConfiguration.mMainFont       = lFontRoot / lUIProperties["main_font"]["regular"].as<string_t>();
        aUIConfiguration.mBoldFont       = lFontRoot / lUIProperties["main_font"]["bold"].as<string_t>();
        aUIConfiguration.mItalicFont     = lFontRoot / lUIProperties["main_font"]["italic"].as<string_t>();
        aUIConfiguration.mBoldItalicFont = lFontRoot / lUIProperties["main_font"]["bold_italic"].as<string_t>();

        aUIConfiguration.mIconFont = path_t( lUIProperties["icon_font"].as<string_t>() );
        aUIConfiguration.mMonoFont = path_t( lUIProperties["mono_font"].as<string_t>() );
    }
}

void SaveConfiguration( path_t aConfigurationFile, math::ivec2 const &aWindowSize, math::ivec2 const &aWindowPosition,
                        UIConfiguration &aUIConfiguration )
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

            lConfigurationOut << YAML::Key << "ui" << YAML::Value;
            lConfigurationOut << YAML::BeginMap;
            lConfigurationOut << YAML::Key << "font_size" << YAML::Value << aUIConfiguration.mFontSize;
            lConfigurationOut << YAML::Key << "icon_font" << YAML::Value << aUIConfiguration.mIconFont.string();
            lConfigurationOut << YAML::Key << "main_font" << YAML::Value;
            lConfigurationOut << YAML::BeginMap;
            lConfigurationOut << YAML::Key << "regular" << YAML::Value << aUIConfiguration.mMainFont.string();
            lConfigurationOut << YAML::Key << "bold" << YAML::Value << aUIConfiguration.mBoldFont.string();
            lConfigurationOut << YAML::Key << "italic" << YAML::Value << aUIConfiguration.mItalicFont.string();
            lConfigurationOut << YAML::Key << "bold_italic" << YAML::Value << aUIConfiguration.mBoldItalicFont.string();
            lConfigurationOut << YAML::EndMap;

            lConfigurationOut << YAML::Key << "mono_font" << YAML::Value << aUIConfiguration.mMonoFont.string();
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

    lProgramArguments->add_argument( "-a", "--application" )
        .help( "Specify input file" )
        .default_value( string_t{ "" } );

    lProgramArguments->add_argument( "-x", "--x" )
        .help( "Specify output file" )
        .scan<'i', int>();

    lProgramArguments->add_argument( "-y", "--y" )
        .help( "Specify output file" )
        .scan<'i', int>();

    lProgramArguments->add_argument( "-w", "--width" )
        .help( "Specify output file" )
        .scan<'i', int>();

    lProgramArguments->add_argument( "-h", "--height" )
        .help( "Specify output file" )
        .scan<'i', int>();

    lProgramArguments->add_argument( "-M", "--coreclr-path" )
        .help( "Specify output file" );

    lProgramArguments->add_argument( "-L", "--log-level" )
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

path_t GetCwd()
{
    char buff[MAX_PATH];
    _getcwd( buff, MAX_PATH );
    path_t lCwd = string_t( buff );

    return lCwd;
}

int main( int argc, char **argv )
{
    auto lProgramArguments = ParseCommandLine( argc, argv );

    // Retrieve the current user's home folder and local configuration folder. This is where we put
    // all configuration data flr all instances of the application
    path_t   lLocalConfigFolder = "";
    path_t   lUserHomeFolder    = "";
    string_t lExePath           = "";
    {
        CHAR    aProfilePath[MAX_PATH];
        HRESULT result = SHGetFolderPathA( NULL, CSIDL_PROFILE, NULL, 0, aProfilePath );
        if( SUCCEEDED( result ) ) lUserHomeFolder = path_t( aProfilePath );

        CHAR aUserAppData[MAX_PATH];
        result = SHGetFolderPathA( NULL, CSIDL_LOCAL_APPDATA, NULL, 0, aUserAppData );
        if( SUCCEEDED( result ) ) lLocalConfigFolder = path_t( aUserAppData );

        CHAR  aExePath[MAX_PATH];
        DWORD count = GetModuleFileNameA( nullptr, aExePath, ( sizeof( aExePath ) / sizeof( aExePath[0] ) ) );
        if( SUCCEEDED( ::GetLastError() ) ) lExePath = string_t( aExePath );
    }

    // Create Saved, Saved/Logs, Saved/Config
    if( !fs::exists( lLocalConfigFolder / "OtdrTool" / "Logs" ) ) fs::create_directories( lLocalConfigFolder / "OtdrTool" / "Logs" );
    if( !fs::exists( lLocalConfigFolder / "OtdrTool" / "Config" ) )
        fs::create_directories( lLocalConfigFolder / "OtdrTool" / "Config" );

    math::ivec2     lWindowSize     = { 640, 480 };
    math::ivec2     lWindowPosition = { 0, 0 };
    UIConfiguration lUIConfiguration{};

    path_t lConfigurationFile = lLocalConfigFolder / "OtdrTool" / "Config" / "Application.yaml";
    if( fs::exists( lConfigurationFile ) )
        LoadConfiguration( lConfigurationFile, lWindowSize, lWindowPosition, lUIConfiguration );
    else
        SaveConfiguration( lConfigurationFile, lWindowSize, lWindowPosition, lUIConfiguration );

    // Override the main window's position and size of provided as command line argumetns
    if( auto lResXOverride = lProgramArguments->present<int>( "--width" ) ) lWindowSize.x = lResXOverride.value();
    if( auto lResYOverride = lProgramArguments->present<int>( "--height" ) ) lWindowSize.y = lResYOverride.value();
    if( auto lPosXOverride = lProgramArguments->present<int>( "--x" ) ) lWindowPosition.x = lPosXOverride.value();
    if( auto lPosYOverride = lProgramArguments->present<int>( "--y" ) ) lWindowPosition.y = lPosYOverride.value();

    // Initialize the engine and load the UI's initialzation file
    SE::Core::Engine::Initialize( lWindowSize, lWindowPosition, lLocalConfigFolder / "OtdrTool" / "Config" / "imgui.ini",
                                  lUIConfiguration );

    CoreCLRHost lCoreCLR;

    // Load the managed part of the application, whose name is given at the command line
    auto   lApplicationName       = lProgramArguments->get<string_t>( "--application" );
    path_t lApplicationConfigPath = "";
    if( !lApplicationName.empty() )
    {
        lApplicationConfigPath = lLocalConfigFolder / "OtdrTool" / "Config" / fmt::format( "{}.yaml", lApplicationName );
        auto lApplicationAssembly =
            path_t( "D:\\Build\\Lib" ) / "debug" / "develop" / "net7" / lApplicationName / fmt::format( "{}.dll", lApplicationName );

        if( !fs::exists( lApplicationConfigPath ) )
            SE::Logging::Info( "Project file '{}' does not exist", lApplicationConfigPath.string() );

        lCoreCLR.LoadApplicationAssembly( lApplicationAssembly.string(), lApplicationName );
    }

    SE::OtdrEditor::Application lEditorApplication( lCoreCLR );

    if( !lApplicationName.empty() )
        lEditorApplication.Init( lApplicationConfigPath );
    else
        lEditorApplication.Init();

    // Hook into the engine's callbacks
    SE::Core::Engine::GetInstance()->UpdateDelegate.connect<&SE::OtdrEditor::Application::Update>( lEditorApplication );
    SE::Core::Engine::GetInstance()->RenderDelegate.connect<&SE::OtdrEditor::Application::RenderScene>( lEditorApplication );
    SE::Core::Engine::GetInstance()->UIDelegate.connect<&SE::OtdrEditor::Application::RenderUI>( lEditorApplication );

    while( SE::Core::Engine::GetInstance()->Tick() )
    {
    }

    lEditorApplication.Shutdown( lApplicationConfigPath );
    SaveConfiguration( lConfigurationFile, lWindowSize, lWindowPosition, lUIConfiguration );

    lCoreCLR.Shutdown();
    SE::Core::Engine::Shutdown();

    return 0;
}
