
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

#include "OtdrEditor/BaseOtdrApplication.h"

#include "DotNet/Runtime.h"

using namespace SE::Core;
using namespace SE::Graphics;
using namespace SE::Core::UI;

namespace fs = std::filesystem;

void LoadConfiguration( fs::path aConfigurationFile, math::ivec2 &aWindowSize, math::ivec2 &aWindowPosition,
                        UIConfiguration &aUIConfiguration )
{
    YAML::Node lRootNode = YAML::LoadFile( aConfigurationFile.string() );
    lRootNode            = lRootNode["application"];

    fs::path lFontRoot = "C:\\Windows\\Fonts";

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

        aUIConfiguration.mMainFont       = lFontRoot / lUIProperties["main_font"]["regular"].as<std::string>();
        aUIConfiguration.mBoldFont       = lFontRoot / lUIProperties["main_font"]["bold"].as<std::string>();
        aUIConfiguration.mItalicFont     = lFontRoot / lUIProperties["main_font"]["italic"].as<std::string>();
        aUIConfiguration.mBoldItalicFont = lFontRoot / lUIProperties["main_font"]["bold_italic"].as<std::string>();

        aUIConfiguration.mIconFont = fs::path( lUIProperties["icon_font"].as<std::string>() );
        aUIConfiguration.mMonoFont = fs::path( lUIProperties["mono_font"].as<std::string>() );
    }
}

void SaveConfiguration( fs::path aConfigurationFile, math::ivec2 const &aWindowSize, math::ivec2 const &aWindowPosition,
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
        .default_value( std::string{ "" } );

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

    lProgramArguments->add_argument( "-M", "--mono-runtime-path" )
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
        if( SUCCEEDED( result ) ) lUserHomeFolder = fs::path( aProfilePath );

        CHAR aUserAppData[MAX_PATH];
        result = SHGetFolderPathA( NULL, CSIDL_LOCAL_APPDATA, NULL, 0, aUserAppData );
        if( SUCCEEDED( result ) ) lLocalConfigFolder = fs::path( aUserAppData );
    }

    // Create Saved, Saved/Logs, Saved/Config
    if( !fs::exists( lLocalConfigFolder / "OtdrTool" / "Logs" ) ) fs::create_directories( lLocalConfigFolder / "OtdrTool" / "Logs" );
    if( !fs::exists( lLocalConfigFolder / "OtdrTool" / "Config" ) )
        fs::create_directories( lLocalConfigFolder / "OtdrTool" / "Config" );

    math::ivec2     lWindowSize     = { 640, 480 };
    math::ivec2     lWindowPosition = { 100, 100 };
    UIConfiguration lUIConfiguration{};

    fs::path lConfigurationFile = lLocalConfigFolder / "OtdrTool" / "Config" / "Application.yaml";
    if( fs::exists( lConfigurationFile ) )
        LoadConfiguration( lConfigurationFile, lWindowSize, lWindowPosition, lUIConfiguration );
    else
        SaveConfiguration( lConfigurationFile, lWindowSize, lWindowPosition, lUIConfiguration );

    if( auto lResXOverride = lProgramArguments->present<int>( "--width" ) ) lWindowSize.x = lResXOverride.value();
    if( auto lResYOverride = lProgramArguments->present<int>( "--height" ) ) lWindowSize.y = lResYOverride.value();
    if( auto lPosXOverride = lProgramArguments->present<int>( "--x" ) ) lWindowPosition.x = lPosXOverride.value();
    if( auto lPosYOverride = lProgramArguments->present<int>( "--y" ) ) lWindowPosition.y = lPosYOverride.value();

    SE::Core::Engine::Initialize( lWindowSize, lWindowPosition, lLocalConfigFolder / "OtdrTool" / "Config" / "imgui.ini",
                                  lUIConfiguration );

    // Retrieve the Mono runtime
    fs::path    lMonoPath = "C:\\Program Files\\Mono\\lib\\mono\\4.5";
    const char *lPath     = std::getenv( "MONO_PATH" );
    if( lPath && fs::exists( lPath ) )
    {
        lMonoPath = lPath;
        if( auto lMonoPathOverride = lProgramArguments->present<std::string>( "--mono-runtime-path" ) )
            if( fs ::exists( lMonoPathOverride.value() ) ) lMonoPath = lMonoPathOverride.value();
    }

    // Retrieve the Mono core assembly path
    fs::path lCoreScriptingPath = "c:/GitLab/SpockEngine/Source/ScriptCore/Build/Debug/SE_Core.dll";

    DotNetRuntime::Initialize( lMonoPath, lCoreScriptingPath );

    auto     lApplicationName              = lProgramArguments->get<std::string>( "--application" );
    fs::path lApplicationConfigurationPath = "";
    if( !lApplicationName.empty() )
    {
        lApplicationConfigurationPath = lLocalConfigFolder / "OtdrTool" / "Config" / fmt::format( "{}.yaml", lApplicationName );
        auto lApplicationAssembly =
            fs::path( "D:\\Build\\Lib" ) / "debug" / "develop" / lApplicationName / fmt::format( "{}.dll", lApplicationName );
        if( fs::exists( lApplicationAssembly ) ) DotNetRuntime::AddAppAssemblyPath( lApplicationAssembly.string(), "APPLICATION" );

        if( !fs::exists( lApplicationConfigurationPath ) )
            SE::Logging::Info( "Project file '{}' does not exist", lApplicationConfigurationPath.string() );
    }

    DotNetRuntime::ReloadAssemblies();

    SE::OtdrEditor::BaseOtdrApplication lEditorApplication;

    if( !lApplicationName.empty() )
        lEditorApplication.Init( fmt::format( "{}.{}", lApplicationName, lApplicationName ), lApplicationConfigurationPath );
    else
        lEditorApplication.Init();

    SE::Core::Engine::GetInstance()->UpdateDelegate.connect<&SE::OtdrEditor::BaseOtdrApplication::Update>( lEditorApplication );
    SE::Core::Engine::GetInstance()->RenderDelegate.connect<&SE::OtdrEditor::BaseOtdrApplication::RenderScene>( lEditorApplication );
    SE::Core::Engine::GetInstance()->UIDelegate.connect<&SE::OtdrEditor::BaseOtdrApplication::RenderUI>( lEditorApplication );

    while( SE::Core::Engine::GetInstance()->Tick() )
    {
    }

    lEditorApplication.Shutdown( lApplicationConfigurationPath );
    SaveConfiguration( lConfigurationFile, lWindowSize, lWindowPosition, lUIConfiguration );

    DotNetRuntime::Shutdown();
    SE::Core::Engine::Shutdown();

    return 0;
}
