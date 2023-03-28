
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

void SaveConfiguration( fs::path aConfigurationFile, fs::path aMetrinoPath, math::ivec2 const &aWindowSize,
                        math::ivec2 const &aWindowPosition, UIConfiguration &aUIConfiguration )
{
    YAML::Emitter lConfigurationOut;
    lConfigurationOut << YAML::BeginMap;
    {
        lConfigurationOut << YAML::Key << "application" << YAML::Value;
        lConfigurationOut << YAML::BeginMap;
        {
            if( aMetrinoPath.string().empty() )
                lConfigurationOut << YAML::Key << "metrino_path" << YAML::Value << YAML::Null;
            else
                lConfigurationOut << YAML::Key << "metrino_path" << YAML::Value << aMetrinoPath.string();

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

    lProgramArguments->add_argument( "-p", "--project" )
        .help( "Specify input file" )
        .default_value( std::string{ "" } );

    lProgramArguments->add_argument( "-a", "--application" )
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

    lProgramArguments->add_argument( "-M", "--mono-runtime-path" )
        .help( "Specify output file" );

    lProgramArguments->add_argument( "-S", "--script-core-binary-path" )
        .help( "Specify output file" );

    lProgramArguments->add_argument( "-S", "--metrino-binary-path" )
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

    // get cwd
    fs::path lProjectRoot = GetCwd();

    SE::Logging::Info( "Current working directory is: '{}'", lProjectRoot.string() );

    // Create Saved, Saved/Logs, Saved/Config
    if( !fs::exists( lProjectRoot / "Saved" / "Logs" ) ) fs::create_directories( lProjectRoot / "Saved" / "Logs" );
    if( !fs::exists( lProjectRoot / "Saved" / "Config" ) ) fs::create_directories( lProjectRoot / "Saved" / "Config" );

    // Create Saved, Saved/Logs, Saved/Config
    if( !fs::exists( lLocalConfigFolder / "OtdrTool" / "Logs" ) ) fs::create_directories( lLocalConfigFolder / "OtdrTool" / "Logs" );
    if( !fs::exists( lLocalConfigFolder / "OtdrTool" / "Config" ) )
        fs::create_directories( lLocalConfigFolder / "OtdrTool" / "Config" );

    // Configure logger to send messages to Saved/Logs/EditorLogs.txt
    auto lOutputLogFile = lProjectRoot / "Saved" / "Logs" / "EditorLogs.txt";
    SE::Logging::Info( "Log file will be written to '{}'", lOutputLogFile.string() );
    SE::Logging::SetLogOutputFile( lProjectRoot / "Saved" / "Logs" / "EditorLogs.txt" );

    math::ivec2     lWindowSize     = { 640, 480 };
    math::ivec2     lWindowPosition = { 100, 100 };
    UIConfiguration lUIConfiguration{};

    fs::path lConfigurationFile = lLocalConfigFolder / "OtdrTool" / "Config" / "Application.yaml";
    if( fs::exists( lConfigurationFile ) )
        LoadConfiguration( lConfigurationFile, lWindowSize, lWindowPosition, lUIConfiguration );
    else
        SaveConfiguration( lConfigurationFile, "", lWindowSize, lWindowPosition, lUIConfiguration );

    if( auto lResXOverride = lProgramArguments->present<int>( "--res_x" ) ) lWindowSize.x = lResXOverride.value();
    if( auto lResYOverride = lProgramArguments->present<int>( "--res_y" ) ) lWindowSize.y = lResYOverride.value();
    if( auto lPosXOverride = lProgramArguments->present<int>( "--pos_x" ) ) lWindowPosition.x = lPosXOverride.value();
    if( auto lPosYOverride = lProgramArguments->present<int>( "--pos_y" ) ) lWindowPosition.y = lPosYOverride.value();

    SE::Core::Engine::Initialize( lWindowSize, lWindowPosition, lLocalConfigFolder / "OtdrTool" / "Config" / "imgui.ini",
                                  lUIConfiguration );

    SE::Graphics::OptixDeviceContextObject::Initialize();

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
    if( auto lCoreScriptingPathOverride = lProgramArguments->present<std::string>( "--script-core-binary-path" ) )
        if( fs ::exists( lCoreScriptingPathOverride.value() ) ) lCoreScriptingPath = lCoreScriptingPathOverride.value();

    DotNetRuntime::Initialize( lMonoPath, lCoreScriptingPath );

    // Load Metrino assembblies
    YAML::Node lRootNode = YAML::LoadFile( lConfigurationFile.string() );
    fs::path   lMetrinoPath;
    {
        lMetrinoPath                     = "D:\\Build\\Lib";
        YAML::Node &lMetrinoPathOverride = lRootNode["application"]["metrino_path"];
        if( !lMetrinoPathOverride.IsNull() && fs::exists( lMetrinoPathOverride.as<std::string>() ) )
            lMetrinoPath = lMetrinoPathOverride.as<std::string>();
        if( auto lMetrinoPathOverride = lProgramArguments->present<std::string>( "--metrino-binary-path" ) )
            if( fs ::exists( lMetrinoPathOverride.value() ) ) lMetrinoPath = lMetrinoPathOverride.value();

        // clang-format off
        const std::vector<std::string> lAssemblies = { 
            "Metrino.Otdr", 
            "Metrino.Otdr.SignalProcessing", 
            "Metrino.Otdr.Simulation", 
            "Metrino.Otdr.Instrument", 
            "Metrino.Otdr.FileConverter", 
            "Metrino.Olm", 
            "Metrino.Olm.SignalProcessing", 
            "Metrino.Olm.Instrument", 
            "Metrino.Interop" };
        // clang-format on

        for( auto const &lAssemblyName : lAssemblies )
        {
            auto lAssemblyDllName = fmt::format( "{}.dll", lAssemblyName );
            DotNetRuntime::AddAppAssemblyPath( lMetrinoPath / "debug" / "Metrino.Interop" / lAssemblyDllName, "METRINO" );
        }
    }

    auto     lProjectName              = lProgramArguments->get<std::string>( "--project" );
    fs::path lProjectConfigurationPath = lProjectRoot / fmt::format( "{}.yaml", lProjectName );
    if( !fs::exists( lProjectConfigurationPath ) )
    {
        SE::Logging::Info( "Project file '{}' does not exist", lProjectConfigurationPath.string() );
    }
    else
    {
        // Load the application assembly and the default scenario file
        YAML::Node lProjectRootNode = YAML::LoadFile( lProjectConfigurationPath.string() );
        {
            YAML::Node &lAssemblyPath = lProjectRootNode["project"]["assembly_path"];
            if( !lAssemblyPath.IsNull() && fs::exists( lAssemblyPath.as<std::string>() ) )
                DotNetRuntime::AddAppAssemblyPath( lAssemblyPath.as<std::string>(), "SYSTEM UNDER TEST" );
        }
    }

    DotNetRuntime::ReloadAssemblies();

    auto &lInitializationClass = DotNetRuntime::GetClassType( "SpockEngine.IO" );
    lInitializationClass.CallMethod( "Initialize" );

    SE::OtdrEditor::BaseOtdrApplication lEditorApplication;
    lEditorApplication.Init();

    SE::Core::Engine::GetInstance()->UpdateDelegate.connect<&SE::OtdrEditor::BaseOtdrApplication::Update>( lEditorApplication );
    SE::Core::Engine::GetInstance()->RenderDelegate.connect<&SE::OtdrEditor::BaseOtdrApplication::RenderScene>( lEditorApplication );
    SE::Core::Engine::GetInstance()->UIDelegate.connect<&SE::OtdrEditor::BaseOtdrApplication::RenderUI>( lEditorApplication );

    while( SE::Core::Engine::GetInstance()->Tick() )
    {
    }

    lEditorApplication.Shutdown();
    SaveConfiguration( lConfigurationFile, lMetrinoPath, lWindowSize, lWindowPosition, lUIConfiguration );

    DotNetRuntime::Shutdown();
    SE::Core::Engine::Shutdown();

    return 0;
}
