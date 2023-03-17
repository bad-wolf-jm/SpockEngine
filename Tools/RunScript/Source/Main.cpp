
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

#include "DotNet/DotNetRuntime.h"

using namespace SE::Core;
using namespace SE::Graphics;
using namespace SE::Core::UI;

namespace fs = std::filesystem;

Ref<argparse::ArgumentParser> ParseCommandLine( int argc, char **argv )
{
    auto lProgramArguments = New<argparse::ArgumentParser>( "bin2ktx" );

    // clang-format off
    lProgramArguments->add_argument( "-p", "--project" )
        .help( "Specify input file" )
        .default_value( std::string{ "" } );

    lProgramArguments->add_argument( "-s", "--script" )
        .help( "Specify input file" );

    lProgramArguments->add_argument( "-M", "--mono-runtime-path" )
        .help( "Specify output file" );

    lProgramArguments->add_argument( "-S", "--script-core-binary-path" )
        .help( "Specify output file" );

    lProgramArguments->add_argument( "-M", "--metrino-binary-path" )
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

    auto     lProjectName              = lProgramArguments->get<std::string>( "--project" );
    fs::path lProjectConfigurationPath = lProjectRoot / fmt::format( "{}.yaml", lProjectName );
    if( !fs::exists( lProjectConfigurationPath ) )
    {
        SE::Logging::Info( "Project file '{}' does not exist", lProjectConfigurationPath.string() );

        std::exit( 1 );
    }

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

    // SE::OtdrEditor::BaseOtdrApplication lEditorApplication;
    // lEditorApplication.Init();

    YAML::Node lRootNode = YAML::LoadFile( lProjectConfigurationPath.string() );

    // Load Metrino assemblies
    {
        fs::path    lMetrinoPath         = "D:\\EXFO\\GitLab\\EXFO\\Build";
        YAML::Node &lMetrinoPathOverride = lRootNode["project"]["metrino_path"];
        if( !lMetrinoPathOverride.IsNull() && fs::exists( lMetrinoPathOverride.as<std::string>() ) )
            lMetrinoPath = lMetrinoPathOverride.as<std::string>();
        if( auto lMetrinoPathOverride = lProgramArguments->present<std::string>( "--metrino-binary-path" ) )
            if( fs::exists( lMetrinoPathOverride.value() ) ) lMetrinoPath = lMetrinoPathOverride.value();

        // clang-format off
        const std::vector<std::string> lAssemblies = { "Metrino.Otdr", "Metrino.Otdr.SignalProcessing", 
            "Metrino.Otdr.Simulation", "Metrino.Otdr.Instrument", "Metrino.Otdr.FileConverter", "Metrino.Olm", 
            "Metrino.Olm.SignalProcessing", "Metrino.Olm.Instrument", "Metrino.Mono" };
        // clang-format on

        for( auto const &lAssemblyName : lAssemblies )
        {
            auto lAssemblyDllName = fmt::format( "{}.dll", lAssemblyName );
            DotNetRuntime::AddAppAssemblyPath( lMetrinoPath / lAssemblyName / "Debug" / lAssemblyDllName, "METRINO" );
        }
    }

    // Load the application assembly and the default scenario file
    {
        YAML::Node &lAssemblyPath = lRootNode["project"]["assembly_path"];
        if( !lAssemblyPath.IsNull() && fs::exists( lAssemblyPath.as<std::string>() ) )
            DotNetRuntime::AddAppAssemblyPath( lAssemblyPath.as<std::string>(), "SYSTEM UNDER TEST" );
    }

    DotNetRuntime::ReloadAssemblies();

    auto lScriptBaseClass = DotNetRuntime::GetClassType( "SpockEngine.Script" );

    auto lScriptToRun = lProgramArguments->present<std::string>( "--script" );
    if( lScriptToRun )
    {
        auto lScriptClass         = DotNetRuntime::GetClassType( lScriptToRun.value() );
        auto lScriptClassInstance = lScriptClass.Instantiate();
        lScriptClassInstance->CallMethod( "BeginScenario" );

        while( true )
        {
            Timestep aTs;
            lScriptClassInstance->CallMethod( "Tick", &aTs );
        }

        lScriptClassInstance->CallMethod( "EndScenario" );
    }

    DotNetRuntime::Shutdown();

    return 0;
}
