
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

#include "Editor/BaseEditorApplication.h"

#include "Mono/MonoRuntime.h"
// #include "Mono/MonoScriptEngine.h"

using namespace SE::Core;
using namespace SE::Graphics;
using namespace SE::Core::UI;

namespace fs = std::filesystem;

fs::path GetCwd()
{
    char buff[MAX_PATH];
    _getcwd( buff, MAX_PATH );
    fs::path lCwd = std::string( buff );

    return lCwd;
}

int main( int argc, char **argv )
{
    fs::path lMonoPath = "C:\\Program Files\\Mono\\lib\\mono\\4.5";
    fs::path lCoreScriptingPath = "c:/GitLab/SpockEngine/Source/ScriptCore/Build/Debug/SE_Core.dll";

    MonoRuntime::Initialize( lMonoPath, lCoreScriptingPath );

    const fs::path METRINO_PATH = "D:\\EXFO\\GitLab\\EXFO\\Build";

    // MonoRuntime::AddAppAssemblyPath( METRINO_PATH / "Metrino.Kernos.dll" );
    // MonoRuntime::AddAppAssemblyPath( METRINO_PATH / "Metrino.Kernos.Platform.dll" );
    // MonoRuntime::AddAppAssemblyPath( METRINO_PATH / "Metrino.Kernos.Instrument.dll" );
    // MonoRuntime::AddAppAssemblyPath( METRINO_PATH / "Metrino.Otdr.dll" );
    // MonoRuntime::AddAppAssemblyPath( METRINO_PATH / "Metrino.Otdr.SignalProcessing.dll" );
    // MonoRuntime::AddAppAssemblyPath( METRINO_PATH / "Metrino.Otdr.Simulation.dll" );
    // MonoRuntime::AddAppAssemblyPath( METRINO_PATH / "Metrino.Otdr" / "Debug" / "fr" / "Metrino.Otdr.resources.dll" );
    MonoRuntime::AddAppAssemblyPath( METRINO_PATH / "Metrino.Otdr" / "Debug" / "Metrino.Otdr.dll", "" );
    MonoRuntime::AddAppAssemblyPath( METRINO_PATH / "Metrino.Otdr.Instrument" / "Debug" / "Metrino.Otdr.Instrument.dll", "" );
    // MonoRuntime::AddAppAssemblyPath( METRINO_PATH / "Metrino.Otdr.FileConverter.dll" );
    // MonoRuntime::AddAppAssemblyPath( METRINO_PATH / "Metrino.Olm.dlqQQQQQl" );
    // MonoRuntime::AddAppAssemblyPath( METRINO_PATH / "Metrino.Olm.SignalProcessing.dll" );
    // MonoRuntime::AddAppAssemblyPath( METRINO_PATH / "Metrino.Olm.Instrument.dll" );
    MonoRuntime::ReloadAssemblies();

    auto lDetector = MonoRuntime::GetClassType( "Metrino.Otdr.Detection.ModuleIdentifier" );
    auto lConnectedModules = lDetector.CallMethod("GetOtdrModules");

    // auto lKernosInstrumentContext = MonoRuntime::GetClassType( "Metrino.Kernos.Instrument.Context" ).Instantiate();
    // MonoString *lAddress = MonoRuntime::NewString( "simulator:" );
    // auto  lInstrumentClass = MonoRuntime::GetClassType( "Metrino.Otdr.Instrument.Instrument7000" )
    //                             .Instantiate( lAddress, lKernosInstrumentContext.GetInstance() );
    // double lTimeout = 10.0;
    // lInstrumentClass.CallMethod( "WaitForReady", &lTimeout );
    // MonoRuntime::Shutdown();

    return 0;
}
