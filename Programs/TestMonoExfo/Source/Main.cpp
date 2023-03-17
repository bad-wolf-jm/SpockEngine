
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

#include "DotNet/Runtime.h"
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

    DotNetRuntime::Initialize( lMonoPath, lCoreScriptingPath );

    const fs::path METRINO_PATH = "D:\\EXFO\\GitLab\\EXFO\\Build";

    // DotNetRuntime::AddAppAssemblyPath( METRINO_PATH / "Metrino.Kernos.dll" );
    // DotNetRuntime::AddAppAssemblyPath( METRINO_PATH / "Metrino.Kernos.Platform.dll" );
    // DotNetRuntime::AddAppAssemblyPath( METRINO_PATH / "Metrino.Kernos.Instrument.dll" );
    // DotNetRuntime::AddAppAssemblyPath( METRINO_PATH / "Metrino.Otdr.dll" );
    // DotNetRuntime::AddAppAssemblyPath( METRINO_PATH / "Metrino.Otdr.SignalProcessing.dll" );
    // DotNetRuntime::AddAppAssemblyPath( METRINO_PATH / "Metrino.Otdr.Simulation.dll" );
    // DotNetRuntime::AddAppAssemblyPath( METRINO_PATH / "Metrino.Otdr" / "Debug" / "fr" / "Metrino.Otdr.resources.dll" );
    DotNetRuntime::AddAppAssemblyPath( METRINO_PATH / "Metrino.Otdr" / "Debug" / "Metrino.Otdr.dll", "" );
    DotNetRuntime::AddAppAssemblyPath( METRINO_PATH / "Metrino.Mono" / "Debug" / "Metrino.Interop.dll", "" );
    // DotNetRuntime::AddAppAssemblyPath( METRINO_PATH / "Metrino.Otdr.FileConverter.dll" );
    // DotNetRuntime::AddAppAssemblyPath( METRINO_PATH / "Metrino.Olm.dlqQQQQQl" );
    // DotNetRuntime::AddAppAssemblyPath( METRINO_PATH / "Metrino.Olm.SignalProcessing.dll" );
    // DotNetRuntime::AddAppAssemblyPath( METRINO_PATH / "Metrino.Olm.Instrument.dll" );
    DotNetRuntime::ReloadAssemblies();

    auto lDetector = DotNetRuntime::GetClassType( "Metrino.Interop.Instruments" );
    lDetector.CallMethod("PrintConnectedModules");

    // auto lKernosInstrumentContext = DotNetRuntime::GetClassType( "Metrino.Kernos.Instrument.Context" ).Instantiate();
    // MonoString *lAddress = DotNetRuntime::NewString( "simulator:" );
    // auto  lInstrumentClass = DotNetRuntime::GetClassType( "Metrino.Otdr.Instrument.Instrument7000" )
    //                             .Instantiate( lAddress, lKernosInstrumentContext.GetInstance() );
    // double lTimeout = 10.0;
    // lInstrumentClass.CallMethod( "WaitForReady", &lTimeout );
    // DotNetRuntime::Shutdown();

    return 0;
}
