#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <filesystem>

#include "TestUtils.h"

#include "Mono/MonoScriptEngine.h"

using namespace SE::Core;
using namespace TestUtils;

namespace fs = std::filesystem;

void InitializeMonoscripting()
{
    fs::path lMonoPath          = "C:\\Program Files\\Mono\\lib\\mono\\4.5";
    fs::path lCoreScriptingPath = "c:\\GitLab\\SpockEngine\\Source\\ScriptCore\\Build\\Debug\\SE_Core.dll";
    MonoScriptEngine::Initialize( lMonoPath, lCoreScriptingPath );
}

TEST_CASE( "Initialize scripting engine", "[MONO_SCRIPTING]" )
{
    InitializeMonoscripting();

    REQUIRE( true );
}

TEST_CASE( "Set app assembly path", "[MONO_SCRIPTING]" )
{
    InitializeMonoscripting();
    fs::path lAppAssemblyPath = "C:\\GitLab\\SpockEngine\\Tests\\Mono\\Build\\Debug\\MonoscriptingTest.dll";
    MonoScriptEngine::SetAppAssemblyPath( lAppAssemblyPath );
    REQUIRE( true );
}