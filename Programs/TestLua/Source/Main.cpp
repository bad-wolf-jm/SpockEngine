#include "Scripting/ScriptingEngine.h"

// #define SOL_ALL_SAFETIES_ON 1
// #include <sol/sol.hpp>

#include <iostream>

int main( int, char *[] )
{
    LTSE::Core::ScriptingEngine lEngine{};

    lEngine.Execute("print('foobar')");

    auto x = lEngine.LoadFile("C:\\GitLab\\LTSimulationEngine\\Programs\\TestLua\\Test\\Script.lua");

    lEngine.Execute(x, "print(f(100))");

    return 0;
}
