echo off
"%MSBUILD_VCVARSALL_BAT%" x64 && ^
cmake -B ./Build/CoreLibrary -G "Ninja" . -DCMAKE_BUILD_TYPE=Debug -DYAML_BUILD_SHARED_LIBS=OFF -DBUILD_SHARED_LIBS=OFF -DYAML_CPP_BUILD_TESTS=OFF -DYAML_CPP_STATIC_DEFINE=1 
@REM  && ^
@REM  cd "./Source/ScriptCore" && build_solution.bat && ^
@REM  cd "../../Programs/Editor" && build_solution.bat && ^
@REM  cd "../../Tests/Mono" && build_solution.bat
