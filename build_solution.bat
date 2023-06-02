echo off
"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat" x64 && ^
cmake -B ./Build -G "Ninja" . -DCMAKE_BUILD_TYPE=Debug -DYAML_BUILD_SHARED_LIBS=OFF -DBUILD_SHARED_LIBS=OFF -DYAML_CPP_BUILD_TESTS=OFF -DYAML_CPP_STATIC_DEFINE=1 && ^
cd "./Source/ScriptCore" && build_solution.bat && ^
cd "../../Tests/Mono" && build_solution.bat