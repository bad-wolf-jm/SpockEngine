"%MSBUILD_VCVARSALL_BAT%" x64 && ^
cmake -B ./Build/CoreLibrary -G "Ninja" . -DCMAKE_BUILD_TYPE=Debug -DYAML_BUILD_SHARED_LIBS=OFF -DBUILD_SHARED_LIBS=OFF -DYAML_CPP_BUILD_TESTS=OFF -DYAML_CPP_STATIC_DEFINE=1 && ^
cd "./Source/ScriptCore" && build_solution.bat && ^
cd "../../Programs/Bootstrap" && build_solution.bat