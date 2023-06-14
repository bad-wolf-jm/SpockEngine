"%MSBUILD_VCVARSALL_BAT%" x64 && cmake --build ./Build && ^
cd "./Source/ScriptCore" && build_library.bat && ^
cd "../../Tests/Mono" && build_library.bat