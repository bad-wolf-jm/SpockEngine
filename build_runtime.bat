echo off
"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 && cmake --build ./Build && ^
cd "./Source/ScriptCore" && build_library.bat && ^
cd "../../Tests/Mono" && build_library.bat