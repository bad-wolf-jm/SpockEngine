"%MSBUILD_VCVARSALL_BAT%" x64 && cmake --build ./Build/CoreLibrary && ^
@REM cd "./Source/ScriptCore" && build_library.bat && ^
cd "../../Programs/Editor" && build_library.bat 
@REM && ^
@REM cd "../../Tests/Mono" && build_library.bat