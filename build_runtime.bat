"%MSBUILD_VCVARSALL_BAT%" x64 && cmake --build ./Build/CoreLibrary
@REM  && ^
@REM cd "./Source/ScriptCore" && build_library.bat && ^
@REM cd "../../Programs/Editor" && build_library.bat && ^
@REM cd "../../Tests/Mono" && build_library.bat