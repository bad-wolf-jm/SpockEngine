"%MSBUILD_VCVARSALL_BAT%" x64 && cmake --build ./Build/CoreLibrary && ^
cd "../../Programs/Editor" && build_library.bat 
