echo off
"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 && cmake -B ./Build -G "Ninja" . -DCMAKE_BUILD_TYPE=Debug -DYAML_BUILD_SHARED_LIBS=OFF -DBUILD_SHARED_LIBS=OFF -DYAML_CPP_BUILD_TESTS=OFF -DYAML_CPP_STATIC_DEFINE=1
