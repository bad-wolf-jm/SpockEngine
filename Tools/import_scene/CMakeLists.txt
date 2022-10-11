cmake_minimum_required(VERSION 3.20)

set(PROGRAM_NAME import_scene)

Include(FetchContent)

set(ASSIMP_BUILD_ASSIMP_TOOLS CACHE INTERNAL OFF)
set(ASSIMP_BUILD_TESTS CACHE INTERNAL OFF)
set(ASSIMP_INSTALL CACHE INTERNAL OFF)
set(BUILD_SHARED_LIBS CACHE INTERNAL OFF)

FetchContent_Declare(
  assimp
  GIT_REPOSITORY https://github.com/assimp/assimp.git
  GIT_TAG        master
  GIT_PROGRESS   TRUE )

FetchContent_MakeAvailable(assimp)

add_executable(${PROGRAM_NAME} WIN32
    Source/ConsoleMain.cpp
    # Source/Conversion.cpp
    # Source/Animation.cpp
    # Source/Importer.cpp
    # Source/Material.cpp
    # Source/Mesh.cpp
    # Source/SceneNode.cpp
)

target_include_directories(${PROGRAM_NAME} PRIVATE Source )
target_compile_definitions(${PROGRAM_NAME} PRIVATE VK_USE_PLATFORM_WIN32_KHR)
target_compile_options( ${PROGRAM_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr -Xcudafe "--display_error_number --diag_suppress=20012"> )
target_link_libraries( ${PROGRAM_NAME} LTSimulationEngineRuntime assimp tinygltf)
set_target_properties( ${PROGRAM_NAME} PROPERTIES LINK_FLAGS /SUBSYSTEM:CONSOLE )
