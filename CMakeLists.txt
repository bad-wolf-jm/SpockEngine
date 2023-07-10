cmake_minimum_required(VERSION 3.20)
# project(LTSimulationEngine LANGUAGES CUDA CXX C)
project(LTSimulationEngine LANGUAGES CUDA CXX C)

set(CMAKE_CXX_STANDARD 17)

# set(PROGRAM_FILES "C:\\Users\\jalbe")

include_directories("ThirdParty/mono/include")
message(STATUS ${CMAKE_SIZEOF_VOID_P})

include(cmake/configure_optix.cmake)
include_directories(${OptiX_INCLUDE})
find_package(CUDAToolkit REQUIRED)

find_package(Vulkan REQUIRED FATAL_ERROR)
Include(FetchContent)
# cmake_policy(SET CMP0002 OLD)

set (test OFF CACHE INTERNAL "Turn off tests")
set (benchmark OFF CACHE INTERNAL "Turn off tests")
FetchContent_Declare(
  uuid_v4
  GIT_REPOSITORY https://github.com/crashoz/uuid_v4.git
  GIT_TAG        master
  GIT_PROGRESS   TRUE )
FetchContent_MakeAvailable(uuid_v4)
include_directories(${uuid_v4_SOURCE_DIR})
set(SE_EXTRA_INCLUDES "${uuid_v4_SOURCE_DIR}")

FetchContent_Declare(
  yaml-cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
  GIT_TAG        yaml-cpp-0.7.0
  GIT_PROGRESS   TRUE )
FetchContent_MakeAvailable(yaml-cpp)
include_directories(${yaml-cpp_SOURCE_DIR}/include)
set(SE_EXTRA_INCLUDES "${yaml-cpp_SOURCE_DIR}/include" ${SE_EXTRA_INCLUDES})

FetchContent_Declare(
  entt
  GIT_REPOSITORY https://github.com/skypjack/entt.git
  GIT_TAG        f1d537d0354dea62d2ba1cc62f444165f29c8fba )
FetchContent_MakeAvailable(entt)
include_directories(${entt_SOURCE_DIR}/single_include)
include_directories(${entt_SOURCE_DIR}/src)
set(SE_EXTRA_INCLUDES ${entt_SOURCE_DIR}/single_include ${entt_SOURCE_DIR}/src ${SE_EXTRA_INCLUDES})

FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG        8.1.1
  GIT_PROGRESS   TRUE )
FetchContent_MakeAvailable(fmt)
include_directories(${fmt_SOURCE_DIR}/include)
set(SE_EXTRA_INCLUDES ${fmt_SOURCE_DIR}/include ${SE_EXTRA_INCLUDES})

FetchContent_Declare(
  glfw
  GIT_REPOSITORY https://github.com/glfw/glfw.git
  GIT_TAG        3.3.6
  GIT_PROGRESS   TRUE  )
FetchContent_MakeAvailable(glfw)
include_directories(${glfw_SOURCE_DIR}/include)
set(SE_EXTRA_INCLUDES ${glfw_SOURCE_DIR}/include ${SE_EXTRA_INCLUDES})

FetchContent_Declare(
  glm
  GIT_REPOSITORY https://github.com/g-truc/glm.git
  GIT_TAG        0.9.9.8
  GIT_PROGRESS   TRUE  )
FetchContent_MakeAvailable(glm)
include_directories(${glm_SOURCE_DIR})
set(SE_EXTRA_INCLUDES ${glm_SOURCE_DIR} ${SE_EXTRA_INCLUDES})

FetchContent_Declare(
  stb
  GIT_REPOSITORY https://github.com/nothings/stb.git
  GIT_TAG        master )
FetchContent_MakeAvailable(stb)
include_directories(${stb_SOURCE_DIR})
set(SE_EXTRA_INCLUDES ${stb_SOURCE_DIR} ${SE_EXTRA_INCLUDES})

FetchContent_Declare(
  argparse
  GIT_REPOSITORY https://github.com/p-ranav/argparse.git
  GIT_TAG        v2.3 )
FetchContent_MakeAvailable(argparse)
include_directories(${argparse_SOURCE_DIR}/include)
set(SE_EXTRA_INCLUDES ${argparse_SOURCE_DIR} ${SE_EXTRA_INCLUDES})

FetchContent_Declare(
  lua
  GIT_REPOSITORY https://github.com/marovira/lua
  GIT_TAG        master )
FetchContent_MakeAvailable(lua)
include_directories(${lua_SOURCE_DIR}/src)
set(SE_EXTRA_INCLUDES ${lua_SOURCE_DIR}/src ${SE_EXTRA_INCLUDES})

FetchContent_Declare(
  sol2
  GIT_REPOSITORY https://github.com/ThePhD/sol2.git
  GIT_TAG        v4.0.0-alpha )
FetchContent_MakeAvailable(sol2)
include_directories(${sol2_SOURCE_DIR}/include)
set(SE_EXTRA_INCLUDES ${sol2_SOURCE_DIR}/include ${SE_EXTRA_INCLUDES})

FetchContent_Declare(
  tinygltf
  GIT_REPOSITORY https://github.com/syoyo/tinygltf.git
  GIT_TAG        release
  GIT_PROGRESS   TRUE )
FetchContent_MakeAvailable(tinygltf)
include_directories(${tinygltf_SOURCE_DIR}/include)
set(SE_EXTRA_INCLUDES ${tinygltf_SOURCE_DIR} ${SE_EXTRA_INCLUDES})

FetchContent_Declare(
  tinyobjloader
  GIT_REPOSITORY https://github.com/tinyobjloader/tinyobjloader.git
  GIT_TAG        release
  GIT_PROGRESS   TRUE )
FetchContent_MakeAvailable(tinyobjloader)
include_directories(${tinyobjloader_SOURCE_DIR})
set(SE_EXTRA_INCLUDES ${tinyobjloader_SOURCE_DIR} ${SE_EXTRA_INCLUDES})

FetchContent_Declare(
  pugixml
  GIT_REPOSITORY https://github.com/zeux/pugixml.git
  GIT_TAG        v1.13
  GIT_PROGRESS   TRUE )
FetchContent_MakeAvailable(pugixml)
include_directories(${pugixml_SOURCE_DIR}/src)
set(SE_EXTRA_INCLUDES ${pugixml_SOURCE_DIR}/src ${SE_EXTRA_INCLUDES})

FetchContent_Declare(
  glslang
  GIT_REPOSITORY https://github.com/KhronosGroup/glslang
  GIT_TAG        12.2.0
  GIT_PROGRESS   TRUE )
FetchContent_MakeAvailable(glslang)
include_directories(${glslang_SOURCE_DIR})
set(SE_EXTRA_INCLUDES ${glslangl_SOURCE_DIR} ${SE_EXTRA_INCLUDES})


set(SE_EXTRA_INCLUDES ${SE_EXTRA_INCLUDES} PARENT_SCOPE)
set(EXTERNAL_LIBRARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty)

set(IMGUI_DIR ${EXTERNAL_LIBRARY_DIR}/imgui)
include_directories(${IMGUI_DIR})
include_directories(${IMGUI_DIR}/backends)
include_directories(${IMGUI_DIR}/misc/cpp)

include_directories(
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
    ${IMGUI_DIR}/misc/cpp )

include_directories(
    ${Vulkan_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${EXTERNAL_LIBRARY_DIR}/gli
    Source )

set(IMGUIZMO_DIR ${EXTERNAL_LIBRARY_DIR}/ImGuizmo)
include_directories(${IMGUIZMO_DIR})

set(IMPLOT_DIR ${EXTERNAL_LIBRARY_DIR}/implot)
include_directories(${IMPLOT_DIR})

set(IMGUI_NODE_EDITOR_DIR ${EXTERNAL_LIBRARY_DIR}/imgui-node-editor)
include_directories(${IMGUI_NODE_EDITOR_DIR})

include_directories(
    ${EXTERNAL_LIBRARY_DIR}/imgui
    ${EXTERNAL_LIBRARY_DIR}/gli
    ${EXTERNAL_LIBRARY_DIR}/zep/include )

add_compile_definitions(NOMINMAX)
add_compile_definitions(ZEP_FEATURE_CPP_FILE_SYSTEM=1)


include_directories(Shaders/Embedded)

add_subdirectory(ThirdParty)
add_subdirectory(Shaders)
add_subdirectory(Source)
