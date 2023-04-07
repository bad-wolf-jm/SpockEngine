cmake_minimum_required(VERSION 3.20)

project(LTSimulationEngine CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)

set(THIRD_PARTY_INSTALLATION_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/libs)
SET(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/libs)

project(LeddarSimulator_ThirdParty CXX)
set(CMAKE_CXX_STANDARD 17)

set(EXTERNAL_LIBRARY_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(IMGUI_DIR ${EXTERNAL_LIBRARY_DIR}/imgui)
include_directories(${IMGUI_DIR})

include_directories(
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
    ${IMGUI_DIR}/misc/cpp )

add_library(imgui STATIC
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_stdlib.cpp
    ${IMGUI_DIR}/imgui_impl_glfw.cpp  )

set(IMGUIZMO_DIR ${EXTERNAL_LIBRARY_DIR}/ImGuizmo)
include_directories(${IMGUIZMO_DIR})
add_library(imguizmo STATIC ${IMGUIZMO_DIR}/ImGuizmo.cpp )

set(IMPLOT_DIR ${EXTERNAL_LIBRARY_DIR}/implot)
include_directories(${IMPLOT_DIR})
add_library(implot STATIC
    ${IMPLOT_DIR}/implot.cpp
    ${IMPLOT_DIR}/implot_items.cpp
    ${IMPLOT_DIR}/implot_demo.cpp )

set(IMGUI_NODE_EDITOR_DIR ${EXTERNAL_LIBRARY_DIR}/imgui-node-editor)
include_directories(${IMGUI_NODE_EDITOR_DIR})
add_library(imgui-node-editor STATIC
    imgui-node-editor/crude_json.cpp
    imgui-node-editor/imgui_canvas.cpp
    imgui-node-editor/imgui_node_editor.cpp
    imgui-node-editor/imgui_node_editor_api.cpp )

include_directories(
    ${EXTERNAL_LIBRARY_DIR}/imgui
    ${EXTERNAL_LIBRARY_DIR}/gli
    ${EXTERNAL_LIBRARY_DIR}/zep/include
)

# add_compile_definitions(ZEP_FEATURE_CPP_FILE_SYSTEM=1)
# add_subdirectory(zep/src)

# add_subdirectory(glfw)
# add_subdirectory(fmt)




# add_compile_definitions(YAML_CPP_STATIC_DEFINE)
# set(CMAKE_POLICY_DEFAULT_CMP0091 NEW)
# add_subdirectory(yaml-cpp)
