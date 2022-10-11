cmake_minimum_required(VERSION 3.20)

set(PROGRAM_NAME package_assets)

add_executable(${PROGRAM_NAME} WIN32
    Source/Main.cpp
)

target_include_directories(${PROGRAM_NAME} PRIVATE Source )

target_compile_definitions(${PROGRAM_NAME} PRIVATE VK_USE_PLATFORM_WIN32_KHR)
target_compile_options( ${PROGRAM_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr -Xcudafe "--display_error_number --diag_suppress=20012"> )
target_link_libraries( ${PROGRAM_NAME} LTSimulationEngineRuntime )
set_target_properties( ${PROGRAM_NAME} PROPERTIES LINK_FLAGS /SUBSYSTEM:CONSOLE )
