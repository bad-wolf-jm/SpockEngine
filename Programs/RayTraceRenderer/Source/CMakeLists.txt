find_package(OpenGL REQUIRED)

include_directories(${OptiX_INCLUDE})

cuda_compile_and_embed(embedded_ptx_code devicePrograms.cu)

# cuda_add_library(toneMap toneMap.cu)

add_executable(ex12_denoiseSeparateChannels
  ${embedded_ptx_code}
  toneMap.cu
  SampleRenderer.h
  SampleRenderer.cpp
  Model.cpp
  main.cpp
)

target_compile_options( ex12_denoiseSeparateChannels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr -Xcudafe "--display_error_number --diag_suppress=20012"> )

target_link_libraries(ex12_denoiseSeparateChannels
  # toneMap
  gdt
  # optix dependencies, for rendering
  ${optix_LIBRARY}
  ${CUDA_LIBRARIES}
  ${CUDA_CUDA_LIBRARY}
  # glfw and opengl, for display
  glfWindow
  glfw
  ${OPENGL_gl_LIBRARY}
  LTSimulationEngineRuntime
  )
