cmake_minimum_required(VERSION 3.20)

Include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.0.0-preview3 )

FetchContent_MakeAvailable(Catch2)

file(GLOB TEST_DATA_FILES "Include/*.cpp")

add_executable(lste_cuda_test WIN32
  TestUtils.cpp

  Test_AbstractTexture.cpp
  Test_CudaTexture.cpp
  Test_CudaBuffer.cpp
  Test_EntityRegistry.cpp
  Test_ComputationScope.cpp
  # Test_ConfigurationReader.cpp
  # ${TEST_DATA_FILES}
  # Test_FPGA_Model.cpp
  # Test_Scripting.cpp
)

target_include_directories(lste_cuda_test PRIVATE Source C:/vcpkg/installed/x64-windows/include ${CUDA_INCLUDE_DIRS})

target_compile_definitions( lste_cuda_test PRIVATE VK_USE_PLATFORM_WIN32_KHR )
target_compile_options( lste_cuda_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr -Xcudafe "--display_error_number --diag_suppress=20012"> )
target_link_libraries( lste_cuda_test LTSimulationEngineRuntime Catch2::Catch2WithMain CUDA::cudart CUDA::curand)
set_target_properties( lste_cuda_test PROPERTIES LINK_FLAGS "/DEBUG /SUBSYSTEM:CONSOLE" )
