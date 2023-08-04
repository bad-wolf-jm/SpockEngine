cmake_minimum_required(VERSION 3.20)

project(LTSimulationEngine LANGUAGES CXX CUDA C)

# cuda_compile_and_embed(g_SensorModelEnvironmentSampler Scene/EnvironmentSampler/DevicePrograms.cu)
# cuda_compile_and_embed(embedded_ptx_code Scene/Renderer/RayTracing/DevicePrograms.cu)

set(SHADER_EMBEDDED_FILES "")
set(SHADER_FILES "")

# add_library(LTSimulationEngineRuntimeShaders STATIC)

function(embed_shader TARGET output_var shader_file)
    set(shader_path ${CMAKE_CURRENT_SOURCE_DIR}/Source/${shader_file})
    set(shader_output_path ${CMAKE_CURRENT_SOURCE_DIR}/Embedded/Shaders)

    if (NOT EXISTS ${shader_output_path})
        file(MAKE_DIRECTORY ${shader_output_path})
    endif ()

    file(READ ${shader_path} content HEX)

    string(REGEX MATCHALL "([A-Fa-f0-9][A-Fa-f0-9])" SEPARATED_HEX ${content})

    set(counter 0)
    # Iterate through each of the bytes from the source file
    foreach (hex IN LISTS SEPARATED_HEX)
        # Write the hex string to the line with an 0x prefix and a , postfix to seperate the bytes of the file.
        string(APPEND output_c "0x${hex},")
        # Increment the element counter before the newline.
        math(EXPR counter "${counter}+1")
        if (counter GREATER 16)
            # Write a newline so that all of the array initializer gets spread across multiple lines.
            string(APPEND output_c "\n        ")
            set(counter 0)
        endif ()
    endforeach ()

    set(output_c "
#include \"${output_var}.h\"

namespace SE::Private::Shaders
{
    const std::vector<uint8_t> ${output_var}_data = {
        ${output_c}
    }\;
}")
    
    # Generate the contents that will be contained in the header file.
    set(output_h "
#pragma once

#include \"stdint.h\"
#include <vector>

namespace SE::Private::Shaders
{
    extern const std::vector<uint8_t> ${output_var}_data\;
}")

    file(WRITE ${shader_output_path}/${output_var}.h ${output_h})
    file(WRITE ${shader_output_path}/${output_var}.cpp ${output_c})

    set_source_files_properties(${shader_output_path}/${output_var}.h PROPERTIES GENERATED TRUE)
    set_source_files_properties(${shader_output_path}/${output_var}.cpp PROPERTIES GENERATED TRUE)
    target_sources(${TARGET} PRIVATE ${shader_path})
    target_sources(${TARGET} PRIVATE ${shader_output_path}/${output_var}.cpp)
endfunction(embed_shader)

 
# embed_shader(
#     LTSimulationEngineRuntimeShaders 
#     gToneMap                                       
#     "Common/ToneMap.glsl"
# )

# embed_shader(
#     LTSimulationEngineRuntimeShaders 
#     gPBRFunctions                                  
#     "Common/PBRFunctions.glsl"
# )

# embed_shader(
#     LTSimulationEngineRuntimeShaders 
#     gGetNormalFromMap                              
#     "Common/GetNormalFromMap.glsl"
# )

# embed_shader(
#     LTSimulationEngineRuntimeShaders 
#     gVertexLayout                                  
#     "Common/VertexLayout.h"
# )

# embed_shader(
#     LTSimulationEngineRuntimeShaders 
#     gUIVertexShader                                
#     "ui_shader.vert"
# )

# embed_shader(
#     LTSimulationEngineRuntimeShaders 
#     gUIFragmentShader                              
#     "ui_shader.frag"
# )

# embed_shader(
#     LTSimulationEngineRuntimeShaders 
#     gCopyFragmentShader                            
#     "copy.frag"
# )

# embed_shader(
#     LTSimulationEngineRuntimeShaders 
#     gFXAAFragmentShader                            
#     "fxaa.frag"
# )

# embed_shader(
#     LTSimulationEngineRuntimeShaders 
#     gFXAACode                                      
#     "fxaa.h"
# )

# embed_shader(
#     LTSimulationEngineRuntimeShaders 
#     gFXAAVertexShader                              
#     "fxaa.vert"
# )

# embed_shader(
#     LTSimulationEngineRuntimeShaders 
#     gDirectionalShadowVertexShader                 
#     "shadow.vert"
# )

# embed_shader(
#     LTSimulationEngineRuntimeShaders 
#     gOmniDirectionalShadowVertexShader             
#     "OmniShadow.vert"
# )

# embed_shader(
#     LTSimulationEngineRuntimeShaders 
#     gOmniDirectionalShadowFragmentShader           
#     "OmniShadow.frag"
# )

# embed_shader(
#     LTSimulationEngineRuntimeShaders 
#     gFilterCubeVertexShader                        
#     "filtercube.vert"
# )

# embed_shader(
#     LTSimulationEngineRuntimeShaders 
#     gIrradianceCubeFragmentShader                  
#     "irradiancecube.frag"
# )

# embed_shader(
#     LTSimulationEngineRuntimeShaders 
#     gPrefilteredEnvironmentFragmentShader          
#     "prefilterenvmap.frag"
# )

# embed_shader(
#     LTSimulationEngineRuntimeShaders 
#     gSkyboxVertexShader                            
#     "skybox.vert"
# )

# embed_shader(
#     LTSimulationEngineRuntimeShaders 
#     gSkyboxFragmentShader                          
#     "skybox.frag"
# )

# embed_shader(
#     LTSimulationEngineRuntimeShaders 
#     gCoordinateGridVertexShader                    
#     "coordinategrid.vert"
# )

# embed_shader(
#     LTSimulationEngineRuntimeShaders 
#     gCoordinateGridFragmentShader                  
#     "coordinategrid.frag"
# )

# embed_shader(
#     LTSimulationEngineRuntimeShaders 
#     gGenerateBrdfVertexShader                      
#     "genbrdflut.vert"
# )

# embed_shader(
#     LTSimulationEngineRuntimeShaders 
#     gGenerateBrdfFragmentShader                    
#     "genbrdflut.frag"
# )

# embed_shader(
#     LTSimulationEngineRuntimeShaders 
#     gPBRMeshVertexShader                           
#     "PBRMeshShader.vert"
# )

# embed_shader(
#     LTSimulationEngineRuntimeShaders 
#     gPBRMeshFragmentShaderPreamble                 
#     "PBRMeshShaderPreamble.frag"
# )

# embed_shader(
#     LTSimulationEngineRuntimeShaders 
#     gPBRMeshFragmentShaderCalculation              
#     "PBRMeshShaderCalculation.frag"
# )

# embed_shader(
#     LTSimulationEngineRuntimeShaders 
#     gParticleSystemVertexShader                    
#     "ParticleSystem.vert"
# )

# embed_shader(
#     LTSimulationEngineRuntimeShaders 
#     gParticleSystemFragmentShader                  
#     "ParticleSystem.frag"
# )

# embed_shader(
#     LTSimulationEngineRuntimeShaders 
#     gSurfaceVertexShader                           
#     "SurfaceShader.vert"
# )

# embed_shader(
#     LTSimulationEngineRuntimeShaders 
#     gSurfaceFragmentShader                         
#     "SurfaceShader.frag"
# )

# embed_shader(
#     LTSimulationEngineRuntimeShaders 
#     gWireframeVertexShader                         
#     "WireframeShader.vert"
# )

# embed_shader(
#     LTSimulationEngineRuntimeShaders 
#     gWirefrrameFragmentShader                      
#     "WireframeShader.frag"
# )

# embed_shader(
#     LTSimulationEngineRuntimeShaders 
#     gUnlitVertexShader                             
#     "Unlit.vert"
# )

# embed_shader(
#     LTSimulationEngineRuntimeShaders 
#     gUnlitFragmentShader                           
#     "Unlit.frag"
# )

# embed_shader(
#     LTSimulationEngineRuntimeShaders 
#     gDeferredLightingVertexShader                  
#     "Deferred/DeferredLighting.vert"
# )

# embed_shader(
#     LTSimulationEngineRuntimeShaders 
#     gDeferredLightingFragmentShaderPreamble        
#     "Deferred/DeferredLighting_preamble.frag"
# )

# embed_shader(
#     LTSimulationEngineRuntimeShaders 
#     gDeferredLightingFragmentShaderCalculation     
#     "Deferred/DeferredLighting_calculation.frag"
# )

# embed_shader(
#     LTSimulationEngineRuntimeShaders 
#     gDeferredGeometryPassVertexShader              
#     "Deferred/MRT.vert"
# )

# embed_shader(
#     LTSimulationEngineRuntimeShaders 
#     gDeferredGeometryPassFragmentShaderPreamble    
#     "Deferred/MRT_preamble.frag"
# )

# embed_shader(
#     LTSimulationEngineRuntimeShaders 
#     gDeferredGeometryPassFragmentShaderCalculation 
#     "Deferred/MRT_calculation.frag"
# )

    
