#include "Compiler.h"

#include <iterator>

#include "glslang/Include/glslang_c_interface.h"
#include "glslang/Include/glslang_c_shader_types.h"
#include "glslang/Public/resource_limits_c.h"

namespace SE::Graphics
{
    void Compile( eShaderStageTypeFlags aShaderStage, std::string const &aCode, std::vector<uint32_t> &aOutput )
    {
        glslang_input_t lInputDescription{};
        lInputDescription.language = GLSLANG_SOURCE_GLSL;

        switch( aShaderStage )
        {
        case eShaderStageTypeFlags::GEOMETRY: lInputDescription.stage = GLSLANG_STAGE_GEOMETRY; break;
        case eShaderStageTypeFlags::FRAGMENT: lInputDescription.stage = GLSLANG_STAGE_FRAGMENT; break;
        case eShaderStageTypeFlags::TESSELATION_CONTROL: lInputDescription.stage = GLSLANG_STAGE_TESSCONTROL; break;
        case eShaderStageTypeFlags::TESSELATION_EVALUATION: lInputDescription.stage = GLSLANG_STAGE_TESSEVALUATION; break;
        case eShaderStageTypeFlags::COMPUTE: lInputDescription.stage = GLSLANG_STAGE_COMPUTE; break;
        case eShaderStageTypeFlags::VERTEX:
        default: lInputDescription.stage = GLSLANG_STAGE_VERTEX; break;
        }

        lInputDescription.stage                             = GLSLANG_STAGE_VERTEX;
        lInputDescription.client                            = GLSLANG_CLIENT_VULKAN;
        lInputDescription.client_version                    = GLSLANG_TARGET_VULKAN_1_1;
        lInputDescription.target_language                   = GLSLANG_TARGET_SPV;
        lInputDescription.target_language_version           = GLSLANG_TARGET_SPV_1_3;
        lInputDescription.code                              = aCode.c_str();
        lInputDescription.default_version                   = 100;
        lInputDescription.default_profile                   = GLSLANG_NO_PROFILE;
        lInputDescription.force_default_version_and_profile = false;
        lInputDescription.forward_compatible                = false;
        lInputDescription.messages                          = GLSLANG_MSG_DEFAULT_BIT;
        lInputDescription.resource                          = glslang_default_resource();

        glslang_initialize_process();

        glslang_shader_t *lNewShader = glslang_shader_create( &lInputDescription );

        if( !glslang_shader_preprocess( lNewShader, &lInputDescription ) )
        {
            // use glslang_shader_get_info_log() and glslang_shader_get_info_debug_log()
        }

        if( !glslang_shader_parse( lNewShader, &lInputDescription ) )
        {
            // use glslang_shader_get_info_log() and glslang_shader_get_info_debug_log()
        }

        glslang_program_t *lNewProgram = glslang_program_create();
        glslang_program_add_shader( lNewProgram, lNewShader );

        if( !glslang_program_link( lNewProgram, GLSLANG_MSG_SPV_RULES_BIT | GLSLANG_MSG_VULKAN_RULES_BIT ) )
        {
            // use glslang_program_get_info_log() and glslang_program_get_info_debug_log();
        }

        glslang_program_SPIRV_generate( lNewProgram, lInputDescription.stage );

        if( glslang_program_SPIRV_get_messages( lNewProgram ) )
        {
            printf( "%s", glslang_program_SPIRV_get_messages( lNewProgram ) );
        }

        auto *lCodePtr  = glslang_program_SPIRV_get_ptr( lNewProgram );
        auto  lCodeSize = glslang_program_SPIRV_get_size( lNewProgram );
        aOutput.reserve( glslang_program_SPIRV_get_size( lNewProgram ) * sizeof( unsigned int ) );
        std::copy( lCodePtr, lCodePtr + lCodeSize, std::back_inserter( aOutput ) );

        glslang_shader_delete( lNewShader );
        glslang_program_delete( lNewProgram );
        glslang_finalize_process();
    }
} // namespace SE::Graphics
