#include "Compiler.h"

#include "Core/Logging.h"

#include <iterator>

#include "glslang/Include/glslang_c_interface.h"
#include "glslang/Include/glslang_c_shader_types.h"
#include "glslang/Public/resource_limits_c.h"

namespace SE::Graphics
{
    static fs::path              gShaderCache       = "";
    static std::vector<fs::path> gShaderIncludePath = {};

    void SetShaderCacheFolder( std::string const &aPath )
    {
        gShaderCache = aPath;
    }
    void AddShaderIncludePath( std::string const &aPath )
    {
        gShaderIncludePath.push_back( aPath );
    }

    static std::tuple<char *, size_t> ReadFile( const char *header_name )
    {
        std::string lHeaderPath = "";
        for( auto const &lIncludeFolder : gShaderIncludePath )
        {
            if( fs::exists( lIncludeFolder / std::string( header_name ) ) )
            {
                lHeaderPath = ( lIncludeFolder / std::string( header_name ) ).string();

                break;
            }
        }

        if( lHeaderPath.empty() )
            return { nullptr, 0 };

        std::ifstream lFileObject( lHeaderPath, std::ios::ate | std::ios::binary );

        if( !lFileObject.is_open() )
            throw std::runtime_error( "failed to open file!" );

        size_t            lFileSize = (size_t)lFileObject.tellg();
        std::vector<char> lBuffer( lFileSize );

        lFileObject.seekg( 0 );
        lFileObject.read( lBuffer.data(), lFileSize );
        lFileObject.close();

        char *lResult = (char *)malloc( lFileSize );

        memcpy( lResult, lBuffer.data(), lFileSize );
        return { lResult, lFileSize };
    }

    static glsl_include_result_t *IncludeLocalFile( void *ctx, const char *header_name, const char *includer_name,
                                                    size_t include_depth )
    {
        auto const [lData, lSize] = ReadFile( header_name );

        return new glsl_include_result_s{ header_name, lData, lSize };
    }

    /* Callback for system file inclusion */
    static glsl_include_result_t *IncludeSystemFile( void *ctx, const char *header_name, const char *includer_name,
                                                     size_t include_depth )
    {
        auto const [lData, lSize] = ReadFile( header_name );

        return new glsl_include_result_s{ header_name, lData, lSize };
    }

    /* Callback for include result destruction */
    static int FreeIncludeResult( void *ctx, glsl_include_result_t *result )
    {
        free( (void *)result->header_data );
        return 0;
    }

    void Compile( eShaderStageTypeFlags aShaderStage, std::string const &aCode, std::vector<uint32_t> &aOutput )
    {
        glslang_input_t lInputDescription{};
        lInputDescription.language = GLSLANG_SOURCE_GLSL;

        switch( aShaderStage )
        {
        case eShaderStageTypeFlags::GEOMETRY:
            lInputDescription.stage = GLSLANG_STAGE_GEOMETRY;
            break;
        case eShaderStageTypeFlags::FRAGMENT:
            lInputDescription.stage = GLSLANG_STAGE_FRAGMENT;
            break;
        case eShaderStageTypeFlags::TESSELATION_CONTROL:
            lInputDescription.stage = GLSLANG_STAGE_TESSCONTROL;
            break;
        case eShaderStageTypeFlags::TESSELATION_EVALUATION:
            lInputDescription.stage = GLSLANG_STAGE_TESSEVALUATION;
            break;
        case eShaderStageTypeFlags::COMPUTE:
            lInputDescription.stage = GLSLANG_STAGE_COMPUTE;
            break;
        case eShaderStageTypeFlags::VERTEX:
        default:
            lInputDescription.stage = GLSLANG_STAGE_VERTEX;
            break;
        }

        lInputDescription.client                            = GLSLANG_CLIENT_VULKAN;
        lInputDescription.client_version                    = GLSLANG_TARGET_VULKAN_1_3;
        lInputDescription.target_language                   = GLSLANG_TARGET_SPV;
        lInputDescription.target_language_version           = GLSLANG_TARGET_SPV_1_3;
        lInputDescription.code                              = aCode.c_str();
        lInputDescription.default_version                   = 460;
        lInputDescription.default_profile                   = GLSLANG_NO_PROFILE;
        lInputDescription.force_default_version_and_profile = false;
        lInputDescription.forward_compatible                = false;
        lInputDescription.messages                          = GLSLANG_MSG_VULKAN_RULES_BIT;
        lInputDescription.resource                          = glslang_default_resource();

        lInputDescription.callbacks.include_system      = IncludeSystemFile;
        lInputDescription.callbacks.include_local       = IncludeLocalFile;
        lInputDescription.callbacks.free_include_result = FreeIncludeResult;

        glslang_initialize_process();

        glslang_shader_t *lNewShader = glslang_shader_create( &lInputDescription );

        if( !glslang_shader_preprocess( lNewShader, &lInputDescription ) )
        {
            SE::Logging::Info( "[PREPROCESS] {}", glslang_shader_get_info_log( lNewShader ) );
            SE::Logging::Info( "[PREPROCESS_DEBUG] {}", glslang_shader_get_info_debug_log( lNewShader ) );
        }

        if( !glslang_shader_parse( lNewShader, &lInputDescription ) )
        {
            // std::ofstream lOutput( "D:\\Work\\Git\\SpockEngine\\test_fs.cpp" );
            // lOutput << glslang_shader_get_preprocessed_code( lNewShader );
            // lOutput.close();
            SE::Logging::Info( "[PARSE] {}", glslang_shader_get_info_log( lNewShader ) );
            SE::Logging::Info( "[PARSE] {}", glslang_shader_get_info_debug_log( lNewShader ) );
        }

        glslang_program_t *lNewProgram = glslang_program_create();
        glslang_program_add_shader( lNewProgram, lNewShader );

        if( !glslang_program_link( lNewProgram, GLSLANG_MSG_SPV_RULES_BIT | GLSLANG_MSG_VULKAN_RULES_BIT ) )
        {
            SE::Logging::Info( "[LINK] {}", glslang_program_get_info_log( lNewProgram ) );
            SE::Logging::Info( "[LINK] {}", glslang_program_get_info_debug_log( lNewProgram ) );
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
