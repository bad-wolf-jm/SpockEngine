
#ifdef APIENTRY
#    undef APIENTRY
#endif
#include <chrono>
#include <cstdlib>
#include <shlobj.h>

#include <argparse/argparse.hpp>
#include <filesystem>
#include <fstream>
#include <yaml-cpp/yaml.h>

#include <direct.h>
#include <fstream>
#include <iostream>
#include <limits.h>
#include <string>

// #include "Core/Logging.h"
// #include "Core/Math/Types.h"
// #include "Core/Memory.h"
// #include "Engine/Engine.h"
#include "Graphics/API.h"

// #include "Editor/BaseEditorApplication.h"
#include "Renderer/MaterialSystem.h"
#include "Shader/Compiler.h"

#include "glslang/Include/glslang_c_interface.h"
#include "glslang/Include/glslang_c_shader_types.h"
#include "glslang/Public/resource_limits_c.h"

// #include "DotNet/Runtime.h"

using namespace SE::Core;
using namespace SE::Graphics;
// using namespace SE::Core::UI;

namespace fs = std::filesystem;

static glslang_shader_t *Compile( eShaderStageTypeFlags aShaderStage, std::string const &aCode )
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

    // lInputDescription.callbacks.include_system      = IncludeSystemFile;
    // lInputDescription.callbacks.include_local       = IncludeLocalFile;
    // lInputDescription.callbacks.free_include_result = FreeIncludeResult;

    glslang_initialize_process();

    glslang_shader_t *lNewShader = glslang_shader_create( &lInputDescription );

    if( !glslang_shader_preprocess( lNewShader, &lInputDescription ) )
    {
        SE::Logging::Info( "{}", glslang_shader_get_info_log( lNewShader ) );
        SE::Logging::Info( "{}", glslang_shader_get_info_debug_log( lNewShader ) );
    }

    return lNewShader;
}

int main( int argc, char **argv )
{
    auto mGraphicContext = CreateGraphicContext( 1 );

    auto lMaterialSystem = New<MaterialSystem>( mGraphicContext );

    auto lMaterial = lMaterialSystem->BeginMaterial( "TEST_MATERIAL" );

    lMaterialSystem->EndMaterial( lMaterial );

    lMaterial.Get<sMaterialInfo>().mRequiresUV0     = false;
    lMaterial.Get<sMaterialInfo>().mRequiresUV1     = false;
    lMaterial.Get<sMaterialInfo>().mRequiresNormals = false;

    {
        auto lShader     = lMaterialSystem->CreateVertexShader( lMaterial );
        auto lShaderCode = lShader->Program();

        glslang_shader_t *lPreprocessed = Compile( eShaderStageTypeFlags::VERTEX, lShaderCode );

        std::ofstream lOutput( "D:\\Work\\Git\\SpockEngine\\test_vs.cpp" );
        lOutput << glslang_shader_get_preprocessed_code( lPreprocessed );
    }

    {
        auto lShader     = lMaterialSystem->CreateFragmentShader( lMaterial );
        auto lShaderCode = lShader->Program();

        glslang_shader_t *lPreprocessed = Compile( eShaderStageTypeFlags::FRAGMENT, lShaderCode );
        std::ofstream     lOutput( "D:\\Work\\Git\\SpockEngine\\test_fs.cpp" );
        lOutput << glslang_shader_get_preprocessed_code( lPreprocessed );
    }

    return 0;
}
