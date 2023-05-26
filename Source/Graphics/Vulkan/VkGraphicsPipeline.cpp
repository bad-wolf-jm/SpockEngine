#include "VkGraphicsPipeline.h"
#include "VkDescriptorSetLayout.h"
#include "VkRenderContext.h"

#include "Core/Logging.h"
#include "Core/Memory.h"
#include "Shader/Compiler.h"

#include <stdexcept>

namespace SE::Graphics
{
    VkGraphicsPipeline::VkGraphicsPipeline( Ref<VkGraphicContext> aGraphicContext, Ref<VkRenderContext> aRenderContext,
                                            ePrimitiveTopology aTopology )
        : IGraphicsPipeline( aGraphicContext, aRenderContext, aTopology )
    {
    }

    static std::vector<char> ReadFile( const std::string &filename )
    {
        std::ifstream lFileObject( filename, std::ios::ate | std::ios::binary );

        if( !lFileObject.is_open() ) throw std::runtime_error( "failed to open file!" );

        size_t            lFileSize = (size_t)lFileObject.tellg();
        std::vector<char> lBuffer( lFileSize );

        lFileObject.seekg( 0 );
        lFileObject.read( lBuffer.data(), lFileSize );
        lFileObject.close();

        return lBuffer;
    }

    void VkGraphicsPipeline::Build()
    {
        for( uint32_t i = 0; i < mDescriptorSets.size(); i++ )
        {
            mDescriptorSetLayouts.push_back(
                Cast<VkDescriptorSetLayoutObject>( mDescriptorSets[i] )->GetVkDescriptorSetLayoutObject() );
        }

        mPipelineLayoutObject =
            SE::Core::New<sVkPipelineLayoutObject>( Cast<VkGraphicContext>( mGraphicContext ), mDescriptorSetLayouts, mPushConstants );

        sDepthTesting lDepth{};
        lDepth.mDepthComparison  = mDepthComparison;
        lDepth.mDepthTestEnable  = mDepthTestEnable;
        lDepth.mDepthWriteEnable = mDepthWriteEnable;

        auto lSampleCount = Cast<VkRenderContext>( mRenderContext )->GetRenderTarget()->mSpec.mSampleCount;

        for( auto const &lShader : mShaderStages )
        {
            fs::path lShaderPath      = "C:\\GitLab\\SpockEngine\\Resources\\Shaders\\Cache";
            fs::path lShaderName      = lShader.mPath.filename();
            fs::path lCacheShaderName = lShaderPath / fmt::format( "{}.spv", lShaderName.string() );

            Ref<ShaderModule> lUIVertexShader{};
            std::filesystem::file_time_type lCachedFileTime = std::filesystem::last_write_time(lCacheShaderName);
            std::filesystem::file_time_type lShaderSourceFileTime = std::filesystem::last_write_time(lShader.mPath);

            if( fs::exists( lCacheShaderName ) && (lCachedFileTime.time_since_epoch().count() > lShaderSourceFileTime.time_since_epoch().count()) )
            {
                lUIVertexShader =
                    New<ShaderModule>( Cast<VkGraphicContext>( mGraphicContext ), lCacheShaderName.string(), lShader.mShaderType );
            }
            else
            {
                auto        lProgram       = ReadFile( lShader.mPath.string() );
                std::string lProgramString = std::string( lProgram.begin(), lProgram.end() );

                std::vector<uint32_t> lByteCode( 0 );
                Compile( lShader.mShaderType, lProgramString, lByteCode );

                std::ofstream lFileObject( lCacheShaderName, std::ios::out | std::ios::binary );
                lFileObject.write( (char*)lByteCode.data(), lByteCode.size() * sizeof(uint32_t) );
                lFileObject.close();

                lUIVertexShader =
                    New<ShaderModule>( Cast<VkGraphicContext>( mGraphicContext ), lCacheShaderName.string(), lShader.mShaderType );
            }

            mShaders.push_back( sShader{ lUIVertexShader, lShader.mEntryPoint } );
        }

        sBlending lBlending{};
        if( !mOpaque )
        {
            lBlending.mEnable              = true;
            lBlending.mSourceColorFactor   = eBlendFactor::SRC_ALPHA;
            lBlending.mDestColorFactor     = eBlendFactor::ONE_MINUS_SRC_ALPHA;
            lBlending.mColorBlendOperation = eBlendOperation::ADD;
            lBlending.mSourceAlphaFactor   = eBlendFactor::ZERO;
            lBlending.mDestAlphaFactor     = eBlendFactor::ONE;
            lBlending.mAlphaBlendOperation = eBlendOperation::MAX;
        }

        mPipelineObject = SE::Core::New<sVkPipelineObject>(
            Cast<VkGraphicContext>( mGraphicContext ), (uint8_t)lSampleCount, mInputLayout, mInstancedInputLayout, mTopology, mCulling,
            mLineWidth, lDepth, lBlending, mShaders, mPipelineLayoutObject,
            Cast<VkRenderPassObject>( Cast<VkRenderContext>( mRenderContext )->GetRenderPass() ) );
    }
} // namespace SE::Graphics