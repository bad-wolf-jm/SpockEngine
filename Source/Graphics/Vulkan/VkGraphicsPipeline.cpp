#include "VkGraphicsPipeline.h"
#include "VkDescriptorSetLayout.h"
#include "VkRenderContext.h"
#include "VkShaderProgram.h"

#include "Core/Logging.h"
#include "Core/Memory.h"
#include "Shader/Compiler.h"

#include <stdexcept>

namespace SE::Graphics
{
    VkGraphicsPipeline::VkGraphicsPipeline( ref_t<VkGraphicContext> aGraphicContext, ref_t<VkRenderContext> aRenderContext,
                                            ePrimitiveTopology aTopology )
        : IGraphicsPipeline( aGraphicContext, aRenderContext, aTopology )
    {
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
            auto lUIVertexShader = Cast<VkShaderProgram>( lShader.mProgram )->GetShaderModule();

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