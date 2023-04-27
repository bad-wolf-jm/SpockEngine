#include "VkGraphicsPipeline.h"

#include "Core/Logging.h"
#include "Core/Memory.h"

#include <stdexcept>

namespace SE::Graphics
{
    VkGraphicsPipeline::VkGraphicsPipeline( Ref<IGraphicContext> aGraphicContext, Ref<IRenderContext> aRenderContext )
        : IGraphicsPipeline( aGraphicContext, aRenderContext )
    {
    }

    void VkGraphicsPipeline::Build()
    {
        std::vector<Ref<sVkDescriptorSetLayoutObject>> lDescriptorSetLayouts( mSetLayouts.size() );
        for( uint32_t i = 0; i < mSetLayouts.size(); i++ ) lDescriptorSetLayouts[i] = mSetLayouts[i]->GetVkDescriptorSetLayoutObject();

        mPipelineLayoutObject =
            SE::Core::New<sVkPipelineLayoutObject>( Cast<VkGraphicContext>( mGraphicContext ), lDescriptorSetLayouts, mPushConstants );

        sDepthTesting lDepth{};
        lDepth.mDepthComparison  = mDepthComparison;
        lDepth.mDepthTestEnable  = mDepthTestEnable;
        lDepth.mDepthWriteEnable = mDepthWriteEnable;

        if( mOpaque )
        {
            sBlending lBlending{};
            mPipelineObject = SE::Core::New<sVkPipelineObject>(
                Cast<VkGraphicContext>( mGraphicContext ), mSampleCount, mInputBufferLayout, mInstanceBufferLayout, mTopology,
                mCulling, mLineWidth, lDepth, lBlending, mShaderStages, mPipelineLayoutObject, mRenderPass );
        }
        else
        {
            sBlending lBlending{};
            lBlending.mEnable              = true;
            lBlending.mSourceColorFactor   = eBlendFactor::SRC_ALPHA;
            lBlending.mDestColorFactor     = eBlendFactor::ONE_MINUS_SRC_ALPHA;
            lBlending.mColorBlendOperation = eBlendOperation::ADD;

            lBlending.mSourceAlphaFactor   = eBlendFactor::ZERO;
            lBlending.mDestAlphaFactor     = eBlendFactor::ONE;
            lBlending.mAlphaBlendOperation = eBlendOperation::MAX;

            mPipelineObject = SE::Core::New<sVkPipelineObject>(
                Cast<VkGraphicContext>( mGraphicContext ), mSampleCount, mInputBufferLayout, mInstanceBufferLayout, mTopology,
                mCulling, mLineWidth, lDepth, lBlending, mShaderStages, mPipelineLayoutObject, mRenderPass );
        }
    }
} // namespace SE::Graphics