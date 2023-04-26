#include "GraphicsPipeline.h"

#include "Core/Logging.h"
#include "Core/Memory.h"

#include <stdexcept>

namespace SE::Graphics
{
    GraphicsPipeline::GraphicsPipeline( Ref<VkGraphicContext> aGraphicContext, GraphicsPipelineCreateInfo &aCreateInfo )
        : mGraphicContext( aGraphicContext )
    {
        std::vector<Ref<sVkDescriptorSetLayoutObject>> l_DescriptorSetLayouts( aCreateInfo.mSetLayouts.size() );
        for( uint32_t i = 0; i < aCreateInfo.mSetLayouts.size(); i++ )
            l_DescriptorSetLayouts[i] = aCreateInfo.mSetLayouts[i]->GetVkDescriptorSetLayoutObject();

        mPipelineLayoutObject =
            SE::Core::New<sVkPipelineLayoutObject>( mGraphicContext, l_DescriptorSetLayouts, aCreateInfo.mPushConstants );

        sDepthTesting lDepth{};
        lDepth.mDepthComparison  = aCreateInfo.mDepthComparison;
        lDepth.mDepthTestEnable  = aCreateInfo.mDepthTestEnable;
        lDepth.mDepthWriteEnable = aCreateInfo.mDepthWriteEnable;

        if( aCreateInfo.mOpaque )
        {
            sBlending lBlending{};
            mPipelineObject = SE::Core::New<sVkPipelineObject>(
                mGraphicContext, aCreateInfo.mSampleCount, aCreateInfo.mInputBufferLayout, aCreateInfo.mInstanceBufferLayout,
                aCreateInfo.mTopology, aCreateInfo.mCulling, aCreateInfo.mLineWidth, lDepth, lBlending, aCreateInfo.mShaderStages,
                mPipelineLayoutObject, aCreateInfo.mRenderPass );
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
                mGraphicContext, aCreateInfo.mSampleCount, aCreateInfo.mInputBufferLayout, aCreateInfo.mInstanceBufferLayout,
                aCreateInfo.mTopology, aCreateInfo.mCulling, aCreateInfo.mLineWidth, lDepth, lBlending, aCreateInfo.mShaderStages,
                mPipelineLayoutObject, aCreateInfo.mRenderPass );
        }
    }

} // namespace SE::Graphics