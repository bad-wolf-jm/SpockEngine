#include "GraphicsPipeline.h"

#include "Core/Logging.h"
#include "Core/Memory.h"

#include <stdexcept>

namespace LTSE::Graphics
{
    GraphicsPipeline::GraphicsPipeline( GraphicContext &a_GraphicContext, GraphicsPipelineCreateInfo &a_CreateInfo )
        : mGraphicContext( a_GraphicContext )
    {
        std::vector<Ref<Internal::sVkDescriptorSetLayoutObject>> l_DescriptorSetLayouts( a_CreateInfo.SetLayouts.size() );
        for( uint32_t i = 0; i < a_CreateInfo.SetLayouts.size(); i++ )
            l_DescriptorSetLayouts[i] = a_CreateInfo.SetLayouts[i]->GetVkDescriptorSetLayoutObject();

        m_PipelineLayoutObject = LTSE::Core::New<Internal::sVkPipelineLayoutObject>(
            mGraphicContext.mContext, l_DescriptorSetLayouts, a_CreateInfo.PushConstants );

        Internal::sDepthTesting lDepth{};
        lDepth.mDepthComparison  = a_CreateInfo.DepthComparison;
        lDepth.mDepthTestEnable  = a_CreateInfo.DepthTestEnable;
        lDepth.mDepthWriteEnable = a_CreateInfo.DepthWriteEnable;

        if( a_CreateInfo.Opaque )
        {
            Internal::sBlending lBlending{};
            m_PipelineObject = LTSE::Core::New<Internal::sVkPipelineObject>( mGraphicContext.mContext, a_CreateInfo.SampleCount,
                a_CreateInfo.InputBufferLayout, a_CreateInfo.InstanceBufferLayout, a_CreateInfo.Topology, a_CreateInfo.Culling,
                a_CreateInfo.LineWidth, lDepth, lBlending, a_CreateInfo.mShaderStages, m_PipelineLayoutObject,
                a_CreateInfo.RenderPass );
        }
        else
        {
            Internal::sBlending lBlending{};
            lBlending.mEnable              = true;
            lBlending.mSourceColorFactor   = eBlendFactor::SRC_ALPHA;
            lBlending.mDestColorFactor     = eBlendFactor::ONE_MINUS_SRC_ALPHA;
            lBlending.mColorBlendOperation = eBlendOperation::ADD;

            lBlending.mSourceAlphaFactor   = eBlendFactor::ZERO;
            lBlending.mDestAlphaFactor     = eBlendFactor::ONE;
            lBlending.mAlphaBlendOperation = eBlendOperation::MAX;

            m_PipelineObject = LTSE::Core::New<Internal::sVkPipelineObject>( mGraphicContext.mContext, a_CreateInfo.SampleCount,
                a_CreateInfo.InputBufferLayout, a_CreateInfo.InstanceBufferLayout, a_CreateInfo.Topology, a_CreateInfo.Culling,
                a_CreateInfo.LineWidth, lDepth, lBlending, a_CreateInfo.mShaderStages, m_PipelineLayoutObject,
                a_CreateInfo.RenderPass );
        }
    }

} // namespace LTSE::Graphics