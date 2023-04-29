#include "VkGraphicsPipeline.h"
#include "VkRenderContext.h"

#include "Core/Logging.h"
#include "Core/Memory.h"

#include <stdexcept>

namespace SE::Graphics
{
    VkGraphicsPipeline::VkGraphicsPipeline( Ref<VkGraphicContext> aGraphicContext, Ref<VkRenderContext> aRenderContext,
                                            ePrimitiveTopology aTopology )
        : IGraphicsPipeline( aGraphicContext, aRenderContext, aTopology )
    {
    }

    void VkGraphicsPipeline::Build()
    {
        for( uint32_t i = 0; i < mDescriptorLayout.size(); i++ )
        {
            auto lDescriptorSet = mDescriptorLayout[i].mDescriptors;

            std::vector<VkDescriptorSetLayoutBinding> lBindings{};

            for( uint32_t j = 0; j < lDescriptorSet.size(); j++ )
            {
                VkDescriptorSetLayoutBinding &lNewBinding = lBindings.emplace_back();

                lNewBinding.binding            = lDescriptorSet[i].mBindingIndex;
                lNewBinding.descriptorCount    = 1;
                lNewBinding.descriptorType     = (VkDescriptorType)lDescriptorSet[i].mType;
                lNewBinding.pImmutableSamplers = nullptr;

                if( lDescriptorSet[j].mShaderStages & eShaderStageTypeFlags::VERTEX )
                    lNewBinding.stageFlags |= VK_SHADER_STAGE_VERTEX_BIT;

                if( lDescriptorSet[j].mShaderStages & eShaderStageTypeFlags::COMPUTE )
                    lNewBinding.stageFlags |= VK_SHADER_STAGE_COMPUTE_BIT;

                if( lDescriptorSet[j].mShaderStages & eShaderStageTypeFlags::GEOMETRY )
                    lNewBinding.stageFlags |= VK_SHADER_STAGE_GEOMETRY_BIT;

                if( lDescriptorSet[j].mShaderStages & eShaderStageTypeFlags::FRAGMENT )
                    lNewBinding.stageFlags |= VK_SHADER_STAGE_FRAGMENT_BIT;

                if( lDescriptorSet[j].mShaderStages & eShaderStageTypeFlags::TESSELATION_CONTROL )
                    lNewBinding.stageFlags |= VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;

                if( lDescriptorSet[j].mShaderStages & eShaderStageTypeFlags::TESSELATION_EVALUATION )
                    lNewBinding.stageFlags |= VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
            }

            if( mDescriptorLayout[i].mIsUnbounded ) lBindings[lBindings.size() - 1].descriptorCount = 1024;

            mDescriptorSetLayouts.push_back( New<sVkDescriptorSetLayoutObject>( Cast<VkGraphicContext>( mGraphicContext ), lBindings,
                                                                                mDescriptorLayout[i].mIsUnbounded ) );
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
            auto lUIVertexShader =
                New<ShaderModule>( Cast<VkGraphicContext>( mGraphicContext ), lShader.mPath.string(), lShader.mShaderType );

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
            Cast<sVkAbstractRenderPassObject>( Cast<VkRenderContext>( mRenderContext )->GetRenderPass() ) );
    }
} // namespace SE::Graphics