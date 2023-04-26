#pragma once

#include <vulkan/vulkan.h>

#include "Core/Memory.h"

#include "DescriptorSet.h"
#include "Graphics/Vulkan/VkAbstractRenderPass.h"
#include "Graphics/Vulkan/VkGraphicContext.h"
#include "Graphics/Vulkan/VkPipeline.h"

namespace SE::Graphics
{
    struct GraphicsPipelineCreateInfo
    {
        sBufferLayout          mInputBufferLayout    = {};
        sBufferLayout          mInstanceBufferLayout = {};
        ePrimitiveTopology     mTopology             = ePrimitiveTopology::TRIANGLES;
        eFaceCulling           mCulling              = eFaceCulling::BACK;
        uint8_t                mSampleCount          = 1;
        float                  mLineWidth            = 1.0f;
        bool                   mDepthWriteEnable     = false;
        bool                   mDepthTestEnable      = false;
        bool                   mOpaque               = false;
        eDepthCompareOperation mDepthComparison      = eDepthCompareOperation::ALWAYS;

        std::vector<sShader> mShaderStages = {};

        std::vector<sPushConstantRange>       mPushConstants = {};
        Ref<sVkAbstractRenderPassObject>      mRenderPass    = nullptr;
        std::vector<Ref<DescriptorSetLayout>> mSetLayouts    = {};
    };

    class GraphicsPipeline
    {
      public:
        GraphicsPipeline( Ref<VkGraphicContext> aGraphicContext, GraphicsPipelineCreateInfo &aCreateInfo );
        ~GraphicsPipeline() = default;

        Ref<sVkPipelineObject>       GetVkPipelineObject() { return mPipelineObject; }
        Ref<sVkPipelineLayoutObject> GetVkPipelineLayoutObject() { return mPipelineLayoutObject; }

      private:
        Ref<VkGraphicContext>        mGraphicContext{};
        Ref<sVkPipelineLayoutObject> mPipelineLayoutObject = nullptr;
        Ref<sVkPipelineObject>       mPipelineObject       = nullptr;
    };

} // namespace SE::Graphics