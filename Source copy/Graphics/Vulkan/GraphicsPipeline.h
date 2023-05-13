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
        std::vector<sShader>   mShaderStages        = {};
        sBufferLayout          InputBufferLayout    = {};
        sBufferLayout          InstanceBufferLayout = {};
        ePrimitiveTopology     Topology             = ePrimitiveTopology::TRIANGLES;
        eFaceCulling           Culling              = eFaceCulling::BACK;
        uint8_t                SampleCount          = 1;
        float                  LineWidth            = 1.0f;
        bool                   DepthWriteEnable     = false;
        bool                   DepthTestEnable      = false;
        bool                   Opaque               = false;
        eDepthCompareOperation DepthComparison      = eDepthCompareOperation::ALWAYS;

        std::vector<sPushConstantRange>            PushConstants = {};
        Ref<sVkAbstractRenderPassObject> RenderPass    = nullptr;
        std::vector<Ref<DescriptorSetLayout>>      SetLayouts    = {};
    };

    class GraphicsPipeline
    {
      public:
        GraphicsPipeline( Ref<VkGraphicContext> a_GraphicContext, GraphicsPipelineCreateInfo &a_CreateInfo );
        ~GraphicsPipeline() = default;

        Ref<sVkPipelineObject>       GetVkPipelineObject() { return m_PipelineObject; }
        Ref<sVkPipelineLayoutObject> GetVkPipelineLayoutObject() { return m_PipelineLayoutObject; }

      private:
        Ref<VkGraphicContext>                  mGraphicContext{};
        Ref<sVkPipelineLayoutObject> m_PipelineLayoutObject = nullptr;
        Ref<sVkPipelineObject>       m_PipelineObject       = nullptr;
    };

} // namespace SE::Graphics