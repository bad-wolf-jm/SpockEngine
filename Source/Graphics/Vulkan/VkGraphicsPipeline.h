#pragma once

#include "Core/Memory.h"

#include "Graphics/Interface/IGraphicsPipeline.h"

#include "VkPipeline.h"
#include "VkRenderContext.h"

namespace SE::Graphics
{
    class VkGraphicsPipeline : public IGraphicsPipeline
    {
      public:
        VkGraphicsPipeline( Ref<VkGraphicContext> aGraphicContext, Ref<VkRenderContext> aRenderContext, ePrimitiveTopology aTopology );
        ~VkGraphicsPipeline() = default;

        Ref<sVkPipelineObject>       GetVkPipelineObject() { return mPipelineObject; }
        Ref<sVkPipelineLayoutObject> GetVkPipelineLayoutObject() { return mPipelineLayoutObject; }

        void Build();

      private:
        std::vector<Ref<sVkDescriptorSetLayoutObject>> mDescriptorSetLayouts{};

        Ref<sVkPipelineLayoutObject> mPipelineLayoutObject = nullptr;
        Ref<sVkPipelineObject>       mPipelineObject       = nullptr;
    };

} // namespace SE::Graphics