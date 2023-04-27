#pragma once

// #include <vulkan/vulkan.h>

#include "Core/Memory.h"

#include "Graphics/Interface/IGraphicsPipeline.h"

#include "DescriptorSet.h"
#include "VkAbstractRenderPass.h"
#include "VkGraphicContext.h"
#include "VkPipeline.h"

namespace SE::Graphics
{
    class VkGraphicsPipeline : IGraphicsPipeline
    {
      public:
        VkGraphicsPipeline( Ref<IGraphicContext> aGraphicContext, Ref<IRenderContext> aRenderContext );
        ~VkGraphicsPipeline() = default;

        Ref<sVkPipelineObject>       GetVkPipelineObject() { return mPipelineObject; }
        Ref<sVkPipelineLayoutObject> GetVkPipelineLayoutObject() { return mPipelineLayoutObject; }

        void Build();

      private:
        Ref<sVkPipelineLayoutObject> mPipelineLayoutObject = nullptr;
        Ref<sVkPipelineObject>       mPipelineObject       = nullptr;
    };

} // namespace SE::Graphics