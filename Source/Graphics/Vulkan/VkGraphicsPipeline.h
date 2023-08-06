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
        VkGraphicsPipeline( ref_t<VkGraphicContext> aGraphicContext, ref_t<VkRenderContext> aRenderContext,
                            ePrimitiveTopology aTopology );
        ~VkGraphicsPipeline() = default;

        ref_t<sVkPipelineObject> GetVkPipelineObject()
        {
            return mPipelineObject;
        }
        
        ref_t<sVkPipelineLayoutObject> GetVkPipelineLayoutObject()
        {
            return mPipelineLayoutObject;
        }

        void Build();

      private:
        vector_t<ref_t<sVkDescriptorSetLayoutObject>> mDescriptorSetLayouts{};

        ref_t<sVkPipelineLayoutObject> mPipelineLayoutObject = nullptr;
        ref_t<sVkPipelineObject>       mPipelineObject       = nullptr;
        vector_t<sShader>              mShaders{};
    };

} // namespace SE::Graphics