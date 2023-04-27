#pragma once

#include "Graphics/Interface/IRenderContext.h"

#include "Graphics/Vulkan/IGraphicContext.h"

#include "Graphics/Vulkan/VkGpuBuffer.h"
#include "GraphicsPipeline.h"
#include "VkRenderTarget.h"

namespace SE::Graphics
{
    class VkRenderContext : IRenderContext
    {
      public:
        VkRenderContext() = default;
        VkRenderContext( Ref<IGraphicContext> aGraphicContext, Ref<IRenderTarget> aRenderTarget );

        ~VkRenderContext() = default;

        Ref<IGraphicContext> GetGraphicContext() { return mGraphicContext; };

        uint32_t           GetOutputImageCount();
        Ref<IRenderTarget> GetRenderTarget() { return mRenderTarget; }

        virtual Ref<sVkAbstractRenderPassObject> GetRenderPass() { return mRenderTarget->GetRenderPass(); }

        bool BeginRender();
        bool EndRender();

        void Bind( Ref<IGraphicsPipeline> aGraphicPipeline );
        void Bind( Ref<IDescriptorSet> aDescriptorSet, uint32_t aSetIndex, int32_t aDynamicOffset = -1 );
        void Bind( Ref<IGraphicBuffer> aVertexBuffer, uint32_t aBindPoint = 0 );
        void Bind( Ref<IGraphicBuffer> aVertexBuffer, Ref<IGraphicBuffer> aIndexBuffer, uint32_t aBindPoint = 0 );
        void PushConstants( ShaderStageType aShaderStages, uint32_t aOffset, void *aValue, uint32_t aSize );

        void InternalDrawIndexed( uint32_t aVertexCount, uint32_t aVertexOffset, uint32_t aVertexBufferOffset = 0,
                                  uint32_t aInstanceCount = 0, uint32_t aFirstInstance = 0 );

        void InternalDrawNonIndexed( uint32_t aVertexCount, uint32_t aVertexOffset, uint32_t aVertexBufferOffset = 0,
                                     uint32_t aInstanceCount = 0, uint32_t aFirstInstance = 0 );

      private:
        uint32_t                                 mCurrentCommandBuffer  = 0;
        std::vector<Ref<sVkCommandBufferObject>> mCommandBufferObject   = {};
        Ref<sVkPipelineLayoutObject>             mCurrentPipelineLayout = nullptr;
        Ref<sVkCommandBufferObject>              GetCurrentCommandBuffer() { return mRenderTarget->GetCurrentCommandBuffer(); }
    };

} // namespace SE::Graphics