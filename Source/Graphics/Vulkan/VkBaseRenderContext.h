#pragma once

#include "Graphics/Interface/IGraphicContext.h"
#include "Graphics/Interface/IRenderContext.h"

#include "Graphics/Vulkan/VkGpuBuffer.h"
#include "VkRenderTarget.h"

namespace SE::Graphics
{
    class VkBaseRenderContext : public IRenderContext
    {
      public:
        VkBaseRenderContext() = default;
        VkBaseRenderContext( ref_t<IGraphicContext> aGraphicContext, ref_t<IRenderTarget> aRenderTarget );

        ~VkBaseRenderContext() = default;

        ref_t<IGraphicContext> GetGraphicContext()
        {
            return mGraphicContext;
        };

        uint32_t           GetOutputImageCount();
        ref_t<IRenderTarget> GetRenderTarget()
        {
            return mRenderTarget;
        }
        ref_t<IRenderPass> GetRenderPass()
        {
            return Cast<VkRenderTarget>( mRenderTarget )->GetRenderPass();
        }

        bool BeginRender();
        bool EndRender();

        void Bind( ref_t<IGraphicsPipeline> aGraphicPipeline );
        void Bind( void *aDescriptorSet, uint32_t aSetIndex, int32_t aDynamicOffset = -1 );
        void Bind( ref_t<IDescriptorSet> aDescriptorSet, uint32_t aSetIndex, int32_t aDynamicOffset = -1 );
        void Bind( ref_t<IGraphicBuffer> aVertexBuffer, uint32_t aBindPoint = 0 );
        void Bind( ref_t<IGraphicBuffer> aVertexBuffer, ref_t<IGraphicBuffer> aIndexBuffer, uint32_t aBindPoint = 0 );
        void PushConstants( ShaderStageType aShaderStages, uint32_t aOffset, void *aValue, uint32_t aSize );
        void SetViewport( math::ivec2 aOffset, math::uvec2 aSize );
        void SetScissor( math::ivec2 aOffset, math::uvec2 aSize );

        void InternalDrawIndexed( uint32_t aVertexCount, uint32_t aVertexOffset, uint32_t aVertexBufferOffset = 0,
                                  uint32_t aInstanceCount = 0, uint32_t aFirstInstance = 0 );

        void InternalDrawNonIndexed( uint32_t aVertexCount, uint32_t aVertexOffset, uint32_t aVertexBufferOffset = 0,
                                     uint32_t aInstanceCount = 0, uint32_t aFirstInstance = 0 );

      protected:
        uint32_t                                 mCurrentCommandBuffer  = 0;
        vec_t<ref_t<sVkCommandBufferObject>> mCommandBufferObject   = {};
        ref_t<sVkPipelineLayoutObject>             mCurrentPipelineLayout = nullptr;

        ref_t<sVkCommandBufferObject> GetCurrentCommandBuffer()
        {
            return Cast<sVkCommandBufferObject>( mRenderTarget->GetCommandBuffer() );
        }
    };

} // namespace SE::Graphics