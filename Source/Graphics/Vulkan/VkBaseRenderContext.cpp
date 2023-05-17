#include "VkBaseRenderContext.h"
#include "VkDescriptorSet.h"
#include "VkGraphicsPipeline.h"

#include "Core/Logging.h"

namespace SE::Graphics
{

    VkBaseRenderContext::VkBaseRenderContext( Ref<IGraphicContext> aGraphicContext, Ref<IRenderTarget> aRenderTarget )
        : IRenderContext( aGraphicContext, aRenderTarget )
    {
    }

    bool VkBaseRenderContext::BeginRender()
    {
        if( mFrameIsStarted ) return true;

        IRenderContext::BeginRender();

        return mFrameIsStarted;
    }

    bool VkBaseRenderContext::EndRender()
    {
        if( !mFrameIsStarted ) return false;

        IRenderContext::EndRender();

        auto lCommandBuffer = GetCurrentCommandBuffer();
        lCommandBuffer->EndRenderPass();
        lCommandBuffer->End();
        lCommandBuffer->SubmitTo( Cast<VkGraphicContext>( mGraphicContext )->GetGraphicsQueue() );

        mRenderTarget->EndRender();

        return true;
    }

    void VkBaseRenderContext::InternalDrawIndexed( uint32_t aVertexCount, uint32_t aVertexOffset, uint32_t aVertexBufferOffset,
                                                   uint32_t aInstanceCount, uint32_t aFirstInstance )
    {
        auto lCommandBuffer = GetCurrentCommandBuffer();

        lCommandBuffer->DrawIndexed( aVertexCount, aVertexOffset, aVertexBufferOffset, aInstanceCount, aFirstInstance );
    }

    void VkBaseRenderContext::InternalDrawNonIndexed( uint32_t aVertexCount, uint32_t aVertexOffset, uint32_t aVertexBufferOffset,
                                                      uint32_t aInstanceCount, uint32_t aFirstInstance )
    {
        auto lCommandBuffer = GetCurrentCommandBuffer();

        lCommandBuffer->Draw( aVertexCount, aVertexOffset, aVertexBufferOffset, aInstanceCount, aFirstInstance );
    }

    void VkBaseRenderContext::Bind( Ref<IGraphicsPipeline> aGraphicPipeline )
    {
        IRenderContext::Bind( aGraphicPipeline );

        auto lCommandBuffer = GetCurrentCommandBuffer();

        lCommandBuffer->Bind( Cast<VkGraphicsPipeline>( aGraphicPipeline )->GetVkPipelineObject(), VK_PIPELINE_BIND_POINT_GRAPHICS );
        mCurrentPipelineLayout = Cast<VkGraphicsPipeline>( aGraphicPipeline )->GetVkPipelineLayoutObject();
    }

    void VkBaseRenderContext::Bind( Ref<IGraphicBuffer> aVertexBuffer, uint32_t aBindPoint )
    {
        IRenderContext::Bind( aVertexBuffer, aBindPoint );

        auto lCommandBuffer = GetCurrentCommandBuffer();

        lCommandBuffer->Bind( Cast<VkGpuBuffer>( aVertexBuffer )->mVkBuffer, aBindPoint );
    }

    void VkBaseRenderContext::Bind( Ref<IGraphicBuffer> aVertexBuffer, Ref<IGraphicBuffer> aIndexBuffer, uint32_t aBindPoint )
    {
        IRenderContext::Bind( aVertexBuffer, aIndexBuffer, aBindPoint );

        auto lCommandBuffer = GetCurrentCommandBuffer();

        lCommandBuffer->Bind( Cast<VkGpuBuffer>( aVertexBuffer )->mVkBuffer, Cast<VkGpuBuffer>( aIndexBuffer )->mVkBuffer,
                              aBindPoint );
    }

    void VkBaseRenderContext::Bind( Ref<IDescriptorSet> aDescriptorSet, uint32_t aSetIndex, int32_t aDynamicOffset )
    {
        auto lCommandBuffer = GetCurrentCommandBuffer();

        lCommandBuffer->Bind( Cast<VkDescriptorSetObject>( aDescriptorSet )->GetVkDescriptorSetObject(),
                              VK_PIPELINE_BIND_POINT_GRAPHICS, mCurrentPipelineLayout, aSetIndex, aDynamicOffset );
    }

    void VkBaseRenderContext::Bind( void *aDescriptorSet, uint32_t aSetIndex, int32_t aDynamicOffset )
    {
        auto lCommandBuffer = GetCurrentCommandBuffer();

        lCommandBuffer->Bind( aDescriptorSet, VK_PIPELINE_BIND_POINT_GRAPHICS, mCurrentPipelineLayout, aSetIndex, aDynamicOffset );
    }

    void VkBaseRenderContext::SetViewport( math::ivec2 aOffset, math::uvec2 aSize )
    {
        GetCurrentCommandBuffer()->SetViewport( aOffset, aSize );
    }

    void VkBaseRenderContext::SetScissor( math::ivec2 aOffset, math::uvec2 aSize )
    {
        GetCurrentCommandBuffer()->SetScissor( aOffset, aSize );
    }

    void VkBaseRenderContext::PushConstants( ShaderStageType aShaderStages, uint32_t aOffset, void *aValue, uint32_t aSize )
    {
        auto lCommandBuffer = GetCurrentCommandBuffer();
        lCommandBuffer->PushConstants( (VkShaderStageFlags)aShaderStages, aOffset, aValue, aSize, mCurrentPipelineLayout );
    }

} // namespace SE::Graphics