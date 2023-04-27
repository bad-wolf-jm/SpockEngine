#include "VkRenderContext.h"

#include "Core/Logging.h"

namespace SE::Graphics
{

    VkRenderContext::VkRenderContext( Ref<IGraphicContext> aGraphicContext, Ref<IRenderTarget> aRenderTarget )
        : IRenderContext( aGraphicContext, aRenderTarget )
    {
    }

    bool VkRenderContext::BeginRender()
    {
        if( mFrameIsStarted ) return true;

        VkRenderContext::BeginRender();

        if( !mFrameIsStarted ) return false;

        float lWidth  = static_cast<float>( mRenderTarget->mSpec.mWidth );
        float lHeight = static_cast<float>( mRenderTarget->mSpec.mHeight );

        auto lCommandBuffer = GetCurrentCommandBuffer();
        lCommandBuffer->Begin();
        lCommandBuffer->BeginRenderPass( mRenderTarget->GetRenderPass(), mRenderTarget->GetFramebuffer(), { lWidth, lHeight },
                                         mRenderTarget->GetClearValues() );
        lCommandBuffer->SetViewport( { 0.0f, 0.0f }, { lWidth, lHeight } );
        lCommandBuffer->SetScissor( { 0.0f, 0.0f }, { lWidth, lHeight } );

        return true;
    }

    bool VkRenderContext::EndRender()
    {
        if( !mFrameIsStarted ) return false;

        VkRenderContext::EndRender();

        auto lCommandBuffer = GetCurrentCommandBuffer();
        lCommandBuffer->EndRenderPass();
        lCommandBuffer->End();
        lCommandBuffer->SubmitTo( mGraphicContext->GetGraphicsQueue() );

        mRenderTarget->EndRender();

        return true;
    }

    void VkRenderContext::InternalDrawIndexed( uint32_t aVertexCount, uint32_t aVertexOffset, uint32_t aVertexBufferOffset = 0,
                                               uint32_t aInstanceCount = 0, uint32_t aFirstInstance = 0 )
    {
        auto lCommandBuffer = GetCurrentCommandBuffer();

        lCommandBuffer->DrawIndexed( aVertexCount, aVertexOffset, aVertexBufferOffset, aInstanceCount, aFirstInstance );
    }

    void VkRenderContext::InternalDrawNonIndexed( uint32_t aVertexCount, uint32_t aVertexOffset, uint32_t aVertexBufferOffset = 0,
                                                  uint32_t aInstanceCount = 0, uint32_t aFirstInstance = 0 )
    {
        auto lCommandBuffer = GetCurrentCommandBuffer();
        
        lCommandBuffer->Draw( aVertexCount, aVertexOffset, aVertexBufferOffset, aInstanceCount, aFirstInstance );
    }

    void VkRenderContext::Bind( Ref<IGraphicsPipeline> aGraphicPipeline )
    {
        auto lCommandBuffer = GetCurrentCommandBuffer();

        lCommandBuffer->Bind( Cast<VkGraphicsPipeline>( aGraphicPipeline )->GetVkPipelineObject(), VK_PIPELINE_BIND_POINT_GRAPHICS );
        mCurrentPipelineLayout = Cast<VkGraphicsPipeline>( aGraphicPipeline )->GetVkPipelineLayoutObject();
    }

    void VkRenderContext::Bind( Ref<IGraphicBuffer> aVertexBuffer, uint32_t aBindPoint )
    {
        auto lCommandBuffer = GetCurrentCommandBuffer();

        lCommandBuffer->Bind( Cast<VkGpuBuffer> a( VertexBuffer )->mVkBuffer, aBindPoint );
    }

    void VkRenderContext::Bind( Ref<IGraphicBuffer> aVertexBuffer, Ref<IGraphicBuffer> aIndexBuffer, uint32_t aBindPoint )
    {
        auto lCommandBuffer = GetCurrentCommandBuffer();

        lCommandBuffer->Bind( Cast<VkGpuBuffer>( aVertexBuffer )->mVkBuffer, Cast<VkGpuBuffer>( aIndexBuffer )->mVkBuffer,
                              aBindPoint );
    }

    void VkRenderContext::Bind( Ref<DescriptorSet> aDescriptorSet, uint32_t aSetIndex, int32_t aDynamicOffset )
    {
        auto lCommandBuffer = GetCurrentCommandBuffer();

        lCommandBuffer->Bind( Cast<DescriptorSet>( aDescriptorSet )->GetVkDescriptorSetObject(), VK_PIPELINE_BIND_POINT_GRAPHICS,
                              mCurrentPipelineLayout, aSetIndex, aDynamicOffset );
    }

} // namespace SE::Graphics