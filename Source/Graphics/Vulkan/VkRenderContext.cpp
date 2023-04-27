#include "VkRenderContext.h"
#include "VkGraphicsPipeline.h"

#include "Core/Logging.h"

namespace SE::Graphics
{

    VkRenderContext::VkRenderContext( Ref<VkGraphicContext> aGraphicContext, Ref<VkRenderTarget> aRenderTarget )
        : IRenderContext( aGraphicContext, aRenderTarget )
    {
    }

    bool VkRenderContext::BeginRender()
    {
        if( mFrameIsStarted ) return true;

        IRenderContext::BeginRender();

        if( !mFrameIsStarted ) return false;

        float lWidth  = static_cast<float>( Cast<VkRenderTarget>( mRenderTarget )->mSpec.mWidth );
        float lHeight = static_cast<float>( Cast<VkRenderTarget>( mRenderTarget )->mSpec.mHeight );

        auto lCommandBuffer = GetCurrentCommandBuffer();
        lCommandBuffer->Begin();
        lCommandBuffer->BeginRenderPass( Cast<VkRenderTarget>( mRenderTarget )->GetRenderPass(),
                                         Cast<VkRenderTarget>( mRenderTarget )->GetFramebuffer(), { lWidth, lHeight },
                                         Cast<VkRenderTarget>( mRenderTarget )->GetClearValues() );
        lCommandBuffer->SetViewport( { 0.0f, 0.0f }, { lWidth, lHeight } );
        lCommandBuffer->SetScissor( { 0.0f, 0.0f }, { lWidth, lHeight } );

        return true;
    }

    bool VkRenderContext::EndRender()
    {
        if( !mFrameIsStarted ) return false;

        IRenderContext::EndRender();

        auto lCommandBuffer = GetCurrentCommandBuffer();
        lCommandBuffer->EndRenderPass();
        lCommandBuffer->End();
        lCommandBuffer->SubmitTo( Cast<VkGraphicContext>( mGraphicContext )->GetGraphicsQueue() );

        Cast<VkRenderTarget>( mRenderTarget )->EndRender();

        return true;
    }

    void VkRenderContext::InternalDrawIndexed( uint32_t aVertexCount, uint32_t aVertexOffset, uint32_t aVertexBufferOffset,
                                               uint32_t aInstanceCount, uint32_t aFirstInstance )
    {
        auto lCommandBuffer = GetCurrentCommandBuffer();

        lCommandBuffer->DrawIndexed( aVertexCount, aVertexOffset, aVertexBufferOffset, aInstanceCount, aFirstInstance );
    }

    void VkRenderContext::InternalDrawNonIndexed( uint32_t aVertexCount, uint32_t aVertexOffset, uint32_t aVertexBufferOffset,
                                                  uint32_t aInstanceCount, uint32_t aFirstInstance )
    {
        auto lCommandBuffer = GetCurrentCommandBuffer();

        lCommandBuffer->Draw( aVertexCount, aVertexOffset, aVertexBufferOffset, aInstanceCount, aFirstInstance );
    }

    void VkRenderContext::Bind( Ref<IGraphicsPipeline> aGraphicPipeline )
    {
        IRenderContext::Bind(aGraphicPipeline);

        auto lCommandBuffer = GetCurrentCommandBuffer();

        lCommandBuffer->Bind( Cast<VkGraphicsPipeline>( aGraphicPipeline )->GetVkPipelineObject(), VK_PIPELINE_BIND_POINT_GRAPHICS );
        mCurrentPipelineLayout = Cast<VkGraphicsPipeline>( aGraphicPipeline )->GetVkPipelineLayoutObject();
    }

    void VkRenderContext::Bind( Ref<IGraphicBuffer> aVertexBuffer, uint32_t aBindPoint )
    {
        IRenderContext::Bind(aVertexBuffer, aBindPoint);

        auto lCommandBuffer = GetCurrentCommandBuffer();

        lCommandBuffer->Bind( Cast<VkGpuBuffer>( aVertexBuffer )->mVkBuffer, aBindPoint );
    }

    void VkRenderContext::Bind( Ref<IGraphicBuffer> aVertexBuffer, Ref<IGraphicBuffer> aIndexBuffer, uint32_t aBindPoint )
    {
        IRenderContext::Bind(aVertexBuffer, aIndexBuffer, aBindPoint);

        auto lCommandBuffer = GetCurrentCommandBuffer();

        lCommandBuffer->Bind( Cast<VkGpuBuffer>( aVertexBuffer )->mVkBuffer, Cast<VkGpuBuffer>( aIndexBuffer )->mVkBuffer,
                              aBindPoint );
    }

    void VkRenderContext::Bind( Ref<IDescriptorSet> aDescriptorSet, uint32_t aSetIndex, int32_t aDynamicOffset )
    {
        auto lCommandBuffer = GetCurrentCommandBuffer();

        lCommandBuffer->Bind( Cast<DescriptorSet>( aDescriptorSet )->GetVkDescriptorSetObject(), VK_PIPELINE_BIND_POINT_GRAPHICS,
                              mCurrentPipelineLayout, aSetIndex, aDynamicOffset );
    }

    void VkRenderContext::Bind( void *aDescriptorSet, uint32_t aSetIndex, int32_t aDynamicOffset )
    {
        auto lCommandBuffer = GetCurrentCommandBuffer();

        lCommandBuffer->Bind( aDescriptorSet, VK_PIPELINE_BIND_POINT_GRAPHICS, mCurrentPipelineLayout, aSetIndex, aDynamicOffset );
    }

    void VkRenderContext::SetViewport( math::ivec2 aOffset, math::uvec2 aSize )
    {
        GetCurrentCommandBuffer()->SetViewport( aOffset, aSize );
    }

    void VkRenderContext::SetScissor( math::ivec2 aOffset, math::uvec2 aSize )
    {
        GetCurrentCommandBuffer()->SetScissor( aOffset, aSize );
    }

    void VkRenderContext::PushConstants( ShaderStageType aShaderStages, uint32_t aOffset, void *aValue, uint32_t aSize ) 
    {
        auto lCommandBuffer = GetCurrentCommandBuffer();
        lCommandBuffer->PushConstants( (VkShaderStageFlags)aShaderStages, aOffset, aValue, aSize, mCurrentPipelineLayout );        
    }

} // namespace SE::Graphics