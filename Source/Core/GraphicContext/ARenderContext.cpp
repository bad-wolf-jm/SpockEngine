#include "ARenderContext.h"

#include "Core/Logging.h"

namespace SE::Graphics
{

    ARenderContext::ARenderContext( GraphicContext const &aGraphicContext, Ref<ARenderTarget> aRenderTarget )
        : mGraphicContext{ aGraphicContext }
        , mRenderTarget{ aRenderTarget }
    {
    }

    bool ARenderContext::BeginRender()
    {
        if( mFrameIsStarted ) return true;

        mFrameIsStarted = mRenderTarget->BeginRender();

        if( !mFrameIsStarted ) return false;

        float lWidth  = static_cast<float>( mRenderTarget->mSpec.mWidth );
        float lHeight = static_cast<float>( mRenderTarget->mSpec.mHeight );

        auto lCommandBuffer = GetCurrentCommandBuffer();
        lCommandBuffer->Begin();
        lCommandBuffer->BeginRenderPass( mRenderTarget->GetRenderPass(), mRenderTarget->GetFramebuffer()->mVkFramebuffer, { lWidth, lHeight },
                                         mRenderTarget->GetClearValues() );
        lCommandBuffer->SetViewport( { 0.0f, 0.0f }, { lWidth, lHeight } );
        lCommandBuffer->SetScissor( { 0.0f, 0.0f }, { lWidth, lHeight } );

        return true;
    }

    bool ARenderContext::EndRender()
    {
        if( !mFrameIsStarted ) return false;

        auto lCommandBuffer = GetCurrentCommandBuffer();
        lCommandBuffer->EndRenderPass();
        lCommandBuffer->End();
        lCommandBuffer->SubmitTo( mGraphicContext.mContext->GetGraphicsQueue() );

        mRenderTarget->EndRender();

        mFrameIsStarted        = false;
        mCurrentPipelineLayout = nullptr;
        mHasIndex              = false;

        return true;
    }

    void ARenderContext::Present() { mRenderTarget->Present(); }

    void ARenderContext::ResetBuffers()
    {
        mHasIndex = false;
    }

    void ARenderContext::Draw( uint32_t aVertexCount, uint32_t aVertexOffset, uint32_t aVertexBufferOffset, uint32_t a_InstanceCount,
                               uint32_t a_FirstInstance )
    {
        auto lCommandBuffer = GetCurrentCommandBuffer();

        if( mHasIndex )
            lCommandBuffer->DrawIndexed( aVertexCount, aVertexOffset, aVertexBufferOffset, a_InstanceCount, a_FirstInstance );
        else
            lCommandBuffer->Draw( aVertexCount, aVertexOffset, aVertexBufferOffset, a_InstanceCount, a_FirstInstance );
    }

    void ARenderContext::Bind( Ref<GraphicsPipeline> aGraphicPipeline )
    {
        auto lCommandBuffer = GetCurrentCommandBuffer();

        lCommandBuffer->Bind( aGraphicPipeline->GetVkPipelineObject(), VK_PIPELINE_BIND_POINT_GRAPHICS );
        mCurrentPipelineLayout = aGraphicPipeline->GetVkPipelineLayoutObject();
    }

    void ARenderContext::Bind( Ref<VkGpuBuffer> aVertexBuffer, uint32_t aBindPoint )
    {
        auto lCommandBuffer = GetCurrentCommandBuffer();

        lCommandBuffer->Bind( aVertexBuffer->mVkBuffer, aBindPoint );
        mHasIndex = false;
    }

    void ARenderContext::Bind( Ref<VkGpuBuffer> aVertexBuffer, Ref<VkGpuBuffer> aIndexBuffer, uint32_t aBindPoint )
    {
        auto lCommandBuffer = GetCurrentCommandBuffer();

        lCommandBuffer->Bind( aVertexBuffer->mVkBuffer, aIndexBuffer->mVkBuffer, aBindPoint );
        mHasIndex = true;
    }

    void ARenderContext::Bind( Ref<DescriptorSet> aDescriptorSet, uint32_t aSetIndex, int32_t aDynamicOffset )
    {
        auto lCommandBuffer = GetCurrentCommandBuffer();

        lCommandBuffer->Bind( aDescriptorSet->GetVkDescriptorSetObject(), VK_PIPELINE_BIND_POINT_GRAPHICS, mCurrentPipelineLayout,
                              aSetIndex, aDynamicOffset );
    }

} // namespace SE::Graphics