#include "ARenderContext.h"

#include "Core/Logging.h"

namespace LTSE::Graphics
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
        lCommandBuffer->BeginRenderPass(
            mRenderTarget->GetRenderPass(), mRenderTarget->GetFramebuffer(), { lWidth, lHeight }, mRenderTarget->GetClearValues() );
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
        mCurrentVertexBuffer   = nullptr;
        mCurrentIndexBuffer    = nullptr;

        return true;
    }

    void ARenderContext::Present() { mRenderTarget->Present(); }

    void ARenderContext::ResetBuffers()
    {
        mCurrentVertexBuffer = nullptr;
        mCurrentIndexBuffer  = nullptr;
    }

    void ARenderContext::Draw( uint32_t aVertexCount, uint32_t aVertexOffset, uint32_t aVertexBufferOffset, uint32_t a_InstanceCount,
        uint32_t a_FirstInstance )
    {
        auto lCommandBuffer = GetCurrentCommandBuffer();

        if( mCurrentIndexBuffer != nullptr )
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

    void ARenderContext::Bind( Ref<Buffer> aVertexBuffer, uint32_t aBindPoint )
    {
        auto lCommandBuffer = GetCurrentCommandBuffer();

        lCommandBuffer->Bind( aVertexBuffer->mVkObject, aBindPoint );
        mCurrentVertexBuffer = aVertexBuffer;
    }

    void ARenderContext::Bind( Ref<Buffer> aVertexBuffer, Ref<Buffer> aIndexBuffer, uint32_t aBindPoint )
    {
        auto lCommandBuffer = GetCurrentCommandBuffer();

        lCommandBuffer->Bind( aVertexBuffer->mVkObject, aIndexBuffer->mVkObject, aBindPoint );
        mCurrentVertexBuffer = aVertexBuffer;
        mCurrentIndexBuffer  = aIndexBuffer;
    }

    void ARenderContext::Bind( Ref<DescriptorSet> aDescriptorSet, uint32_t aSetIndex, int32_t aDynamicOffset )
    {
        auto lCommandBuffer = GetCurrentCommandBuffer();

        lCommandBuffer->Bind( aDescriptorSet->GetVkDescriptorSetObject(), VK_PIPELINE_BIND_POINT_GRAPHICS, mCurrentPipelineLayout,
            aSetIndex, aDynamicOffset );
    }

} // namespace LTSE::Graphics