#include "ARenderContext.h"

#include "Core/Logging.h"

namespace LTSE::Graphics
{

    ARenderContext::ARenderContext( GraphicContext const &aGraphicContext, Ref<ARenderTarget> aRenderTarget )
        : mGraphicContext{ aGraphicContext }
        , mRenderTarget{ aRenderTarget }
    {
        auto lCommandBuffers = aGraphicContext.mContext->AllocateCommandBuffer( aRenderTarget->GetImageCount() );

        mCommandBufferObject.reserve( lCommandBuffers.size() );

        for( auto &lCB : lCommandBuffers )
            mCommandBufferObject.push_back( New<Internal::sVkCommandBufferObject>( aGraphicContext.mContext, lCB ) );

        for( size_t i = 0; i < aRenderTarget->GetImageCount(); i++ )
        {
            auto lImageAvailableSemaphore = aRenderTarget->GetImageAvailableSemaphore( i );
            if( lImageAvailableSemaphore )
                mCommandBufferObject[i]->AddWaitSemaphore( lImageAvailableSemaphore, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT );

            auto lRenderFinishedSemaphore = aRenderTarget->GetRenderFinishedSemaphore( i );
            if( lRenderFinishedSemaphore ) mCommandBufferObject[i]->AddSignalSemaphore( lRenderFinishedSemaphore );

            auto lSubmitFence = aRenderTarget->GetInFlightFence( i );
            if( lSubmitFence ) mCommandBufferObject[i]->SetSubmitFence( lSubmitFence );
        }
    }

    bool ARenderContext::BeginRender()
    {
        if( mFrameIsStarted ) return true;

        mFrameIsStarted = mRenderTarget->BeginRender();

        if( !mFrameIsStarted ) return false;

        mCurrentCommandBuffer = mRenderTarget->GetCurrentImage();

        float lWidth  = static_cast<float>( mRenderTarget->mSpec.mWidth );
        float lHeight = static_cast<float>( mRenderTarget->mSpec.mHeight );

        auto lCommandBuffer = mCommandBufferObject[mCurrentCommandBuffer];
        lCommandBuffer->Begin();
        lCommandBuffer->BeginRenderPass( mRenderTarget->GetRenderPass(), mRenderTarget->GetFramebuffer(), { lWidth, lHeight },
            mRenderTarget->GetRenderPass()->GetClearValues() );
        lCommandBuffer->SetViewport( { 0.0f, 0.0f }, { lWidth, lHeight } );
        lCommandBuffer->SetScissor( { 0.0f, 0.0f }, { lWidth, lHeight } );

        return true;
    }

    bool ARenderContext::EndRender()
    {
        if( !mFrameIsStarted ) return false;

        auto lCommandBuffer = mCommandBufferObject[mCurrentCommandBuffer];
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

    void ARenderContext::Draw( uint32_t a_VertexCount, uint32_t a_VertexOffset, uint32_t a_VertexBufferOffset,
        uint32_t a_InstanceCount, uint32_t a_FirstInstance )
    {
        auto lCommandBuffer = mCommandBufferObject[mCurrentCommandBuffer];

        if( mCurrentIndexBuffer != nullptr )
            lCommandBuffer->DrawIndexed( a_VertexCount, a_VertexOffset, a_VertexBufferOffset, a_InstanceCount, a_FirstInstance );
        else
            lCommandBuffer->Draw( a_VertexCount, a_VertexOffset, a_VertexBufferOffset, a_InstanceCount, a_FirstInstance );
    }

    void ARenderContext::Bind( Ref<GraphicsPipeline> a_GraphicPipeline )
    {
        auto lCommandBuffer = mCommandBufferObject[mCurrentCommandBuffer];

        lCommandBuffer->Bind( a_GraphicPipeline->GetVkPipelineObject(), VK_PIPELINE_BIND_POINT_GRAPHICS );
        mCurrentPipelineLayout = a_GraphicPipeline->GetVkPipelineLayoutObject();
    }

    void ARenderContext::Bind( Ref<Buffer> a_VertexBuffer, uint32_t a_BindPoint )
    {
        auto lCommandBuffer = mCommandBufferObject[mCurrentCommandBuffer];

        lCommandBuffer->Bind( a_VertexBuffer->mVkObject, a_BindPoint );
        mCurrentVertexBuffer = a_VertexBuffer;
    }

    void ARenderContext::Bind( Ref<Buffer> a_VertexBuffer, Ref<Buffer> a_IndexBuffer, uint32_t a_BindPoint )
    {
        auto lCommandBuffer = mCommandBufferObject[mCurrentCommandBuffer];

        lCommandBuffer->Bind( a_VertexBuffer->mVkObject, a_IndexBuffer->mVkObject, a_BindPoint );
        mCurrentVertexBuffer = a_VertexBuffer;
        mCurrentIndexBuffer  = a_IndexBuffer;
    }

    void ARenderContext::Bind( Ref<DescriptorSet> a_DescriptorSet, uint32_t a_SetIndex, int32_t a_DynamicOffset )
    {
        auto lCommandBuffer = mCommandBufferObject[mCurrentCommandBuffer];

        lCommandBuffer->Bind( a_DescriptorSet->GetVkDescriptorSetObject(), VK_PIPELINE_BIND_POINT_GRAPHICS, mCurrentPipelineLayout,
            a_SetIndex, a_DynamicOffset );
    }

} // namespace LTSE::Graphics