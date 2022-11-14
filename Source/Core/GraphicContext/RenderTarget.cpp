#include "RenderTarget.h"
#include "Core/Logging.h"

namespace LTSE::Graphics
{
    AbstractRenderTarget::AbstractRenderTarget( GraphicContext &a_GraphicContext )
        : mGraphicContext{ a_GraphicContext }
    {
    }

    void AbstractRenderTarget::Initialize( RenderTargetDescription &aSpec )
    {
        Spec = aSpec;

        mRenderPassObject = New<sVkRenderPassObject>(
            mGraphicContext.mContext, ToVkFormat( aSpec.Format ), Spec.SampleCount, Spec.Sampled, Spec.Presented, Spec.ClearColor );

        if( Spec.SampleCount > 1 )
        {
            mMSAAOutputTexture = New<sVkFramebufferImage>( mGraphicContext.mContext, ToVkFormat( aSpec.Format ), Spec.Width,
                Spec.Height, Spec.SampleCount, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, false );
        }

        if( !Spec.OutputTexture )
        {
            mOutputTexture = New<sVkFramebufferImage>( mGraphicContext.mContext, ToVkFormat( aSpec.Format ), Spec.Width, Spec.Height,
                1, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, true );
        }
        else
        {
            mOutputTexture = Spec.OutputTexture;
        }

        mDepthTexture = New<sVkFramebufferImage>( mGraphicContext.mContext, mGraphicContext.mContext->GetDepthFormat(), Spec.Width,
            Spec.Height, Spec.SampleCount, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, false );

        std::vector<Ref<sVkFramebufferImage>> l_AttachmentViews{};

        if( Spec.SampleCount == 1 )
            l_AttachmentViews = { mOutputTexture, mDepthTexture };
        else
            l_AttachmentViews = { mMSAAOutputTexture, mOutputTexture, mDepthTexture };

        mFramebufferObject = New<sVkFramebufferObject>(
            mGraphicContext.mContext, Spec.Width, Spec.Height, 1, mRenderPassObject->mVkObject, l_AttachmentViews );
    }

    void AbstractRenderTarget::InitializeCommandBuffers()
    {
        auto lCommandBuffers = mGraphicContext.mContext->AllocateCommandBuffer( GetImageCount() );

        mCommandBufferObject = {};

        for( auto &lCB : lCommandBuffers )
            mCommandBufferObject.push_back( New<sVkCommandBufferObject>( mGraphicContext.mContext, lCB ) );

        for( size_t i = 0; i < GetImageCount(); i++ )
        {
            auto lImageAvailableSemaphore = GetImageAvailableSemaphore( i );
            if( lImageAvailableSemaphore )
                mCommandBufferObject[i]->AddWaitSemaphore( lImageAvailableSemaphore, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT );

            auto lRenderFinishedSemaphore = GetRenderFinishedSemaphore( i );
            if( lRenderFinishedSemaphore ) mCommandBufferObject[i]->AddSignalSemaphore( lRenderFinishedSemaphore );

            auto lSubmitFence = GetInFlightFence( i );
            if( lSubmitFence ) mCommandBufferObject[i]->SetSubmitFence( lSubmitFence );
        }
    }

    bool     AbstractRenderTarget::BeginRender() { return true; }
    void     AbstractRenderTarget::EndRender() {}
    void     AbstractRenderTarget::Present() {}
    uint32_t AbstractRenderTarget::GetCurrentImage() { return 0; };

    Ref<sVkFramebufferObject>   AbstractRenderTarget::GetFramebuffer() { return nullptr; }
    Ref<sVkCommandBufferObject> AbstractRenderTarget::GetCommandBuffer( uint32_t i ) { return mCommandBufferObject[i]; }
    VkSemaphore                 AbstractRenderTarget::GetImageAvailableSemaphore( uint32_t i ) { return VK_NULL_HANDLE; }
    VkSemaphore                 AbstractRenderTarget::GetRenderFinishedSemaphore( uint32_t i ) { return VK_NULL_HANDLE; }
    VkFence                     AbstractRenderTarget::GetInFlightFence( uint32_t i ) { return VK_NULL_HANDLE; }

    SwapChainRenderTargetImage::SwapChainRenderTargetImage( GraphicContext &a_GraphicContext, RenderTargetDescription &aSpec )
        : AbstractRenderTarget( a_GraphicContext )
    {
        mImageCount = 1;
        Initialize( aSpec );
    }

    bool                        SwapChainRenderTargetImage::BeginRender() { return true; }
    void                        SwapChainRenderTargetImage::EndRender() {}
    void                        SwapChainRenderTargetImage::Present() {}
    uint32_t                    SwapChainRenderTargetImage::GetCurrentImage() { return 0; };
    Ref<sVkFramebufferObject>   SwapChainRenderTargetImage::GetFramebuffer() { return mFramebufferObject; }
    Ref<sVkCommandBufferObject> SwapChainRenderTargetImage::GetCommandBuffer( uint32_t i ) { return nullptr; }
    VkSemaphore                 SwapChainRenderTargetImage::GetImageAvailableSemaphore( uint32_t i )
    {
        return AbstractRenderTarget::GetImageAvailableSemaphore( i );
    }
    VkSemaphore SwapChainRenderTargetImage::GetRenderFinishedSemaphore( uint32_t i )
    {
        return AbstractRenderTarget::GetRenderFinishedSemaphore( i );
    }
    VkFence SwapChainRenderTargetImage::GetInFlightFence( uint32_t i ) { return AbstractRenderTarget::GetInFlightFence( i ); }

    SwapChainRenderTarget::SwapChainRenderTarget( GraphicContext &a_GraphicContext, SwapChainRenderTargetDescription &aSpec )
        : AbstractRenderTarget( a_GraphicContext )
    {
        Spec.SampleCount = aSpec.SampleCount;

        RecreateSwapChain();
    }

    bool SwapChainRenderTarget::BeginRender()
    {
        mGraphicContext.mContext->WaitForFence( mInFlightFences[mCurrentImage] );

        uint64_t l_Timeout = std::numeric_limits<uint64_t>::max();
        VkResult result =
            mGraphicContext.mContext->AcquireNextImage( mVkObject, l_Timeout, mImageAvailableSemaphores[mCurrentImage], &mCurrentImage );

        if( result == VK_ERROR_OUT_OF_DATE_KHR )
        {
            mFrameIsStarted = false;
            RecreateSwapChain();
            return mFrameIsStarted;
        }
        else
        {
            VK_CHECK_RESULT( result );
            mFrameIsStarted = true;
            mRenderTargets[mCurrentImage]->BeginRender();
            return mFrameIsStarted;
        }

        return mFrameIsStarted;
    }

    void SwapChainRenderTarget::EndRender()
    {
        mRenderTargets[mCurrentImage]->EndRender();
        mFrameIsStarted = false;
    }

    void SwapChainRenderTarget::Present()
    {

        VkResult l_PresentResult =
            mGraphicContext.mContext->Present( mVkObject, mCurrentImage, mRenderFinishedSemaphores[mCurrentImage] );

        if( ( l_PresentResult == VK_ERROR_OUT_OF_DATE_KHR ) || ( l_PresentResult == VK_SUBOPTIMAL_KHR ) ||
            mGraphicContext.m_ViewportClient->WindowWasResized() )
        {
            mGraphicContext.m_ViewportClient->ResetWindowResizedFlag();
            RecreateSwapChain();
        }
        else if( l_PresentResult != VK_SUCCESS )
            throw std::runtime_error( "failed to present swap chain image!" );
    }

    uint32_t SwapChainRenderTarget::GetCurrentImage() { return mCurrentImage; };

    Ref<sVkFramebufferObject>   SwapChainRenderTarget::GetFramebuffer() { return mRenderTargets[mCurrentImage]->GetFramebuffer(); }
    Ref<sVkCommandBufferObject> SwapChainRenderTarget::GetCommandBuffer( uint32_t i )
    {
        return AbstractRenderTarget::GetCommandBuffer( i );
    }
    VkSemaphore SwapChainRenderTarget::GetImageAvailableSemaphore( uint32_t i ) { return mImageAvailableSemaphores[i]; }
    VkSemaphore SwapChainRenderTarget::GetRenderFinishedSemaphore( uint32_t i ) { return mRenderFinishedSemaphores[i]; }
    VkFence     SwapChainRenderTarget::GetInFlightFence( uint32_t i ) { return mInFlightFences[i]; }

    void SwapChainRenderTarget::RecreateSwapChain()
    {
        mGraphicContext.mContext->WaitIdle();

        auto [lSwapChainImageFormat, lSwapChainImageCount, lSwapchainExtent, lNewSwapchain] =
            mGraphicContext.mContext->CreateSwapChain();

        mVkObject   = lNewSwapchain;
        mImageFormat = lSwapChainImageFormat;
        mExtent      = lSwapchainExtent;

        mImages     = mGraphicContext.mContext->GetSwapChainImages( mVkObject );
        mImageCount = mImages.size();

        mImageAvailableSemaphores.resize( mImageCount );
        mRenderFinishedSemaphores.resize( mImageCount );
        mInFlightFences.resize( mImageCount );

        for( size_t i = 0; i < mImageCount; i++ )
        {
            mImageAvailableSemaphores[i] = mGraphicContext.mContext->CreateVkSemaphore();
            mRenderFinishedSemaphores[i] = mGraphicContext.mContext->CreateVkSemaphore();
            mInFlightFences[i]           = mGraphicContext.mContext->CreateFence();
        }

        Spec.Format     = ToLtseFormat( lSwapChainImageFormat );
        Spec.ClearColor = { 0.01f, 0.01f, 0.03f, 1.0f };
        Spec.Width      = lSwapchainExtent.width;
        Spec.Height     = lSwapchainExtent.height;
        Spec.Sampled    = false;
        Spec.Presented  = true;

        mRenderTargets.resize( mImageCount );

        for( int i = 0; i < mImageCount; i++ )
        {
            RenderTargetDescription l_RTSpec{};
            l_RTSpec.SampleCount   = Spec.SampleCount;
            l_RTSpec.Format        = ToLtseFormat( lSwapChainImageFormat );
            l_RTSpec.ClearColor    = { 0.01f, 0.01f, 0.03f, 1.0f };
            l_RTSpec.Width         = lSwapchainExtent.width;
            l_RTSpec.Height        = lSwapchainExtent.height;
            l_RTSpec.Sampled       = false;
            l_RTSpec.Presented     = true;
            l_RTSpec.OutputTexture = New<sVkFramebufferImage>( mGraphicContext.mContext, mImages[i], mImageFormat,
                lSwapchainExtent.width, lSwapchainExtent.height, Spec.SampleCount, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, false );

            mRenderTargets[i] = New<SwapChainRenderTargetImage>( mGraphicContext, l_RTSpec );
        }

        InitializeCommandBuffers();
    }

} // namespace LTSE::Graphics