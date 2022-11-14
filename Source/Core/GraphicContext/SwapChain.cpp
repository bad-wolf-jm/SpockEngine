#include "SwapChain.h"

#include "Core/Memory.h"
#include "GraphicContext.h"

namespace LTSE::Graphics
{
    SwapChain::SwapChain( GraphicContext &aGraphicContext )
        : ARenderTarget( aGraphicContext, sRenderTargetDescription{} )
    {
        RecreateSwapChain();
    }

    void SwapChain::RecreateSwapChain()
    {
        mGraphicContext.mContext->WaitIdle();

        auto [lSwapChainImageFormat, lSwapChainImageCount, lSwapchainExtent, lNewSwapchain] =
            mGraphicContext.mContext->CreateSwapChain();

        mVkObject    = lNewSwapchain;
        auto lImages = mGraphicContext.mContext->GetSwapChainImages( mVkObject );

        mImageCount = lImages.size();

        mImageAvailableSemaphores.resize( mImageCount );
        mRenderFinishedSemaphores.resize( mImageCount );
        mInFlightFences.resize( mImageCount );

        for( size_t i = 0; i < mImageCount; i++ )
        {
            mImageAvailableSemaphores[i] = mGraphicContext.mContext->CreateVkSemaphore();
            mRenderFinishedSemaphores[i] = mGraphicContext.mContext->CreateVkSemaphore();
            mInFlightFences[i]           = mGraphicContext.mContext->CreateFence();
        }

        mSpec.mSampleCount = 1;
        mSpec.mWidth       = lSwapchainExtent.width;
        mSpec.mHeight      = lSwapchainExtent.height;

        for( int i = 0; i < mImageCount; i++ )
        {
            auto lFramebufferImage = New<sVkFramebufferImage>( mGraphicContext.mContext, lImages[i], lSwapChainImageFormat,
                mSpec.mWidth, mSpec.mHeight, 1, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, false );

            sRenderTargetDescription lCreateInfo{};
            lCreateInfo.mWidth   = lSwapchainExtent.width;
            lCreateInfo.mHeight  = lSwapchainExtent.height;
            auto lSwapChainImage = New<ARenderTarget>( mGraphicContext, lCreateInfo );

            lSwapChainImage->AddAttachment( "SWAPCHAIN_OUTPUT", eAttachmentType::COLOR, ToLtseFormat( lSwapChainImageFormat ),
                { 0.01f, 0.01f, 0.03f, 1.0f }, false, true, eAttachmentLoadOp::CLEAR, eAttachmentStoreOp::UNSPECIFIED,
                lFramebufferImage );

            lSwapChainImage->Finalize();

            mRenderTargets.push_back( lSwapChainImage );
        }
    }

    bool SwapChain::BeginRender()
    {
        mGraphicContext.mContext->WaitForFence( mInFlightFences[mCurrentImage] );

        uint64_t lTimeout           = std::numeric_limits<uint64_t>::max();
        VkResult lBeginRenderResult = mGraphicContext.mContext->AcquireNextImage(
            mVkObject, lTimeout, mImageAvailableSemaphores[mCurrentImage], &mCurrentImage );

        if( lBeginRenderResult == VK_ERROR_OUT_OF_DATE_KHR )
        {
            mFrameIsStarted = false;
            RecreateSwapChain();
        }
        else
        {
            VK_CHECK_RESULT( lBeginRenderResult );

            mFrameIsStarted = true;
            mRenderTargets[mCurrentImage]->BeginRender();
        }

        return mFrameIsStarted;
    }

    void SwapChain::EndRender()
    {
        mRenderTargets[mCurrentImage]->EndRender();
        mFrameIsStarted = false;
    }

    void SwapChain::Present()
    {
        VkResult lPresentResult =
            mGraphicContext.mContext->Present( mVkObject, mCurrentImage, mRenderFinishedSemaphores[mCurrentImage] );

        if( ( lPresentResult == VK_ERROR_OUT_OF_DATE_KHR ) || ( lPresentResult == VK_SUBOPTIMAL_KHR ) ||
            mGraphicContext.m_ViewportClient->WindowWasResized() )
        {
            mGraphicContext.m_ViewportClient->ResetWindowResizedFlag();
            RecreateSwapChain();
        }
        else if( lPresentResult != VK_SUCCESS )
        {
            throw std::runtime_error( "failed to present swap chain image!" );
        }
    }
} // namespace LTSE::Graphics
