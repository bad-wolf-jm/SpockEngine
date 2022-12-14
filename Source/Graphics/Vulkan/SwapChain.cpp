#include "SwapChain.h"

#include "Core/Memory.h"
// #include "GraphicContext.h"

namespace SE::Graphics
{
    SwapChain::SwapChain( Ref<VkGraphicContext> aGraphicContext, Ref<IWindow> aWindow )
        : ARenderTarget( aGraphicContext, sRenderTargetDescription{} )
        , mViewportClient{ aWindow }
    {
        RecreateSwapChain();
    }

    void SwapChain::RecreateSwapChain()
    {
        mGraphicContext->WaitIdle();
        mRenderTargets.clear();

        auto [lSwapChainImageFormat, lSwapChainImageCount, lSwapchainExtent, lNewSwapchain] = mGraphicContext->CreateSwapChain();

        mVkObject    = lNewSwapchain;
        auto lImages = mGraphicContext->GetSwapChainImages( mVkObject );

        mImageCount = lImages.size();

        mImageAvailableSemaphores.resize( mImageCount );
        mRenderFinishedSemaphores.resize( mImageCount );
        mInFlightFences.resize( mImageCount );

        for( size_t i = 0; i < mImageCount; i++ )
        {
            mImageAvailableSemaphores[i] = mGraphicContext->CreateVkSemaphore();
            mRenderFinishedSemaphores[i] = mGraphicContext->CreateVkSemaphore();
            mInFlightFences[i]           = mGraphicContext->CreateFence();
        }

        mSpec.mSampleCount = 1;
        mSpec.mWidth       = lSwapchainExtent.width;
        mSpec.mHeight      = lSwapchainExtent.height;

        for( int i = 0; i < mImageCount; i++ )
        {
            sTextureCreateInfo lTextureCreateInfo{};
            lTextureCreateInfo.mFormat = ToLtseFormat( lSwapChainImageFormat );
            lTextureCreateInfo.mWidth  = mSpec.mWidth;
            lTextureCreateInfo.mHeight = mSpec.mHeight;

            auto lFramebufferImage = New<VkTexture2D>( mGraphicContext, lTextureCreateInfo, lImages[i] );

            sRenderTargetDescription lCreateInfo{};
            lCreateInfo.mWidth   = lSwapchainExtent.width;
            lCreateInfo.mHeight  = lSwapchainExtent.height;
            auto lSwapChainImage = New<ARenderTarget>( mGraphicContext, lCreateInfo );
            lSwapChainImage->AddAttachment( "SWAPCHAIN_OUTPUT", eAttachmentType::COLOR, ToLtseFormat( lSwapChainImageFormat ),
                                            { 0.01f, 0.01f, 0.03f, 1.0f }, false, true, eAttachmentLoadOp::CLEAR,
                                            eAttachmentStoreOp::STORE, lFramebufferImage );

            lSwapChainImage->Finalize();

            mRenderTargets.push_back( lSwapChainImage );
        }

        mRenderPassObject = mRenderTargets[0]->GetRenderPass();

        InitializeCommandBuffers();
    }

    bool SwapChain::BeginRender()
    {
        mGraphicContext->WaitForFence( mInFlightFences[mCurrentImage] );

        uint64_t lTimeout = std::numeric_limits<uint64_t>::max();
        VkResult lBeginRenderResult =
            mGraphicContext->AcquireNextImage( mVkObject, lTimeout, mImageAvailableSemaphores[mCurrentImage], &mCurrentImage );

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
        VkResult lPresentResult = mGraphicContext->Present( mVkObject, mCurrentImage, mRenderFinishedSemaphores[mCurrentImage] );

        if( ( lPresentResult == VK_ERROR_OUT_OF_DATE_KHR ) || ( lPresentResult == VK_SUBOPTIMAL_KHR ) ||
            mViewportClient->WindowWasResized() )
        {
            mViewportClient->ResetWindowResizedFlag();
            RecreateSwapChain();
        }
        else if( lPresentResult != VK_SUCCESS )
        {
            throw std::runtime_error( "failed to present swap chain image!" );
        }
    }
} // namespace SE::Graphics
