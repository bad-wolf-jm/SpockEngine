#include "SwapChain.h"

#include "Core/Memory.h"
// #include "GraphicContext.h"

namespace SE::Graphics
{
    SwapChain::SwapChain( Ref<VkGraphicContext> aGraphicContext, Ref<IWindow> aWindow )
        : VkRenderTarget( aGraphicContext, sRenderTargetDescription{} )
        , mViewportClient{ aWindow }
    {
        mVkSurface = aGraphicContext->CreateVkSurface( aWindow );

        RecreateSwapChain();
    }

    SwapChain::~SwapChain() { GraphicContext<VkGraphicContext>()->DestroyVkSurface( mVkSurface ); }

    void SwapChain::RecreateSwapChain()
    {
        GraphicContext<VkGraphicContext>()->WaitIdle();
        mRenderTargets.clear();

        GraphicContext<VkGraphicContext>()->DestroySwapChain( mVkObject );

        auto [lSwapChainImageFormat, lSwapChainImageCount, lSwapchainExtent, lNewSwapchain] =
            GraphicContext<VkGraphicContext>()->CreateSwapChain( mViewportClient->GetExtent(), mVkSurface);

        mVkObject    = lNewSwapchain;
        auto lImages = GraphicContext<VkGraphicContext>()->GetSwapChainImages( mVkObject );

        mImageCount = lImages.size();

        mImageAvailableSemaphores.resize( mImageCount );
        mRenderFinishedSemaphores.resize( mImageCount );
        mInFlightFences.resize( mImageCount );

        for( size_t i = 0; i < mImageCount; i++ )
        {
            mImageAvailableSemaphores[i] = GraphicContext<VkGraphicContext>()->CreateVkSemaphore();
            mRenderFinishedSemaphores[i] = GraphicContext<VkGraphicContext>()->CreateVkSemaphore();
            mInFlightFences[i]           = GraphicContext<VkGraphicContext>()->CreateFence();
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

            auto lFramebufferImage =
                New<VkTexture2D>( GraphicContext<VkGraphicContext>(), lTextureCreateInfo, lImages[i] );

            sRenderTargetDescription lCreateInfo{};
            lCreateInfo.mWidth  = lSwapchainExtent.width;
            lCreateInfo.mHeight = lSwapchainExtent.height;
            auto lSwapChainImage =
                New<VkRenderTarget>( GraphicContext<VkGraphicContext>(), lCreateInfo );
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
        GraphicContext<VkGraphicContext>()->WaitForFence( mInFlightFences[mCurrentImage] );

        uint64_t lTimeout = std::numeric_limits<uint64_t>::max();
        VkResult lBeginRenderResult =
            GraphicContext<VkGraphicContext>()
                ->AcquireNextImage( mVkObject, lTimeout, mImageAvailableSemaphores[mCurrentImage], &mCurrentImage );

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
        VkResult lPresentResult = GraphicContext<VkGraphicContext>()
                                      ->Present( mVkObject, mCurrentImage, mRenderFinishedSemaphores[mCurrentImage] );

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
