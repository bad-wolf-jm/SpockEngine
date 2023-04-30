#include "VkSwapChain.h"

#include "Core/Memory.h"
// #include "GraphicContext.h"

namespace SE::Graphics
{
    VkSwapChain::VkSwapChain( Ref<IGraphicContext> aGraphicContext, Ref<IWindow> aWindow )
        : ISwapChain( aGraphicContext, aWindow )
        , mViewportClient{ aWindow }
    {
        mVkSurface = Cast<VkGraphicContext>( mGraphicContext )->CreateVkSurface( aWindow );

        RecreateSwapChain();
    }

    VkSwapChain::~VkSwapChain() { GraphicContext<VkGraphicContext>()->DestroyVkSurface( mVkSurface ); }

    void VkSwapChain::RecreateSwapChain()
    {
        GraphicContext<VkGraphicContext>()->WaitIdle();
        mRenderTargets.clear();

        GraphicContext<VkGraphicContext>()->DestroySwapChain( mVkObject );

        auto [lSwapChainImageFormat, lSwapChainImageCount, lSwapchainExtent, lNewSwapchain] =
            GraphicContext<VkGraphicContext>()->CreateSwapChain( mViewportClient->GetExtent(), mVkSurface );

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

            auto lFramebufferImage = New<VkTexture2D>( GraphicContext<VkGraphicContext>(), lTextureCreateInfo, lImages[i] );

            sRenderTargetDescription lCreateInfo{};
            lCreateInfo.mWidth   = lSwapchainExtent.width;
            lCreateInfo.mHeight  = lSwapchainExtent.height;
            auto lSwapChainImage = New<VkRenderTarget>( GraphicContext<VkGraphicContext>(), lCreateInfo );
            lSwapChainImage->AddAttachment( "SWAPCHAIN_OUTPUT", eAttachmentType::COLOR, ToLtseFormat( lSwapChainImageFormat ),
                                            { 0.01f, 0.01f, 0.03f, 1.0f }, false, true, eAttachmentLoadOp::CLEAR,
                                            eAttachmentStoreOp::STORE, lFramebufferImage );

            lSwapChainImage->Finalize();

            mRenderTargets.push_back( lSwapChainImage );
        }

        InitializeCommandBuffers();
    }

    bool VkSwapChain::BeginRender()
    {
        GraphicContext<VkGraphicContext>()->WaitForFence( mInFlightFences[mCurrentImage] );

        uint64_t lTimeout           = std::numeric_limits<uint64_t>::max();
        VkResult lBeginRenderResult = GraphicContext<VkGraphicContext>()->AcquireNextImage(
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

    void VkSwapChain::EndRender()
    {
        mRenderTargets[mCurrentImage]->EndRender();
        mFrameIsStarted = false;
    }

    void VkSwapChain::Present()
    {
        VkResult lPresentResult =
            GraphicContext<VkGraphicContext>()->Present( mVkObject, mCurrentImage, mRenderFinishedSemaphores[mCurrentImage] );

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

    void VkSwapChain::InitializeCommandBuffers()
    {
        auto lCommandBuffers = GraphicContext<VkGraphicContext>()->AllocateCommandBuffer( mImageCount );

        mCommandBufferObject = {};

        for( auto &lCB : lCommandBuffers )
            mCommandBufferObject.push_back( New<sVkCommandBufferObject>( GraphicContext<VkGraphicContext>(), lCB ) );

        for( size_t i = 0; i < mImageCount; i++ )
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

} // namespace SE::Graphics
