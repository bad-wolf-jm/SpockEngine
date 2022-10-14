#include "RenderTarget.h"
#include "Core/Logging.h"

namespace LTSE::Graphics
{
    AbstractRenderTarget::AbstractRenderTarget( GraphicContext &a_GraphicContext )
        : mGraphicContext{ a_GraphicContext }
    {
    }

    void AbstractRenderTarget::Initialize( RenderTargetDescription &a_Spec )
    {
        Spec = a_Spec;

        m_RenderPassObject =
            New<Internal::sVkRenderPassObject>( mGraphicContext.mContext, ToVkFormat( a_Spec.Format ), Spec.SampleCount, Spec.Sampled, Spec.Presented, Spec.ClearColor );

        if( Spec.SampleCount > 1 )
        {
            m_MSAAOutputTexture = New<Internal::sVkFramebufferImage>( mGraphicContext.mContext, ToVkFormat( a_Spec.Format ), Spec.Width, Spec.Height, Spec.SampleCount,
                                                                      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, false );
        }

        if( !Spec.OutputTexture )
        {
            m_OutputTexture =
                New<Internal::sVkFramebufferImage>( mGraphicContext.mContext, ToVkFormat( a_Spec.Format ), Spec.Width, Spec.Height, 1, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, true );
        }
        else
        {
            m_OutputTexture = Spec.OutputTexture;
        }

        m_DepthTexture = New<Internal::sVkFramebufferImage>( mGraphicContext.mContext, mGraphicContext.mContext->GetDepthFormat(), Spec.Width, Spec.Height, Spec.SampleCount,
                                                             VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, false );

        std::vector<Ref<Internal::sVkFramebufferImage>> l_AttachmentViews{};

        if( Spec.SampleCount == 1 )
            l_AttachmentViews = { m_OutputTexture, m_DepthTexture };
        else
            l_AttachmentViews = { m_MSAAOutputTexture, m_OutputTexture, m_DepthTexture };

        m_FramebufferObject = New<Internal::sVkFramebufferObject>( mGraphicContext.mContext, Spec.Width, Spec.Height, 1, m_RenderPassObject->mVkObject, l_AttachmentViews );
    }

    void AbstractRenderTarget::InitializeCommandBuffers()
    {
        auto lCommandBuffers = mGraphicContext.mContext->AllocateCommandBuffer( GetImageCount() );

        mCommandBufferObject = {};

        for( auto &lCB : lCommandBuffers )
            mCommandBufferObject.push_back( New<Internal::sVkCommandBufferObject>( mGraphicContext.mContext, lCB ) );

        for( size_t i = 0; i < GetImageCount(); i++ )
        {
            auto lImageAvailableSemaphore = GetImageAvailableSemaphore( i );
            if( lImageAvailableSemaphore )
                mCommandBufferObject[i]->AddWaitSemaphore( lImageAvailableSemaphore, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT );

            auto lRenderFinishedSemaphore = GetRenderFinishedSemaphore( i );
            if( lRenderFinishedSemaphore )
                mCommandBufferObject[i]->AddSignalSemaphore( lRenderFinishedSemaphore );

            auto lSubmitFence = GetInFlightFence( i );
            if( lSubmitFence )
                mCommandBufferObject[i]->SetSubmitFence( lSubmitFence );
        }
    }

    bool AbstractRenderTarget::BeginRender() { return true; }
    void AbstractRenderTarget::EndRender() {}
    void AbstractRenderTarget::Present() {}
    uint32_t AbstractRenderTarget::GetCurrentImage() { return 0; };

    Ref<Internal::sVkFramebufferObject> AbstractRenderTarget::GetFramebuffer() { return nullptr; }
    Ref<Internal::sVkCommandBufferObject> AbstractRenderTarget::GetCommandBuffer( uint32_t i ) { return mCommandBufferObject[i]; }
    VkSemaphore AbstractRenderTarget::GetImageAvailableSemaphore( uint32_t i ) { return VK_NULL_HANDLE; }
    VkSemaphore AbstractRenderTarget::GetRenderFinishedSemaphore( uint32_t i ) { return VK_NULL_HANDLE; }
    VkFence AbstractRenderTarget::GetInFlightFence( uint32_t i ) { return VK_NULL_HANDLE; }

    OffscreenRenderTarget::OffscreenRenderTarget( GraphicContext &a_GraphicContext, OffscreenRenderTargetDescription &a_Spec )
        : AbstractRenderTarget( a_GraphicContext )
    {
        mImageCount = 1;

        RenderTargetDescription l_RTDEscription{};
        l_RTDEscription.SampleCount   = a_Spec.SampleCount;
        l_RTDEscription.Format        = a_Spec.Format;
        l_RTDEscription.ClearColor    = a_Spec.ClearColor;
        l_RTDEscription.Width         = a_Spec.OutputSize.x;
        l_RTDEscription.Height        = a_Spec.OutputSize.y;
        l_RTDEscription.Sampled       = a_Spec.Sampled;
        l_RTDEscription.OutputTexture = nullptr;

        Initialize( l_RTDEscription );
        InitializeCommandBuffers();
    }

    void OffscreenRenderTarget::Resize( uint32_t aWidth, uint32_t aHeight )
    {
        Spec.Width         = aWidth;
        Spec.Height        = aHeight;
        Spec.OutputTexture = nullptr;

        Initialize( Spec );
        InitializeCommandBuffers();
    }

    bool OffscreenRenderTarget::BeginRender() { return true; }
    void OffscreenRenderTarget::EndRender() {}
    void OffscreenRenderTarget::Present() {}
    uint32_t OffscreenRenderTarget::GetCurrentImage() { return 0; };
    Ref<Internal::sVkFramebufferObject> OffscreenRenderTarget::GetFramebuffer() { return m_FramebufferObject; }
    Ref<Internal::sVkCommandBufferObject> OffscreenRenderTarget::GetCommandBuffer( uint32_t i ) { return AbstractRenderTarget::GetCommandBuffer( i ); }
    VkSemaphore OffscreenRenderTarget::GetImageAvailableSemaphore( uint32_t i ) { return AbstractRenderTarget::GetImageAvailableSemaphore( i ); }
    VkSemaphore OffscreenRenderTarget::GetRenderFinishedSemaphore( uint32_t i ) { return AbstractRenderTarget::GetRenderFinishedSemaphore( i ); }
    VkFence OffscreenRenderTarget::GetInFlightFence( uint32_t i ) { return AbstractRenderTarget::GetInFlightFence( i ); }

    SwapChainRenderTargetImage::SwapChainRenderTargetImage( GraphicContext &a_GraphicContext, RenderTargetDescription &a_Spec )
        : AbstractRenderTarget( a_GraphicContext )
    {
        mImageCount = 1;
        Initialize( a_Spec );
    }

    bool SwapChainRenderTargetImage::BeginRender() { return true; }
    void SwapChainRenderTargetImage::EndRender() {}
    void SwapChainRenderTargetImage::Present() {}
    uint32_t SwapChainRenderTargetImage::GetCurrentImage() { return 0; };
    Ref<Internal::sVkFramebufferObject> SwapChainRenderTargetImage::GetFramebuffer() { return m_FramebufferObject; }
    Ref<Internal::sVkCommandBufferObject> SwapChainRenderTargetImage::GetCommandBuffer( uint32_t i ) { return nullptr; }
    VkSemaphore SwapChainRenderTargetImage::GetImageAvailableSemaphore( uint32_t i ) { return AbstractRenderTarget::GetImageAvailableSemaphore( i ); }
    VkSemaphore SwapChainRenderTargetImage::GetRenderFinishedSemaphore( uint32_t i ) { return AbstractRenderTarget::GetRenderFinishedSemaphore( i ); }
    VkFence SwapChainRenderTargetImage::GetInFlightFence( uint32_t i ) { return AbstractRenderTarget::GetInFlightFence( i ); }

    SwapChainRenderTarget::SwapChainRenderTarget( GraphicContext &a_GraphicContext, SwapChainRenderTargetDescription &a_Spec )
        : AbstractRenderTarget( a_GraphicContext )
    {
        Spec.SampleCount = a_Spec.SampleCount;

        RecreateSwapChain();
    }

    bool SwapChainRenderTarget::BeginRender()
    {
        mGraphicContext.mContext->WaitForFence( InFlightFences[CurrentImage] );

        uint64_t l_Timeout = std::numeric_limits<uint64_t>::max();
        VkResult result    = mGraphicContext.mContext->AcquireNextImage( mVkObject, l_Timeout, ImageAvailableSemaphores[CurrentImage], &CurrentImage );

        if( result == VK_ERROR_OUT_OF_DATE_KHR )
        {
            FrameIsStarted = false;
            RecreateSwapChain();
            return FrameIsStarted;
        }
        else
        {
            VK_CHECK_RESULT( result );
            FrameIsStarted = true;
            m_RenderTargets[CurrentImage]->BeginRender();
            return FrameIsStarted;
        }

        // if( lFrameIsStarted )
        // {
        // }
        // else
        // {
        // }

        return FrameIsStarted;
    }

    void SwapChainRenderTarget::EndRender()
    {
        m_RenderTargets[CurrentImage]->EndRender();
        FrameIsStarted = false;
    }

    void SwapChainRenderTarget::Present()
    {

        VkResult l_PresentResult = mGraphicContext.mContext->Present( mVkObject, CurrentImage, RenderFinishedSemaphores[CurrentImage] );

        if( ( l_PresentResult == VK_ERROR_OUT_OF_DATE_KHR ) || ( l_PresentResult == VK_SUBOPTIMAL_KHR ) || mGraphicContext.m_ViewportClient->WindowWasResized() )
        {
            mGraphicContext.m_ViewportClient->ResetWindowResizedFlag();
            RecreateSwapChain();
        }
        else if( l_PresentResult != VK_SUCCESS )
            throw std::runtime_error( "failed to present swap chain image!" );
    }

    uint32_t SwapChainRenderTarget::GetCurrentImage() { return CurrentImage; };

    Ref<Internal::sVkFramebufferObject> SwapChainRenderTarget::GetFramebuffer() { return m_RenderTargets[CurrentImage]->GetFramebuffer(); }
    Ref<Internal::sVkCommandBufferObject> SwapChainRenderTarget::GetCommandBuffer( uint32_t i ) { return AbstractRenderTarget::GetCommandBuffer( i ); }
    VkSemaphore SwapChainRenderTarget::GetImageAvailableSemaphore( uint32_t i ) { return ImageAvailableSemaphores[i]; }
    VkSemaphore SwapChainRenderTarget::GetRenderFinishedSemaphore( uint32_t i ) { return RenderFinishedSemaphores[i]; }
    VkFence SwapChainRenderTarget::GetInFlightFence( uint32_t i ) { return InFlightFences[i]; }

    void SwapChainRenderTarget::RecreateSwapChain()
    {
        mGraphicContext.mContext->WaitIdle();

        auto [lSwapChainImageFormat, lSwapChainImageCount, lSwapchainExtent, lNewSwapchain] = mGraphicContext.mContext->CreateSwapChain();

        mVkObject   = lNewSwapchain;
        ImageFormat = lSwapChainImageFormat;
        Extent      = lSwapchainExtent;

        Images     = mGraphicContext.mContext->GetSwapChainImages( mVkObject );
        ImageCount = Images.size();

        ImageAvailableSemaphores.resize( ImageCount );
        RenderFinishedSemaphores.resize( ImageCount );
        InFlightFences.resize( ImageCount );

        for( size_t i = 0; i < ImageCount; i++ )
        {
            ImageAvailableSemaphores[i] = mGraphicContext.mContext->CreateVkSemaphore();
            RenderFinishedSemaphores[i] = mGraphicContext.mContext->CreateVkSemaphore();
            InFlightFences[i]           = mGraphicContext.mContext->CreateFence();
        }

        mImageCount     = ImageCount;
        Spec.Format     = ToLtseFormat( lSwapChainImageFormat );
        Spec.ClearColor = { 0.01f, 0.01f, 0.03f, 1.0f };
        Spec.Width      = lSwapchainExtent.width;
        Spec.Height     = lSwapchainExtent.height;
        Spec.Sampled    = false;
        Spec.Presented  = true;

        m_RenderTargets.resize( mImageCount );

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
            l_RTSpec.OutputTexture = New<Internal::sVkFramebufferImage>( mGraphicContext.mContext, Images[i], ImageFormat, lSwapchainExtent.width, lSwapchainExtent.height,
                                                                         Spec.SampleCount, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, false );

            m_RenderTargets[i] = New<SwapChainRenderTargetImage>( mGraphicContext, l_RTSpec );
        }

        InitializeCommandBuffers();
    }

} // namespace LTSE::Graphics