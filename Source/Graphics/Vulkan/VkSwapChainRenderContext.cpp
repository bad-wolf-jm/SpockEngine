#include "VkSwapChainRenderContext.h"
#include "VkGraphicsPipeline.h"
#include "SwapChain.h"
#include "Core/Logging.h"

namespace SE::Graphics
{

    VkSwapChainRenderContext::VkSwapChainRenderContext( Ref<IGraphicContext> aGraphicContext, Ref<IRenderTarget> aRenderTarget )
        : VkBaseRenderContext( aGraphicContext, aRenderTarget )
    {
    }

    bool VkSwapChainRenderContext::BeginRender()
    {
        if( mFrameIsStarted ) return true;

        VkBaseRenderContext::BeginRender();

        if( !mFrameIsStarted ) return false;

        float lWidth  = static_cast<float>( Cast<VkSwapChain>( mRenderTarget )->Spec().mWidth );
        float lHeight = static_cast<float>( Cast<VkSwapChain>( mRenderTarget )->Spec().mHeight );

        auto lCommandBuffer = GetCurrentCommandBuffer();
        lCommandBuffer->Begin();
        lCommandBuffer->BeginRenderPass( Cast<VkSwapChain>( mRenderTarget )->GetRenderPass(),
                                         Cast<VkSwapChain>( mRenderTarget )->GetFramebuffer(), { lWidth, lHeight },
                                         Cast<VkSwapChain>( mRenderTarget )->GetClearValues() );
        lCommandBuffer->SetViewport( { 0.0f, 0.0f }, { lWidth, lHeight } );
        lCommandBuffer->SetScissor( { 0.0f, 0.0f }, { lWidth, lHeight } );

        return true;
    }
} // namespace SE::Graphics