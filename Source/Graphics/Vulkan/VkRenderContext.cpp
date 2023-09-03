#include "VkRenderContext.h"
#include "VkGraphicsPipeline.h"

#include "Core/Logging.h"

namespace SE::Graphics
{

    VkRenderContext::VkRenderContext( ref_t<IGraphicContext> aGraphicContext, ref_t<IRenderTarget> aRenderTarget )
        : VkBaseRenderContext( aGraphicContext, aRenderTarget )
    {
    }

    bool VkRenderContext::BeginRender()
    {
        if( mFrameIsStarted ) return true;

        VkBaseRenderContext::BeginRender();

        if( !mFrameIsStarted ) return false;

        float lWidth  = static_cast<float>( Cast<VkRenderTarget>( mRenderTarget )->mSpec.mWidth );
        float lHeight = static_cast<float>( Cast<VkRenderTarget>( mRenderTarget )->mSpec.mHeight );

        auto lCommandBuffer = GetCurrentCommandBuffer();
        lCommandBuffer->Begin();
        lCommandBuffer->BeginRenderPass( Cast<VkRenderTarget>( mRenderTarget )->GetRenderPass(),
                                         Cast<VkRenderTarget>( mRenderTarget )->GetFramebuffer(), { lWidth, lHeight },
                                         Cast<VkRenderTarget>( mRenderTarget )->GetClearValues() );
        lCommandBuffer->SetViewport( { 0.0f, 0.0f }, { lWidth, lHeight } );
        lCommandBuffer->SetScissor( { 0.0f, 0.0f }, { lWidth, lHeight } );

        return true;
    }
} // namespace SE::Graphics