#include "IRenderContext.h"

#include "Core/Logging.h"

namespace SE::Graphics
{

    IRenderContext::IRenderContext( Ref<IGraphicContext> aGraphicContext, Ref<IRenderTarget> aRenderTarget )
        : mGraphicContext{ aGraphicContext }
        , mRenderTarget{ aRenderTarget }
    {
    }

    bool IRenderContext::BeginRender()
    {
        mFrameIsStarted = mRenderTarget->BeginRender();

        return mFrameIsStarted;
    }

    bool IRenderContext::EndRender()
    {
        mRenderTarget->EndRender();

        mFrameIsStarted   = false;
        mGraphicsPipeline = nullptr;
        mHasIndex         = false;

        return mFrameIsStarted;
    }

    void IRenderContext::Present() { mRenderTarget->Present(); }

    void IRenderContext::ResetBuffers() { mHasIndex = false; }

    void IRenderContext::Draw( uint32_t aVertexCount, uint32_t aVertexOffset, uint32_t aVertexBufferOffset, uint32_t aInstanceCount,
                               uint32_t aFirstInstance )
    {
        if( mHasIndex )
            InternalDrawIndexed( aVertexCount, aVertexOffset, aVertexBufferOffset, aInstanceCount, aFirstInstance );
        else
            InternalDrawNonIndexed( aVertexCount, aVertexOffset, aVertexBufferOffset, aInstanceCount, aFirstInstance );
    }

    void IRenderContext::Bind( Ref<IGraphicsPipeline> aGraphicPipeline ) { mGraphicsPipeline = aGraphicPipeline; }

    void IRenderContext::Bind( Ref<IGraphicBuffer> aVertexBuffer, uint32_t aBindPoint )
    {
        mVertexBuffer = aVertexBuffer;

        mHasIndex = false;
    }

    void IRenderContext::Bind( Ref<IGraphicBuffer> aVertexBuffer, Ref<IGraphicBuffer> aIndexBuffer, uint32_t aBindPoint )
    {
        mVertexBuffer = aVertexBuffer;
        mIndexBuffer  = aIndexBuffer;

        mHasIndex = true;
    }
} // namespace SE::Graphics