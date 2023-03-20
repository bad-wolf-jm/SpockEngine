#include "Figure.h"

namespace SE::Core
{

    Figure::Figure( Ref<VkGraphicContext> aGraphicContext )
        : mGraphicContext{ aGraphicContext }
    {
    }

    void Figure::ResizeOutput( uint32_t aOutputWidth, uint32_t aOutputHeight )
    {
        mViewWidth  = aOutputWidth;
        mViewHeight = aOutputHeight;

        sRenderTargetDescription lRenderTargetSpec{};
        lRenderTargetSpec.mWidth       = mViewWidth;
        lRenderTargetSpec.mHeight      = aOutputHeight;
        lRenderTargetSpec.mSampleCount = mViewHeight;
        mRenderTarget                  = New<VkRenderTarget>( mGraphicContext, lRenderTargetSpec );

        sAttachmentDescription lAttachmentCreateInfo{};
        lAttachmentCreateInfo.mType        = eAttachmentType::COLOR;
        lAttachmentCreateInfo.mFormat      = eColorFormat::RGBA8_UNORM;
        lAttachmentCreateInfo.mClearColor  = { 0.0f, 0.0f, 0.0f, 1.0f };
        lAttachmentCreateInfo.mLoadOp      = eAttachmentLoadOp::CLEAR;
        lAttachmentCreateInfo.mStoreOp     = eAttachmentStoreOp::STORE;
        lAttachmentCreateInfo.mIsSampled   = true;
        lAttachmentCreateInfo.mIsPresented = false;
        mRenderTarget->AddAttachment( "OUTPUT", lAttachmentCreateInfo );

        lAttachmentCreateInfo.mType       = eAttachmentType::DEPTH;
        lAttachmentCreateInfo.mClearColor = { 1.0f, 0.0f, 0.0f, 0.0f };
        lAttachmentCreateInfo.mLoadOp     = eAttachmentLoadOp::LOAD;
        lAttachmentCreateInfo.mStoreOp    = eAttachmentStoreOp::UNSPECIFIED;
        lAttachmentCreateInfo.mIsDefined  = false;
        mRenderTarget->AddAttachment( "DEPTH_STENCIL", lAttachmentCreateInfo );

        mRenderTarget->Finalize();

        mRenderContext = ARenderContext( mGraphicContext, mRenderTarget );
    }

    Ref<VkTexture2D> Figure::GetOutputImage() { return mRenderTarget->GetAttachment( "OUTPUT" ); }

    void Figure::Render()
    {
        mRenderContext.BeginRender();

        mRenderContext.EndRender();
    }
} // namespace SE::Core