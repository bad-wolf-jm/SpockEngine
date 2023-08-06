#include "IRenderTarget.h"

namespace SE::Graphics
{
    IRenderTarget::IRenderTarget( ref_t<IGraphicContext> aGraphicContext, sRenderTargetDescription const &aRenderTargetDescription )
        : mGraphicContext{ aGraphicContext }
        , mSpec{ aRenderTargetDescription }
    {
    }

    ref_t<ITexture> IRenderTarget::GetAttachment( string_t const &aKey )
    {
        return mAttachments[aKey].mTexture;
    }

    void IRenderTarget::AddAttachment( string_t const &aAttachmentID, sAttachmentDescription const &aCreateInfo,
                                       ref_t<ITexture> aFramebufferImage )
    {
        AddAttachment( aAttachmentID, aCreateInfo, aFramebufferImage, eCubeFace::NEGATIVE_Z );
    }

    void IRenderTarget::AddAttachment( string_t const &aAttachmentID, sAttachmentDescription const &aCreateInfo,
                                       ref_t<ITexture> aFramebufferImage, eCubeFace aFace )
    {
        mAttachmentInfo.push_back( aCreateInfo );
        mAttachmentIDs.push_back( aAttachmentID );

        if( aCreateInfo.mType == eAttachmentType::DEPTH )
            mAttachmentInfo.back().mFormat = mGraphicContext->GetDepthFormat();

        mAttachments[aAttachmentID] = sAttachmentResource{ aFramebufferImage, aFace };
    }

    void IRenderTarget::AddAttachment( string_t const &aAttachmentID, eAttachmentType aType, eColorFormat aFormat,
                                       math::vec4 aClearColor, bool aIsSampled, bool aIsPresented, eAttachmentLoadOp aLoadOp,
                                       eAttachmentStoreOp eStoreOp, ref_t<ITexture> aFramebufferImage )
    {
        sAttachmentDescription lCreateInfo{};
        lCreateInfo.mType        = aType;
        lCreateInfo.mFormat      = aFormat;
        lCreateInfo.mClearColor  = aClearColor;
        lCreateInfo.mIsSampled   = aIsSampled;
        lCreateInfo.mIsPresented = aIsPresented;
        lCreateInfo.mLoadOp      = aLoadOp;
        lCreateInfo.mStoreOp     = eStoreOp;

        AddAttachment( aAttachmentID, lCreateInfo, aFramebufferImage );
    }

    bool IRenderTarget::BeginRender()
    {
        return true;
    }

    void IRenderTarget::EndRender()
    {
    }

    void IRenderTarget::Present()
    {
    }

    void IRenderTarget::Finalize()
    {
    }
    ref_t<IRenderPass> IRenderTarget::GetRenderPass()
    {
        return nullptr;
    }
} // namespace SE::Graphics
