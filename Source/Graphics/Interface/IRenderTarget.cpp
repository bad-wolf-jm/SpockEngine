#include "IRenderTarget.h"

namespace SE::Graphics
{
    IRenderTarget::IRenderTarget( Ref<IGraphicContext> aGraphicContext, sRenderTargetDescription const &aRenderTargetDescription )
        : mGraphicContext{ aGraphicContext }
        , mSpec{ aRenderTargetDescription }
    {
        mImageCount = 1;
    }

    void IRenderTarget::AddAttachment( std::string const &aAttachmentID, sAttachmentDescription const &aCreateInfo,
                                       Ref<ITexture> aFramebufferImage )
    {
        AddAttachment( aAttachmentID, aCreateInfo, aFramebufferImage, eCubeFace::NEGATIVE_Z );
    }

    void IRenderTarget::AddAttachment( std::string const &aAttachmentID, sAttachmentDescription const &aCreateInfo,
                                       Ref<ITexture> aFramebufferImage, eCubeFace aFace )
    {
        mAttachmentInfo.push_back( aCreateInfo );
        mAttachmentIDs.push_back( aAttachmentID );

        if( aCreateInfo.mType == eAttachmentType::DEPTH ) mAttachmentInfo.back().mFormat = mGraphicContext->GetDepthFormat();

        mAttachments[aAttachmentID] = sAttachmentResource{ aFramebufferImage, aFace };
    }

    void IRenderTarget::AddAttachment( std::string const &aAttachmentID, eAttachmentType aType, eColorFormat aFormat,
                                       math::vec4 aClearColor, bool aIsSampled, bool aIsPresented, eAttachmentLoadOp aLoadOp,
                                       eAttachmentStoreOp eStoreOp, Ref<ITexture> aFramebufferImage )
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

    Ref<ITexture> IRenderTarget::GetAttachment( std::string const &aKey )
    {
        //
        return mAttachments[aKey].mTexture;
    }

    bool IRenderTarget::BeginRender() { return true; }

    void IRenderTarget::EndRender() {}

    void IRenderTarget::Present() {}

    void IRenderTarget::Finalize() {}
} // namespace SE::Graphics
