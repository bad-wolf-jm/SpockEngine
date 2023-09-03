#include "IRenderTarget.h"

#include "Graphics/API.h"

namespace SE::Graphics
{

    sAttachmentDescription::sAttachmentDescription( eAttachmentType aType, eColorFormat aFormat, bool aIsSampled )
        : mType{ aType }
        , mFormat{ aFormat }
        , mIsSampled{ aIsSampled }
        , mIsPresented{ false }
        , mLoadOp{ eAttachmentLoadOp::CLEAR }
        , mStoreOp{ eAttachmentStoreOp::STORE }
    {
        switch( mType )
        {
        case eAttachmentType::COLOR:
        case eAttachmentType::MSAA_RESOLVE:
            mClearColor = { 0.0f, 0.0f, 0.0f, 1.0f };
            break;
        case eAttachmentType::DEPTH:
            mClearColor = { 1.0f, 0.0f, 0.0f, 0.0f };
            break;
        }
    }

    IRenderTarget::IRenderTarget( ref_t<IGraphicContext> aGraphicContext, sRenderTargetDescription const &aRenderTargetDescription )
        : mGraphicContext{ aGraphicContext }
        , mSpec{ aRenderTargetDescription }
    {
    }

    ref_t<ITexture2D> IRenderTarget::GetAttachment( string_t const &aKey )
    {
        return mAttachments[aKey].mTexture;
    }

    void IRenderTarget::AddAttachment( string_t const &aAttachmentID, sAttachmentDescription const &aCreateInfo )
    {
        sTextureCreateInfo lTextureCreateInfo{};
        lTextureCreateInfo.mFormat         = aCreateInfo.mFormat;
        lTextureCreateInfo.mWidth          = mSpec.mWidth;
        lTextureCreateInfo.mHeight         = mSpec.mHeight;
        lTextureCreateInfo.mDepth          = 1;
        lTextureCreateInfo.mMipLevels      = 1;
        lTextureCreateInfo.mIsDepthTexture = ( aCreateInfo.mType == eAttachmentType::DEPTH );
        uint32_t lSampleCount              = ( aCreateInfo.mType == eAttachmentType::MSAA_RESOLVE ) ? 1 : mSpec.mSampleCount;

        auto lNewAttachment = CreateTexture2D( mGraphicContext, lTextureCreateInfo, lSampleCount, false, true, true, true );
        IRenderTarget::AddAttachment( aAttachmentID, aCreateInfo, lNewAttachment );
    }

    void IRenderTarget::AddAttachment( string_t const &aAttachmentID, sAttachmentDescription const &aCreateInfo,
                                       ref_t<ITexture2D> aFramebufferImage )
    {
        AddAttachment( aAttachmentID, aCreateInfo, aFramebufferImage, eCubeFace::POSITIVE_X );
    }

    void IRenderTarget::AddAttachment( string_t const &aAttachmentID, sAttachmentDescription const &aCreateInfo,
                                       ref_t<ITexture2D> aFramebufferImage, eCubeFace aFace )
    {
        mAttachmentInfo.push_back( aCreateInfo );
        mAttachmentIDs.push_back( aAttachmentID );

        if( aCreateInfo.mType == eAttachmentType::DEPTH )
            mAttachmentInfo.back().mFormat = mGraphicContext->GetDepthFormat();

        mAttachments[aAttachmentID] = sAttachmentResource{ aFramebufferImage, aFace };
    }

    void IRenderTarget::AddAttachment( string_t const &aAttachmentID, eAttachmentType aType, eColorFormat aFormat,
                                       math::vec4 aClearColor, bool aIsSampled, bool aIsPresented, eAttachmentLoadOp aLoadOp,
                                       eAttachmentStoreOp eStoreOp, ref_t<ITexture2D> aFramebufferImage )
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
