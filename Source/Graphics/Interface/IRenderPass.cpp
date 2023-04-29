#include "IRenderPass.h"

#include "Core/Memory.h"
#include <stdexcept>

namespace SE::Graphics
{
    IRenderPass::IRenderPass( Ref<IGraphicContext> aGraphicContext, uint32_t aSampleCount )
        : mGraphicContext{ aGraphicContext }
        , mSampleCount{ aSampleCount }
    {
    }

    void IRenderPass::AddAttachment( std::string const &aAttachmentID, sAttachmentDescription const &aCreateInfo,
                                       Ref<ITexture> aFramebufferImage )
    {
        AddAttachment( aAttachmentID, aCreateInfo, aFramebufferImage, eCubeFace::NEGATIVE_Z );
    }

    void IRenderPass::AddAttachment( std::string const &aAttachmentID, sAttachmentDescription const &aCreateInfo,
                                       Ref<ITexture> aFramebufferImage, eCubeFace aFace )
    {
        mAttachmentInfo.push_back( aCreateInfo );
        mAttachmentIDs.push_back( aAttachmentID );

        if( aCreateInfo.mType == eAttachmentType::DEPTH ) mAttachmentInfo.back().mFormat = mGraphicContext->GetDepthFormat();

        mAttachments[aAttachmentID] = sAttachmentResource{ aFramebufferImage, aFace };
    }

    void IRenderPass::AddAttachment( std::string const &aAttachmentID, eAttachmentType aType, eColorFormat aFormat,
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



} // namespace SE::Graphics