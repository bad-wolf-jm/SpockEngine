#include "ARenderTarget.h"

namespace LTSE::Graphics
{
    void ARenderTarget::AddAttachment( std::string const &aAttachmentID, sAttachmentDescription const &aCreateInfo )
    {
        mAttachmentInfo.push_back( aCreateInfo );
        VkImageUsageFlags lAttachmentType =
            ( aCreateInfo.mType == eAttachmentType::COLOR || aCreateInfo.mType == eAttachmentType::MSAA_RESOLVE )
                ? VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
                : VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

        uint32_t lSampleCount = ( aCreateInfo.mType == eAttachmentType::MSAA_RESOLVE ) ? 1 : mSpec.mSampleCount;

        mAttachmentIDs.push_back( aAttachmentID );
        mAttachments[aAttachmentID] = New<sVkFramebufferImage>( mGraphicContext.mContext, ToVkFormat( aCreateInfo.mFormat ),
            mSpec.mWidth, mSpec.mHeight, lSampleCount, lAttachmentType, aCreateInfo.mIsSampled );
    }

    ARenderTarget &ARenderTarget::AddAttachment( std::string const &aAttachmentID, eAttachmentType aType, eColorFormat aFormat,
        math::vec4 aClearColor, bool aIsSampled, bool aIsPresented, eAttachmentLoadOp aLoadOp, eAttachmentStoreOp eStoreOp )
    {
        sAttachmentDescription lCreateInfo{};
        lCreateInfo.mType        = aType;
        lCreateInfo.mFormat      = aFormat;
        lCreateInfo.mClearColor  = aClearColor;
        lCreateInfo.mIsSampled   = aIsSampled;
        lCreateInfo.mIsPresented = aIsPresented;
        lCreateInfo.mLoadOp      = aLoadOp;
        lCreateInfo.mStoreOp     = eStoreOp;

        AddAttachment( aAttachmentID, lCreateInfo );

        return *this;
    }

    void ARenderTarget::Finalize()
    {
        std::vector<Ref<Internal::sVkFramebufferImage>> lAttachments{};
        for( auto const &lAttachmentID : mAttachmentIDs ) lAttachments.push_back( mAttachments[lAttachmentID] );

        mRenderPassObject  = CreateDefaultRenderPass();
        mFramebufferObject = New<sVkFramebufferObject>(
            mGraphicContext.mContext, mSpec.mWidth, mSpec.mHeight, mSpec.mSampleCount, mRenderPassObject->mVkObject, lAttachments );
    }

    Ref<sVkAbstractRenderPassObject> ARenderTarget::CreateDefaultRenderPass()
    {
        Ref<sVkAbstractRenderPassObject> lNewRenderPass = New<sVkAbstractRenderPassObject>(
            mGraphicContext.mContext, VK_FORMAT_UNDEFINED, mSpec.mSampleCount, false, false, math::vec4( 0.0f ) );

        std::vector<VkAttachmentDescription> lAttachmentDescriptions{};
        std::vector<VkAttachmentReference>   lColorAttachmentReferences{};

        VkAttachmentReference  lDepthAttachment{};
        VkAttachmentReference *lDepthAttachmentPtr = nullptr;

        VkAttachmentReference  lResolveAttachment{};
        VkAttachmentReference *lResolveAttachmentPtr = nullptr;

        for( uint32_t i = 0; i < mAttachmentInfo.size(); i++ )
        {
            VkAttachmentDescription lDescription;
            switch( mAttachmentInfo[i].mType )
            {
            case eAttachmentType::COLOR:
            {
                lDescription = lNewRenderPass->ColorAttachment( ToVkFormat( mAttachmentInfo[i].mFormat ), mSpec.mSampleCount,
                    mAttachmentInfo[i].mIsSampled, mAttachmentInfo[i].mIsPresented );
                lAttachmentDescriptions.push_back( lDescription );
                VkAttachmentReference lAttachmentReference{};
                lAttachmentReference.attachment = i;
                lAttachmentReference.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                lColorAttachmentReferences.push_back( lAttachmentReference );
                break;
            }
            case eAttachmentType::MSAA_RESOLVE:
            {
                lDescription = lNewRenderPass->ColorAttachment( ToVkFormat( mAttachmentInfo[i].mFormat ), mSpec.mSampleCount,
                    mAttachmentInfo[i].mIsSampled, mAttachmentInfo[i].mIsPresented );
                lAttachmentDescriptions.push_back( lDescription );
                lResolveAttachment.attachment = i;
                lResolveAttachment.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                lResolveAttachmentPtr         = &lResolveAttachment;
                break;
            }
            case eAttachmentType::DEPTH:
            {
                lDescription = lNewRenderPass->DepthAttachment( mSpec.mSampleCount );
                lAttachmentDescriptions.push_back( lDescription );
                lDepthAttachment.attachment = i;
                lDepthAttachment.layout     = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
                lDepthAttachmentPtr         = &lDepthAttachment;
                break;
            }
            default:
                break;
            }
        }

        lNewRenderPass->CreateUnderlyingRenderpass(
            lAttachmentDescriptions, lColorAttachmentReferences, lDepthAttachmentPtr, lResolveAttachmentPtr );

        return lNewRenderPass;
    }

} // namespace LTSE::Graphics
