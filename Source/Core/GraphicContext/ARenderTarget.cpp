#include "ARenderTarget.h"

namespace LTSE::Graphics
{
    ARenderTarget::ARenderTarget( GraphicContext &aGraphicContext, sRenderTargetDescription const &aRenderTargetDescription )
        : mGraphicContext{ aGraphicContext }
        , mSpec{ aRenderTargetDescription }
    {
        mImageCount = 1;
    }

    std::vector<VkClearValue> ARenderTarget::GetClearValues()
    {
        std::vector<VkClearValue> lValues;
        for( auto const &lInfo : mAttachmentInfo )
        {
            if( ( lInfo.mType == eAttachmentType::COLOR ) || ( lInfo.mType == eAttachmentType::MSAA_RESOLVE ) )
            {
                VkClearValue lValue{};
                lValue.color = { lInfo.mClearColor.x, lInfo.mClearColor.y, lInfo.mClearColor.z, lInfo.mClearColor.w };
                lValues.push_back( lValue );
            }
            else
            {
                VkClearValue lValue{};
                lValue.depthStencil = { lInfo.mClearColor.x, static_cast<uint32_t>( lInfo.mClearColor.y ) };
                lValues.push_back( lValue );
            }
        }

        return lValues;
    }

    void ARenderTarget::AddAttachment( std::string const &aAttachmentID, sAttachmentDescription const &aCreateInfo )
    {
        mAttachmentInfo.push_back( aCreateInfo );
        VkImageUsageFlags lAttachmentType =
            ( aCreateInfo.mType == eAttachmentType::COLOR || aCreateInfo.mType == eAttachmentType::MSAA_RESOLVE )
                ? VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
                : VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

        uint32_t lSampleCount = ( aCreateInfo.mType == eAttachmentType::MSAA_RESOLVE ) ? 1 : mSpec.mSampleCount;
        auto     lVkFormat    = ( aCreateInfo.mType == eAttachmentType::DEPTH ) ? mGraphicContext.mContext->GetDepthFormat()
                                                                                : ToVkFormat( aCreateInfo.mFormat );

        mAttachmentIDs.push_back( aAttachmentID );
        mAttachments[aAttachmentID] = New<sVkFramebufferImage>( mGraphicContext.mContext, lVkFormat, mSpec.mWidth, mSpec.mHeight,
                                                                lSampleCount, lAttachmentType, aCreateInfo.mIsSampled );
    }

    void ARenderTarget::AddAttachment( std::string const &aAttachmentID, sAttachmentDescription const &aCreateInfo,
                                       Ref<Internal::sVkFramebufferImage> aFramebufferImage )
    {
        mAttachmentInfo.push_back( aCreateInfo );
        VkImageUsageFlags lAttachmentType =
            ( aCreateInfo.mType == eAttachmentType::COLOR || aCreateInfo.mType == eAttachmentType::MSAA_RESOLVE )
                ? VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
                : VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

        uint32_t lSampleCount = ( aCreateInfo.mType == eAttachmentType::MSAA_RESOLVE ) ? 1 : mSpec.mSampleCount;
        auto     lVkFormat    = ( aCreateInfo.mType == eAttachmentType::DEPTH ) ? mGraphicContext.mContext->GetDepthFormat()
                                                                                : ToVkFormat( aCreateInfo.mFormat );

        mAttachmentIDs.push_back( aAttachmentID );
        mAttachments[aAttachmentID] = aFramebufferImage;
    }

    ARenderTarget &ARenderTarget::AddAttachment( std::string const &aAttachmentID, eAttachmentType aType, eColorFormat aFormat,
                                                 math::vec4 aClearColor, bool aIsSampled, bool aIsPresented, eAttachmentLoadOp aLoadOp,
                                                 eAttachmentStoreOp eStoreOp )
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

    ARenderTarget &ARenderTarget::AddAttachment( std::string const &aAttachmentID, eAttachmentType aType, eColorFormat aFormat,
                                                 math::vec4 aClearColor, bool aIsSampled, bool aIsPresented, eAttachmentLoadOp aLoadOp,
                                                 eAttachmentStoreOp eStoreOp, Ref<Internal::sVkFramebufferImage> aFramebufferImage )
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

        return *this;
    }

    void ARenderTarget::Finalize()
    {
        std::vector<Ref<Internal::sVkFramebufferImage>> lAttachments{};
        for( auto const &lAttachmentID : mAttachmentIDs ) lAttachments.push_back( mAttachments[lAttachmentID] );

        mRenderPassObject  = CreateDefaultRenderPass();
        mFramebufferObject = New<sVkFramebufferObject>( mGraphicContext.mContext, mSpec.mWidth, mSpec.mHeight, 1,
                                                        mRenderPassObject->mVkObject, lAttachments );

        InitializeCommandBuffers();
    }

    VkAttachmentLoadOp ToVkLoadOp( eAttachmentLoadOp aOp )
    {
        switch( aOp )
        {
        case eAttachmentLoadOp::CLEAR: return VK_ATTACHMENT_LOAD_OP_CLEAR;
        case eAttachmentLoadOp::LOAD: return VK_ATTACHMENT_LOAD_OP_LOAD;
        case eAttachmentLoadOp::UNSPECIFIED:
        default: return VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        }
    }

    VkAttachmentStoreOp ToVkStoreOp( eAttachmentStoreOp aOp )
    {
        switch( aOp )
        {
        case eAttachmentStoreOp::STORE: return VK_ATTACHMENT_STORE_OP_STORE;
        case eAttachmentStoreOp::UNSPECIFIED:
        default: return VK_ATTACHMENT_STORE_OP_DONT_CARE;
        }
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
            VkAttachmentLoadOp      lAttachmentLoadOp  = ToVkLoadOp( mAttachmentInfo[i].mLoadOp );
            VkAttachmentStoreOp     lAttachmentStoreOp = ToVkStoreOp( mAttachmentInfo[i].mStoreOp );

            switch( mAttachmentInfo[i].mType )
            {
            case eAttachmentType::COLOR:
            {
                lDescription = lNewRenderPass->ColorAttachment( ToVkFormat( mAttachmentInfo[i].mFormat ), mSpec.mSampleCount,
                                                                mAttachmentInfo[i].mIsSampled, mAttachmentInfo[i].mIsPresented, mAttachmentInfo[i].mIsDefined,
                                                                lAttachmentLoadOp, lAttachmentStoreOp );
                lAttachmentDescriptions.push_back( lDescription );
                VkAttachmentReference lAttachmentReference{};
                lAttachmentReference.attachment = i;
                lAttachmentReference.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                lColorAttachmentReferences.push_back( lAttachmentReference );
                break;
            }
            case eAttachmentType::MSAA_RESOLVE:
            {
                lDescription =
                    lNewRenderPass->ColorAttachment( ToVkFormat( mAttachmentInfo[i].mFormat ), 1, mAttachmentInfo[i].mIsSampled,
                                                     mAttachmentInfo[i].mIsPresented, mAttachmentInfo[i].mIsDefined, lAttachmentLoadOp, lAttachmentStoreOp );
                lAttachmentDescriptions.push_back( lDescription );
                lResolveAttachment.attachment = i;
                lResolveAttachment.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                lResolveAttachmentPtr         = &lResolveAttachment;
                break;
            }
            case eAttachmentType::DEPTH:
            {
                lDescription = lNewRenderPass->DepthAttachment( mAttachmentInfo[i].mIsDefined, mSpec.mSampleCount, lAttachmentLoadOp, lAttachmentStoreOp );
                lAttachmentDescriptions.push_back( lDescription );
                lDepthAttachment.attachment = i;
                lDepthAttachment.layout     = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
                lDepthAttachmentPtr         = &lDepthAttachment;
                break;
            }
            default: break;
            }
        }

        lNewRenderPass->CreateUnderlyingRenderpass( lAttachmentDescriptions, lColorAttachmentReferences, lDepthAttachmentPtr,
                                                    lResolveAttachmentPtr );

        return lNewRenderPass;
    }

    Ref<sVkFramebufferImage> &ARenderTarget::GetAttachment( std::string const &aKey )
    {
        //
        return mAttachments[aKey];
    }

    bool ARenderTarget::BeginRender() { return true; }

    void ARenderTarget::EndRender() {}

    void ARenderTarget::Present() {}

    uint32_t ARenderTarget::GetCurrentImage() { return 0; };

    Ref<sVkFramebufferObject> ARenderTarget::GetFramebuffer() { return mFramebufferObject; }

    VkSemaphore ARenderTarget::GetImageAvailableSemaphore( uint32_t i ) { return VK_NULL_HANDLE; }

    VkSemaphore ARenderTarget::GetRenderFinishedSemaphore( uint32_t i ) { return VK_NULL_HANDLE; }

    VkFence ARenderTarget::GetInFlightFence( uint32_t i ) { return VK_NULL_HANDLE; }

    void ARenderTarget::InitializeCommandBuffers()
    {
        auto lCommandBuffers = mGraphicContext.mContext->AllocateCommandBuffer( GetImageCount() );

        mCommandBufferObject = {};

        for( auto &lCB : lCommandBuffers )
            mCommandBufferObject.push_back( New<sVkCommandBufferObject>( mGraphicContext.mContext, lCB ) );

        for( size_t i = 0; i < GetImageCount(); i++ )
        {
            auto lImageAvailableSemaphore = GetImageAvailableSemaphore( i );
            if( lImageAvailableSemaphore )
                mCommandBufferObject[i]->AddWaitSemaphore( lImageAvailableSemaphore, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT );

            auto lRenderFinishedSemaphore = GetRenderFinishedSemaphore( i );
            if( lRenderFinishedSemaphore ) mCommandBufferObject[i]->AddSignalSemaphore( lRenderFinishedSemaphore );

            auto lSubmitFence = GetInFlightFence( i );
            if( lSubmitFence ) mCommandBufferObject[i]->SetSubmitFence( lSubmitFence );
        }
    }

} // namespace LTSE::Graphics
