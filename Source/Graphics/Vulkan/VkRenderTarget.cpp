#include "VkRenderTarget.h"

namespace SE::Graphics
{
    VkRenderTarget::VkRenderTarget( ref_t<VkGraphicContext> aGraphicContext, sRenderTargetDescription const &aRenderTargetDescription )
        : IRenderTarget( aGraphicContext, aRenderTargetDescription )
    {
    }

    VkRenderTarget::~VkRenderTarget()
    {
        GraphicContext<VkGraphicContext>()->DestroyFramebuffer( mVkFramebuffer );

        for( auto &lImageView : mVkImageViews )
            GraphicContext<VkGraphicContext>()->DestroyImageView( lImageView );
    }

    vector_t<VkClearValue> VkRenderTarget::GetClearValues()
    {
        vector_t<VkClearValue> lValues;
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

    void VkRenderTarget::AddAttachment( string_t const &aAttachmentID, sAttachmentDescription const &aCreateInfo )
    {
        sTextureCreateInfo lTextureCreateInfo{};
        lTextureCreateInfo.mFormat         = aCreateInfo.mFormat;
        lTextureCreateInfo.mWidth          = mSpec.mWidth;
        lTextureCreateInfo.mHeight         = mSpec.mHeight;
        lTextureCreateInfo.mDepth          = 1;
        lTextureCreateInfo.mMipLevels      = 1;
        lTextureCreateInfo.mIsDepthTexture = ( aCreateInfo.mType == eAttachmentType::DEPTH );
        uint32_t lSampleCount              = ( aCreateInfo.mType == eAttachmentType::MSAA_RESOLVE ) ? 1 : mSpec.mSampleCount;

        auto lNewAttachment =
            New<VkTexture2D>( GraphicContext<VkGraphicContext>(), lTextureCreateInfo, lSampleCount, false, true, true, true );
        IRenderTarget::AddAttachment( aAttachmentID, aCreateInfo, lNewAttachment );
    }

    void VkRenderTarget::AddAttachment( string_t const &aAttachmentID, sAttachmentDescription const &aCreateInfo,
                                        ref_t<VkTexture2D> aFramebufferImage )
    {
        IRenderTarget::AddAttachment( aAttachmentID, aCreateInfo, aFramebufferImage );
    }

    void VkRenderTarget::AddAttachment( string_t const &aAttachmentID, sAttachmentDescription const &aCreateInfo,
                                        ref_t<VkTexture2D> aFramebufferImage, eCubeFace aFace )
    {
        IRenderTarget::AddAttachment( aAttachmentID, aCreateInfo, aFramebufferImage, aFace );
    }

    void VkRenderTarget::AddAttachment( string_t const &aAttachmentID, eAttachmentType aType, eColorFormat aFormat,
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
    }

    void VkRenderTarget::AddAttachment( string_t const &aAttachmentID, eAttachmentType aType, eColorFormat aFormat,
                                        math::vec4 aClearColor, bool aIsSampled, bool aIsPresented, eAttachmentLoadOp aLoadOp,
                                        eAttachmentStoreOp eStoreOp, ref_t<VkTexture2D> aFramebufferImage )
    {
        IRenderTarget::AddAttachment( aAttachmentID, aType, aFormat, aClearColor, aIsSampled, aIsPresented, aLoadOp, eStoreOp,
                                      aFramebufferImage );
    }

    void VkRenderTarget::Finalize()
    {
        vector_t<sAttachmentResource> lAttachments{};
        for( auto const &lAttachmentID : mAttachmentIDs )
        {
            auto lAttachment = mAttachments[lAttachmentID];

            lAttachments.push_back( lAttachment );
        }

        constexpr VkComponentMapping lSwizzles{ VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                                                VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY };
        for( auto lTextureData : lAttachments )
        {
            auto lVkTextureData = std::static_pointer_cast<VkTexture2D>( lTextureData.mTexture );

            VkImageAspectFlags lImageAspect = 0;
            if( lVkTextureData->mSpec.mIsDepthTexture )
                lImageAspect |= VK_IMAGE_ASPECT_DEPTH_BIT;
            else
                lImageAspect |= VK_IMAGE_ASPECT_COLOR_BIT;

            auto lVkImageView = GraphicContext<VkGraphicContext>()->CreateImageView(
                lVkTextureData->mVkImage, (uint32_t)lTextureData.mFace, 1, VK_IMAGE_VIEW_TYPE_2D,
                ToVkFormat( lVkTextureData->mSpec.mFormat ), lImageAspect, lSwizzles );

            mVkImageViews.push_back( lVkImageView );
        }

        mRenderPassObject = CreateDefaultRenderPass();
        mVkFramebuffer    = GraphicContext<VkGraphicContext>()->CreateFramebuffer( mVkImageViews, mSpec.mWidth, mSpec.mHeight, 1,
                                                                                   mRenderPassObject->mVkObject );

        InitializeCommandBuffers();
    }

    VkAttachmentLoadOp ToVkLoadOp( eAttachmentLoadOp aOp )
    {
        switch( aOp )
        {
        case eAttachmentLoadOp::CLEAR:
            return VK_ATTACHMENT_LOAD_OP_CLEAR;
        case eAttachmentLoadOp::LOAD:
            return VK_ATTACHMENT_LOAD_OP_LOAD;
        case eAttachmentLoadOp::UNSPECIFIED:
        default:
            return VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        }
    }

    VkAttachmentStoreOp ToVkStoreOp( eAttachmentStoreOp aOp )
    {
        switch( aOp )
        {
        case eAttachmentStoreOp::STORE:
            return VK_ATTACHMENT_STORE_OP_STORE;
        case eAttachmentStoreOp::UNSPECIFIED:
        default:
            return VK_ATTACHMENT_STORE_OP_DONT_CARE;
        }
    }

    ref_t<VkRenderPassObject> VkRenderTarget::CreateDefaultRenderPass()
    {
        ref_t<VkRenderPassObject> lNewRenderPass = New<VkRenderPassObject>( GraphicContext<VkGraphicContext>(), VK_FORMAT_UNDEFINED,
                                                                          mSpec.mSampleCount, false, false, math::vec4( 0.0f ) );

        vector_t<VkAttachmentDescription> lAttachmentDescriptions{};
        vector_t<VkAttachmentReference>   lColorAttachmentReferences{};

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
                                                                mAttachmentInfo[i].mIsSampled, mAttachmentInfo[i].mIsPresented,
                                                                mAttachmentInfo[i].mIsDefined, lAttachmentLoadOp, lAttachmentStoreOp );
                lAttachmentDescriptions.push_back( lDescription );
                VkAttachmentReference lAttachmentReference{};
                lAttachmentReference.attachment = i;
                lAttachmentReference.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                lColorAttachmentReferences.push_back( lAttachmentReference );
                break;
            }
            case eAttachmentType::MSAA_RESOLVE:
            {
                lDescription = lNewRenderPass->ColorAttachment( ToVkFormat( mAttachmentInfo[i].mFormat ), 1,
                                                                mAttachmentInfo[i].mIsSampled, mAttachmentInfo[i].mIsPresented,
                                                                mAttachmentInfo[i].mIsDefined, lAttachmentLoadOp, lAttachmentStoreOp );
                lAttachmentDescriptions.push_back( lDescription );
                lResolveAttachment.attachment = i;
                lResolveAttachment.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                lResolveAttachmentPtr         = &lResolveAttachment;
                break;
            }
            case eAttachmentType::DEPTH:
            {
                lDescription = lNewRenderPass->DepthAttachment( mAttachmentInfo[i].mIsDefined, mSpec.mSampleCount,
                                                                mAttachmentInfo[i].mIsSampled, lAttachmentLoadOp, lAttachmentStoreOp );
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

        lNewRenderPass->CreateUnderlyingRenderpass( lAttachmentDescriptions, lColorAttachmentReferences, lDepthAttachmentPtr,
                                                    lResolveAttachmentPtr );

        return lNewRenderPass;
    }

    ref_t<VkTexture2D> VkRenderTarget::GetAttachment( string_t const &aKey )
    {
        //
        return std::static_pointer_cast<VkTexture2D>( mAttachments[aKey].mTexture );
    }

    bool VkRenderTarget::BeginRender()
    {
        return true;
    }

    void VkRenderTarget::EndRender()
    {
    }

    void VkRenderTarget::Present()
    {
    }

    uint32_t VkRenderTarget::GetCurrentImage()
    {
        return 0;
    };

    VkFramebuffer VkRenderTarget::GetFramebuffer()
    {
        return mVkFramebuffer;
    }

    VkSemaphore VkRenderTarget::GetImageAvailableSemaphore( uint32_t i )
    {
        return VK_NULL_HANDLE;
    }

    VkSemaphore VkRenderTarget::GetRenderFinishedSemaphore( uint32_t i )
    {
        return VK_NULL_HANDLE;
    }

    VkFence VkRenderTarget::GetInFlightFence( uint32_t i )
    {
        return VK_NULL_HANDLE;
    }

    void VkRenderTarget::InitializeCommandBuffers()
    {
        auto lCommandBuffers = GraphicContext<VkGraphicContext>()->AllocateCommandBuffer( GetImageCount() );

        mCommandBufferObject = {};

        for( auto &lCB : lCommandBuffers )
            mCommandBufferObject.push_back( New<sVkCommandBufferObject>( GraphicContext<VkGraphicContext>(), lCB ) );

        for( size_t i = 0; i < GetImageCount(); i++ )
        {
            auto lImageAvailableSemaphore = GetImageAvailableSemaphore( i );
            if( lImageAvailableSemaphore )
                mCommandBufferObject[i]->AddWaitSemaphore( lImageAvailableSemaphore, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT );

            auto lRenderFinishedSemaphore = GetRenderFinishedSemaphore( i );
            if( lRenderFinishedSemaphore )
                mCommandBufferObject[i]->AddSignalSemaphore( lRenderFinishedSemaphore );

            auto lSubmitFence = GetInFlightFence( i );
            if( lSubmitFence )
                mCommandBufferObject[i]->SetSubmitFence( lSubmitFence );
        }
    }

} // namespace SE::Graphics
