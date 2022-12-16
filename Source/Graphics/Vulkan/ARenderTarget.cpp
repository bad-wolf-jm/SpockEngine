#include "ARenderTarget.h"

namespace SE::Graphics
{
    ARenderTarget::ARenderTarget( Ref<VkGraphicContext> aGraphicContext, sRenderTargetDescription const &aRenderTargetDescription )
        : IRenderTarget( aGraphicContext, aRenderTargetDescription )
    // : std::reinterpret_pointer_cast<VkGraphicContext>(mGraphicContext){ aGraphicContext }
    // , mSpec{ aRenderTargetDescription }
    {
        // mImageCount = 1;
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
        sTextureCreateInfo lTextureCreateInfo{};
        lTextureCreateInfo.mFormat         = aCreateInfo.mFormat;
        lTextureCreateInfo.mWidth          = mSpec.mWidth;
        lTextureCreateInfo.mHeight         = mSpec.mHeight;
        lTextureCreateInfo.mDepth          = 1;
        lTextureCreateInfo.mIsDepthTexture = ( aCreateInfo.mType == eAttachmentType::DEPTH );
        uint32_t lSampleCount              = ( aCreateInfo.mType == eAttachmentType::MSAA_RESOLVE ) ? 1 : mSpec.mSampleCount;

        auto lNewAttachment = New<VkTexture2D>( std::reinterpret_pointer_cast<VkGraphicContext>( mGraphicContext ), lTextureCreateInfo,
                                                lSampleCount, false, true, true, true );
        IRenderTarget::AddAttachment( aAttachmentID, aCreateInfo, lNewAttachment );
    }

    void ARenderTarget::AddAttachment( std::string const &aAttachmentID, sAttachmentDescription const &aCreateInfo,
                                       Ref<VkTexture2D> aFramebufferImage )
    {
        IRenderTarget::AddAttachment( aAttachmentID, aCreateInfo, aFramebufferImage );
    }

    void ARenderTarget::AddAttachment( std::string const &aAttachmentID, eAttachmentType aType, eColorFormat aFormat,
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

    void ARenderTarget::AddAttachment( std::string const &aAttachmentID, eAttachmentType aType, eColorFormat aFormat,
                                       math::vec4 aClearColor, bool aIsSampled, bool aIsPresented, eAttachmentLoadOp aLoadOp,
                                       eAttachmentStoreOp eStoreOp, Ref<VkTexture2D> aFramebufferImage )
    {
        IRenderTarget::AddAttachment( aAttachmentID, aType, aFormat, aClearColor, aIsSampled, aIsPresented, aLoadOp, eStoreOp,
                                      aFramebufferImage );
    }

    void ARenderTarget::Finalize()
    {
        std::vector<Ref<VkTexture2D>> lAttachments{};
        for( auto const &lAttachmentID : mAttachmentIDs )
            lAttachments.push_back( std::reinterpret_pointer_cast<VkTexture2D>( mAttachments[lAttachmentID] ) );

        mRenderPassObject  = CreateDefaultRenderPass();
        mFramebufferObject = New<VkRenderTarget>( std::reinterpret_pointer_cast<VkGraphicContext>( mGraphicContext ), mSpec.mWidth,
                                                  mSpec.mHeight, 1, mRenderPassObject->mVkObject, lAttachments );

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
        Ref<sVkAbstractRenderPassObject> lNewRenderPass =
            New<sVkAbstractRenderPassObject>( std::reinterpret_pointer_cast<VkGraphicContext>( mGraphicContext ), VK_FORMAT_UNDEFINED,
                                              mSpec.mSampleCount, false, false, math::vec4( 0.0f ) );

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
                lDescription = lNewRenderPass->DepthAttachment( mAttachmentInfo[i].mIsDefined, mSpec.mSampleCount, lAttachmentLoadOp,
                                                                lAttachmentStoreOp );
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

    Ref<VkTexture2D> ARenderTarget::GetAttachment( std::string const &aKey )
    {
        //
        return std::reinterpret_pointer_cast<VkTexture2D>( mAttachments[aKey] );
    }

    bool ARenderTarget::BeginRender() { return true; }

    void ARenderTarget::EndRender() {}

    void ARenderTarget::Present() {}

    uint32_t ARenderTarget::GetCurrentImage() { return 0; };

    Ref<VkRenderTarget> ARenderTarget::GetFramebuffer() { return mFramebufferObject; }

    VkSemaphore ARenderTarget::GetImageAvailableSemaphore( uint32_t i ) { return VK_NULL_HANDLE; }

    VkSemaphore ARenderTarget::GetRenderFinishedSemaphore( uint32_t i ) { return VK_NULL_HANDLE; }

    VkFence ARenderTarget::GetInFlightFence( uint32_t i ) { return VK_NULL_HANDLE; }

    void ARenderTarget::InitializeCommandBuffers()
    {
        auto lCommandBuffers =
            std::reinterpret_pointer_cast<VkGraphicContext>( mGraphicContext )->AllocateCommandBuffer( GetImageCount() );

        mCommandBufferObject = {};

        for( auto &lCB : lCommandBuffers )
            mCommandBufferObject.push_back(
                New<sVkCommandBufferObject>( std::reinterpret_pointer_cast<VkGraphicContext>( mGraphicContext ), lCB ) );

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

} // namespace SE::Graphics
