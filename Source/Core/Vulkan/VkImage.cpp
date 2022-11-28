#include "VkImage.h"

#include <set>
#include <unordered_set>

#include "VkCoreMacros.h"

namespace SE::Graphics::Internal
{

    sVkImageObject::sVkImageObject( Ref<VkContext> aContext, VkImage aExternalImage )
        : mVkObject{ aExternalImage }
        , mVkMemory{ VK_NULL_HANDLE }
        , mContext{ aContext }
        , mExternal{ true }
    {
    }

    sVkImageObject::sVkImageObject( Ref<VkContext> aContext, uint32_t aWidth, uint32_t aHeight, uint32_t aDepth, uint32_t aMipLevels,
                                    uint32_t aLayers, uint8_t aSampleCount, bool aCudaCompatible, bool aCubeCompatible,
                                    VkFormat aFormat, VkMemoryPropertyFlags aProperties, VkImageUsageFlags aUsage )
        : mContext{ aContext }
        , mExternal{ false }
    {
        mVkObject = mContext->CreateImage( aWidth, aHeight, aDepth, aMipLevels, aLayers, aSampleCount, aCubeCompatible, aFormat,
                                           aProperties, aUsage );
        mVkMemory = mContext->AllocateMemory( mVkObject, 0, false, aCudaCompatible );

        mContext->BindMemory( mVkObject, mVkMemory );
    }

    sVkImageObject::~sVkImageObject()
    {
        if( !mExternal ) mContext->DestroyImage( mVkObject );

        mContext->FreeMemory( mVkMemory );
    }

    sVkImageSamplerObject::sVkImageSamplerObject( Ref<VkContext> aContext, VkFilter aMinificationFilter, VkFilter aMagnificationFilter,
                                                  VkSamplerAddressMode aWrappingMode, VkSamplerMipmapMode aMipmapMode )
        : mContext{ aContext }
    {
        mVkObject = mContext->CreateSampler( aMinificationFilter, aMagnificationFilter, aWrappingMode, aMipmapMode );
    }

    sVkImageSamplerObject::~sVkImageSamplerObject() { mContext->DestroySampler( mVkObject ); }

    sVkImageViewObject::sVkImageViewObject( Ref<VkContext> aContext, Ref<sVkImageObject> aImageObject, uint32_t aLayerCount,
                                            VkImageViewType aViewType, VkFormat aImageFormat, VkImageAspectFlags aAspectMask,
                                            VkComponentMapping aComponentSwizzle )
        : mContext{ aContext }
        , mImageObject{ aImageObject }
    {
        mVkObject =
            mContext->CreateImageView( aImageObject->mVkObject, aLayerCount, aViewType, aImageFormat, aAspectMask, aComponentSwizzle );
    }

    sVkImageViewObject::~sVkImageViewObject() { mContext->DestroyImageView( mVkObject ); }

    sVkFramebufferImage::sVkFramebufferImage( Ref<VkContext> aContext, VkFormat aFormat, uint32_t aWidth, uint32_t aHeight,
                                              uint32_t aSampleCount, VkImageUsageFlags aUsage, bool aIsSampled )
        : mContext{ aContext }
    {
        VkImageUsageFlags lUsage = aUsage;
        if( aIsSampled ) lUsage |= VK_IMAGE_USAGE_SAMPLED_BIT;

        mImage = New<sVkImageObject>( mContext, aWidth, aHeight, 1, 1, 1, aSampleCount, false, false, aFormat,
                                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, lUsage );

        VkComponentMapping lSwizzle{ VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                                     VK_COMPONENT_SWIZZLE_IDENTITY };
        VkImageAspectFlags lImageAspect = 0;

        if( aUsage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT ) lImageAspect |= VK_IMAGE_ASPECT_DEPTH_BIT;

        if( aUsage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT ) lImageAspect |= VK_IMAGE_ASPECT_COLOR_BIT;

        mImageView = New<sVkImageViewObject>( mContext, mImage, 1, VK_IMAGE_VIEW_TYPE_2D, aFormat, lImageAspect, lSwizzle );

        if( aIsSampled )
            mImageSampler = New<sVkImageSamplerObject>( mContext, VK_FILTER_LINEAR, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_REPEAT,
                                                        VK_SAMPLER_MIPMAP_MODE_LINEAR );
    }

    sVkFramebufferImage::sVkFramebufferImage( Ref<VkContext> aContext, VkImage aImage, VkFormat aFormat, uint32_t aWidth,
                                              uint32_t aHeight, uint32_t aSampleCount, VkImageUsageFlags aUsage, bool aIsSampled )
        : mContext{ aContext }
    {
        VkImageUsageFlags lUsage = aUsage;
        if( aIsSampled ) lUsage |= VK_IMAGE_USAGE_SAMPLED_BIT;

        mImage = New<sVkImageObject>( mContext, aImage );

        VkComponentMapping lSwizzle{ VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                                     VK_COMPONENT_SWIZZLE_IDENTITY };
        VkImageAspectFlags lImageAspect = 0;

        if( aUsage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT ) lImageAspect |= VK_IMAGE_ASPECT_DEPTH_BIT;

        if( aUsage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT ) lImageAspect |= VK_IMAGE_ASPECT_COLOR_BIT;

        mImageView = New<sVkImageViewObject>( mContext, mImage, 1, VK_IMAGE_VIEW_TYPE_2D, aFormat, lImageAspect, lSwizzle );

        if( aIsSampled )
            mImageSampler = New<sVkImageSamplerObject>( mContext, VK_FILTER_LINEAR, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_REPEAT,
                                                        VK_SAMPLER_MIPMAP_MODE_LINEAR );
    }

    sVkFramebufferObject::sVkFramebufferObject( Ref<VkContext> aContext, uint32_t aWidth, uint32_t aHeight, uint32_t aLayers,
                                                VkRenderPass aRenderPass, std::vector<Ref<sVkFramebufferImage>> aAttachments )
        : mContext{ aContext }
    {

        std::vector<VkImageView> lAttachmentViews( aAttachments.size() );

        for( uint32_t i = 0; i < aAttachments.size(); i++ ) lAttachmentViews[i] = aAttachments[i]->mImageView->mVkObject;

        mVkObject = mContext->CreateFramebuffer( lAttachmentViews, aWidth, aHeight, aLayers, aRenderPass );
    }

    sVkFramebufferObject::~sVkFramebufferObject() { mContext->DestroyFramebuffer( mVkObject ); }
} // namespace SE::Graphics::Internal