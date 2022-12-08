#pragma once

#include <memory>
#include <vulkan/vulkan.h>

#include "Core/Memory.h"
#include "New.h"
#include "VkContext.h"

namespace SE::Graphics::Internal
{
    using namespace SE::Core;


    struct sVkImageObject
    {
        VkImage        mVkObject = VK_NULL_HANDLE;
        VkDeviceMemory mVkMemory = VK_NULL_HANDLE;

        sVkImageObject()                   = default;
        sVkImageObject( sVkImageObject & ) = default;
        sVkImageObject( Ref<VkContext> aContext, VkImage aExternalImage );
        sVkImageObject( Ref<VkContext> aContext, uint32_t aWidth, uint32_t aHeight, uint32_t aDepth, uint32_t aMipLevels,
                        uint32_t aLayers, uint8_t aSampleCount, bool aCudaCompatible, bool aCubeCompatible, VkFormat aFormat,
                        VkMemoryPropertyFlags aProperties, VkImageUsageFlags aUsage );

        ~sVkImageObject();

        size_t GetMemorySize() { return mMemorySize; }

      private:
        bool           mExternal   = false;
        Ref<VkContext> mContext    = nullptr;
        size_t         mMemorySize = 0;
    };

    struct sVkImageSamplerObject
    {
        VkSampler mVkObject = VK_NULL_HANDLE;

        sVkImageSamplerObject()                          = default;
        sVkImageSamplerObject( sVkImageSamplerObject & ) = default;
        sVkImageSamplerObject( Ref<VkContext> aContext, VkFilter aMinificationFilter, VkFilter aMagnificationFilter,
                               VkSamplerAddressMode aWrappingMode, VkSamplerMipmapMode aMipmapMode );

        ~sVkImageSamplerObject();

      private:
        Ref<VkContext> mContext = nullptr;
    };

    struct sVkImageViewObject
    {
        VkImageView mVkObject = VK_NULL_HANDLE;

        sVkImageViewObject()                       = default;
        sVkImageViewObject( sVkImageViewObject & ) = default;
        sVkImageViewObject( Ref<VkContext> aContext, Ref<sVkImageObject> aImageObject, uint32_t aLayerCount, VkImageViewType aViewType,
                            VkFormat aImageFormat, VkImageAspectFlags aAspectMask, VkComponentMapping aComponentSwizzle );

        ~sVkImageViewObject();

      private:
        Ref<VkContext>      mContext     = nullptr;
        Ref<sVkImageObject> mImageObject = nullptr;
    };

    struct sVkFramebufferImage
    {
        Ref<sVkImageObject>        mImage        = nullptr;
        Ref<sVkImageViewObject>    mImageView    = nullptr;
        Ref<sVkImageSamplerObject> mImageSampler = nullptr;

        sVkFramebufferImage( Ref<VkContext> aContext, VkFormat aFormat, uint32_t aWidth, uint32_t aHeight, uint32_t aSampleCount,
                             VkImageUsageFlags aUsage, bool aIsSampled );
        sVkFramebufferImage( Ref<VkContext> aContext, VkImage aImage, VkFormat aFormat, uint32_t aWidth, uint32_t aHeight,
                             uint32_t aSampleCount, VkImageUsageFlags aUsage, bool aIsSampled );

        ~sVkFramebufferImage() = default;

      private:
        Ref<VkContext> mContext = nullptr;
    };

    struct sVkFramebufferObject
    {
        VkFramebuffer mVkObject = VK_NULL_HANDLE;

        sVkFramebufferObject()                         = default;
        sVkFramebufferObject( sVkFramebufferObject & ) = default;
        sVkFramebufferObject( Ref<VkContext> aContext, uint32_t aWidth, uint32_t aHeight, uint32_t aLayers, VkRenderPass aRenderPass,
                              std::vector<Ref<sVkFramebufferImage>> aAttachments );

        ~sVkFramebufferObject();

      private:
        Ref<VkContext> mContext = nullptr;
    };

} // namespace SE::Graphics::Internal
