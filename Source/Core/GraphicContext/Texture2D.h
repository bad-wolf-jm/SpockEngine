#pragma once

#include <memory>

#include <gli/gli.hpp>
#include <vulkan/vulkan.h>

#include "Core/CUDA/Texture/TextureData.h"
#include "Core/Memory.h"
#include "Core/Types.h"

#include "Core/CUDA/Texture/ColorFormat.h"
#include "Core/CUDA/Texture/TextureTypes.h"

#include "Buffer.h"
#include "GraphicContext.h"

namespace SE::Graphics
{

    using namespace SE::Core;

    /** @brief */
    enum class TextureUsageFlags : uint32_t
    {
        TRANSFER_SOURCE                  = VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        TRANSFER_DESTINATION             = VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        SAMPLED                          = VK_IMAGE_USAGE_SAMPLED_BIT,
        STORAGE                          = VK_IMAGE_USAGE_STORAGE_BIT,
        COLOR_ATTACHMENT                 = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        DEPTH_STENCIL_ATTACHMENT         = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
        TRANSIENT_ATTACHMENT             = VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT,
        INPUT_ATTACHMENT                 = VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT,
        FRAGMENT_DENSITY_MAP             = VK_IMAGE_USAGE_FRAGMENT_DENSITY_MAP_BIT_EXT,
        FRAGMENT_SHADING_RATE_ATTACHMENT = VK_IMAGE_USAGE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR,
    };

    /* @brief */
    using TextureUsage = EnumSet<TextureUsageFlags, 0x000001ff>;

    /** @brief */
    enum class TextureAspectMaskBits : uint32_t
    {
        COLOR   = VK_IMAGE_ASPECT_COLOR_BIT,
        DEPTH   = VK_IMAGE_ASPECT_DEPTH_BIT,
        STENCIL = VK_IMAGE_ASPECT_STENCIL_BIT
    };

    /* @brief */
    using TextureAspectMask = EnumSet<TextureAspectMaskBits, 0x000001ff>;

    struct Mip
    {
        uint32_t Width    = 0;
        uint32_t Height   = 0;
        uint32_t MipLevel = 1;
        size_t   Size     = 0;
    };

    /** @brief */
    struct TextureDescription
    {
        bool                   ExternalImage       = false; /**!< */
        bool                   IsHostVisible       = false;
        bool                   IsCudaVisible       = false;
        TextureUsage           Usage               = { TextureUsageFlags::SAMPLED };
        TextureAspectMask      AspectMask          = { TextureAspectMaskBits::COLOR };
        VkFilter               MinificationFilter  = VK_FILTER_LINEAR;
        VkFilter               MagnificationFilter = VK_FILTER_LINEAR;
        VkSamplerMipmapMode    MipmapMode          = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        VkSamplerAddressMode   WrappingMode        = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        eColorFormat           Format              = eColorFormat::RGBA8_UNORM;
        std::vector<Mip>       MipLevels           = {};
        bool                   Sampled             = false;
        uint8_t                SampleCount         = 1;
        // std::array<Swizzle, 4> ComponentSwizzle    = { Swizzle::IDENTITY, Swizzle::IDENTITY, Swizzle::IDENTITY,
        //                                                Swizzle::IDENTITY }; /**!< */

        TextureDescription()                       = default;
        TextureDescription( TextureDescription & ) = default;
    };

    /** @brief */
    struct TextureData
    {
        size_t ByteSize; /**!< */
        void  *Data;     /**!< */

        TextureData()                = default;
        TextureData( TextureData & ) = default;
    };

    /** @brief */
    class Texture2D
    {
      public:
        TextureDescription Spec; /**!< */
        Core::TextureData::sCreateInfo mSpec;
        sTextureSamplingInfo mSamplingSpec; /**!< */

        /** @brief */
        Texture2D( GraphicContext &aGraphicContext, TextureDescription &aBufferDescription, VkImage aImage );

        /** @brief */
        Texture2D( GraphicContext &aGraphicContext, TextureDescription &aBufferDescription, TextureData &aBufferData );

        /** @brief */
        Texture2D( GraphicContext &aGraphicContext, TextureData2D &aCubeMapData, TextureSampler2D &aSamplingInfo,
                   bool aCudaVisible = true );

        /** @brief */
        Texture2D( GraphicContext &aGraphicContext, TextureDescription &aBufferDescription );

        /** @brief */
        Texture2D( GraphicContext &aGraphicContext, TextureDescription &aBufferDescription,
                   Ref<Internal::sVkFramebufferImage> a_FramebufferImage );

        /** @brief */
        ~Texture2D() = default;

        TextureData GetImageData();

        /** @brief */
        inline VkImageView GetImageView() { return mTextureView->mVkObject; }

        /** @brief */
        inline VkImage GetImage() { return mTextureImageObject->mVkObject; }

        /** @brief */
        inline Ref<Internal::sVkImageObject> GetImageObject() { return mTextureImageObject; }

        /** @brief */
        inline VkSampler GetSampler() { return mTextureSamplerObject->mVkObject; }

        void                 GetTextureData( TextureData2D &aTextureData );
        sTextureSamplingInfo GetTextureSampling();

        void  *GetMemoryHandle() { return mGraphicContext.mContext->GetSharedMemoryHandle( mTextureImageObject->mVkMemory ); }
        size_t GetMemorySize() { return mTextureImageObject->GetMemorySize(); }

        void CopyBufferToImage( Buffer &a_Buffer );
        void TransitionImageLayout( VkImageLayout oldLayout, VkImageLayout newLayout );

      private:
        void CreateImageView();
        void CreateImageSampler();

      private:
        GraphicContext mGraphicContext{};

        Ref<Internal::sVkImageObject>        mTextureImageObject   = nullptr;
        Ref<Internal::sVkImageSamplerObject> mTextureSamplerObject = nullptr;
        Ref<Internal::sVkImageViewObject>    mTextureView          = nullptr;
    };
} // namespace SE::Graphics
