#pragma once

#include <memory>

#include <gli/gli.hpp>
#include <vulkan/vulkan.h>

#include "Core/Memory.h"
#include "Core/TextureData.h"
#include "Core/Types.h"

#include "Core/ColorFormat.h"

#include "Buffer.h"
#include "GraphicContext.h"

namespace LTSE::Graphics
{

    using namespace LTSE::Core;

    /** @brief */
    enum class SamplerFilter : uint32_t
    {
        NEAREST = VK_FILTER_NEAREST,
        LINEAR  = VK_FILTER_LINEAR
    };

    /** @brief */
    enum class SamplerMipmap : uint32_t
    {
        NEAREST = VK_SAMPLER_MIPMAP_MODE_NEAREST,
        LINEAR  = VK_SAMPLER_MIPMAP_MODE_LINEAR
    };

    /** @brief */
    enum class SamplerWrapping : uint32_t
    {
        REPEAT                 = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        MIRRORED_REPEAT        = VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT,
        CLAMP_TO_EDGE          = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        CLAMP_TO_BORDER        = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
        MIRROR_CLAMP_TO_BORDER = VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE
    };

    /** @brief */
    enum class Swizzle : uint32_t
    {
        IDENTITY = VK_COMPONENT_SWIZZLE_IDENTITY,
        ZERO     = VK_COMPONENT_SWIZZLE_ZERO,
        ONE      = VK_COMPONENT_SWIZZLE_ONE,
        R        = VK_COMPONENT_SWIZZLE_R,
        G        = VK_COMPONENT_SWIZZLE_G,
        B        = VK_COMPONENT_SWIZZLE_B,
        A        = VK_COMPONENT_SWIZZLE_A
    };

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
        size_t Size       = 0;
    };

    /** @brief */
    struct TextureDescription
    {
        bool ExternalImage                      = false; /**!< */
        bool IsHostVisible                      = false;
        TextureUsage Usage                      = { TextureUsageFlags::SAMPLED };
        TextureAspectMask AspectMask            = { TextureAspectMaskBits::COLOR };
        SamplerFilter MinificationFilter        = SamplerFilter::LINEAR;
        SamplerFilter MagnificationFilter       = SamplerFilter::LINEAR;
        SamplerMipmap MipmapMode                = SamplerMipmap::LINEAR;
        SamplerWrapping WrappingMode            = SamplerWrapping::CLAMP_TO_BORDER;
        eColorFormat Format                     = eColorFormat::RGBA8_UNORM;
        std::vector<Mip> MipLevels              = {};
        bool Sampled                            = false;
        uint8_t SampleCount                     = 1;
        std::array<Swizzle, 4> ComponentSwizzle = { Swizzle::IDENTITY, Swizzle::IDENTITY, Swizzle::IDENTITY, Swizzle::IDENTITY }; /**!< */

        TextureDescription()                       = default;
        TextureDescription( TextureDescription & ) = default;
    };

    /** @brief */
    struct TextureData
    {
        size_t ByteSize; /**!< */
        void *Data;      /**!< */

        TextureData()                = default;
        TextureData( TextureData & ) = default;
    };

    /** @brief */
    class Texture2D
    {
      public:
        TextureDescription Spec; /**!< */

        /** @brief */
        Texture2D( GraphicContext &a_GraphicContext, TextureDescription &a_BufferDescription, VkImage a_Image );

        /** @brief */
        Texture2D( GraphicContext &a_GraphicContext, TextureDescription &a_BufferDescription, TextureData &a_BufferData );

        /** @brief */
        Texture2D( GraphicContext &a_GraphicContext, TextureDescription &a_BufferDescription, sImageData &a_ImageData );

        /** @brief */
        Texture2D( GraphicContext &a_GraphicContext, TextureDescription &a_BufferDescription, gli::texture2d &a_CubeMapData );

        /** @brief */
        Texture2D( GraphicContext &a_GraphicContext, TextureData2D &a_CubeMapData, TextureSampler2D &aSamplingInfo );

        /** @brief */
        Texture2D( GraphicContext &a_GraphicContext, TextureDescription &a_BufferDescription );

        /** @brief */
        Texture2D( GraphicContext &a_GraphicContext, TextureDescription &a_BufferDescription, Ref<Internal::sVkFramebufferImage> a_FramebufferImage );

        /** @brief */
        ~Texture2D() = default;

        TextureData GetImageData();

        /** @brief */
        inline VkImageView GetImageView() { return m_TextureView->mVkObject; }

        /** @brief */
        inline VkImage GetImage() { return m_TextureImageObject->mVkObject; }

        /** @brief */
        inline Ref<Internal::sVkImageObject> GetImageObject() { return m_TextureImageObject; }

        /** @brief */
        inline VkSampler GetSampler() { return m_TextureSamplerObject->mVkObject; }

      private:
        void TransitionImageLayout( VkImageLayout oldLayout, VkImageLayout newLayout );
        void CopyBufferToImage( Buffer &a_Buffer );
        void CreateImageView();
        void CreateImageSampler();

      private:
        GraphicContext mGraphicContext{};

        Ref<Internal::sVkImageObject> m_TextureImageObject          = nullptr;
        Ref<Internal::sVkImageSamplerObject> m_TextureSamplerObject = nullptr;
        Ref<Internal::sVkImageViewObject> m_TextureView             = nullptr;
    };
} // namespace LTSE::Graphics
