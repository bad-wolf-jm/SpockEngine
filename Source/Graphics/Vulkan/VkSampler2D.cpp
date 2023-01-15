#include "VkSampler2D.h"

#include "Core/Core.h"
#include "Core/Memory.h"
#include "Graphics/Vulkan/VkCoreMacros.h"

#include "Core/Logging.h"

namespace SE::Graphics
{
    static VkFilter Convert( eSamplerFilter aValue )
    {
        switch( aValue )
        {
        case eSamplerFilter::NEAREST: return VK_FILTER_NEAREST;
        case eSamplerFilter::LINEAR:
        default: return VK_FILTER_LINEAR;
        }
    }

    static VkSamplerMipmapMode Convert( eSamplerMipmap aValue )
    {
        switch( aValue )
        {
        case eSamplerMipmap::NEAREST: return VK_SAMPLER_MIPMAP_MODE_NEAREST;
        case eSamplerMipmap::LINEAR:
        default: return VK_SAMPLER_MIPMAP_MODE_LINEAR;
        }
    }

    static VkSamplerAddressMode Convert( eSamplerWrapping aValue )
    {
        switch( aValue )
        {
        case eSamplerWrapping::MIRRORED_REPEAT: return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
        case eSamplerWrapping::CLAMP_TO_EDGE: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        case eSamplerWrapping::CLAMP_TO_BORDER: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        case eSamplerWrapping::MIRROR_CLAMP_TO_BORDER: return VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE;
        case eSamplerWrapping::REPEAT:
        default: return VK_SAMPLER_ADDRESS_MODE_REPEAT;
        }
    }

    /** @brief */
    VkSampler2D::VkSampler2D( Ref<VkGraphicContext> aGraphicContext, Ref<VkTexture2D> aTextureData,
                              sTextureSamplingInfo const &aSamplingSpec )
        : ISampler2D( aGraphicContext, aTextureData, aSamplingSpec )
    {
        constexpr VkComponentMapping lSwizzles{ VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                                                VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY };

        VkImageAspectFlags lImageAspect = 0;
        if( mTextureData->mSpec.mIsDepthTexture )
            lImageAspect |= VK_IMAGE_ASPECT_DEPTH_BIT;
        else
            lImageAspect |= VK_IMAGE_ASPECT_COLOR_BIT;

        mVkImageView =
            std::reinterpret_pointer_cast<VkGraphicContext>( mGraphicContext )
                ->CreateImageView( std::reinterpret_pointer_cast<VkTexture2D>( mTextureData )->mVkImage, mTextureData->mSpec.mLayers,
                                   VK_IMAGE_VIEW_TYPE_2D, ToVkFormat( mTextureData->mSpec.mFormat ), lImageAspect, lSwizzles );

        mVkImageSampler = std::reinterpret_pointer_cast<VkGraphicContext>( mGraphicContext )
                              ->CreateSampler( Convert( mSpec.mFilter ), Convert( mSpec.mFilter ), Convert( mSpec.mWrapping ),
                                               Convert( mSpec.mMipFilter ) );

        if( mTextureData->mIsGraphicsOnly ) return;

        InitializeTextureSampler();
    }

    VkSampler2D::~VkSampler2D()
    {
        std::reinterpret_pointer_cast<VkGraphicContext>( mGraphicContext )->DestroySampler( mVkImageSampler );
        std::reinterpret_pointer_cast<VkGraphicContext>( mGraphicContext )->DestroyImageView( mVkImageView );
    }

} // namespace SE::Graphics
