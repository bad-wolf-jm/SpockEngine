#include "VkSampler2D.h"

#include "Core/Core.h"
#include "Core/Memory.h"
#include "Core/Vulkan/VkCoreMacros.h"

#include "Core/Logging.h"

namespace SE::Graphics
{
    using namespace Internal;

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
    VkSampler2D::VkSampler2D( GraphicContext &aGraphicContext, Ref<VkTexture2D> aTextureData,
                              sTextureSamplingInfo const &aSamplingSpec )
        : mGraphicContext( aGraphicContext )
        , mTextureData{ aTextureData }
    {
        mSpec = aSamplingSpec;
        
        constexpr VkComponentMapping lSwizzles{ VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                                                VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY };

        mVkImageView = mGraphicContext.mContext->CreateImageView( mTextureData->mVkImage, mTextureData->mSpec.mLayers,
                                                                  VK_IMAGE_VIEW_TYPE_2D, ToVkFormat( mTextureData->mSpec.mFormat ),
                                                                  VK_IMAGE_ASPECT_COLOR_BIT, lSwizzles );

        mVkImageSampler =
            mGraphicContext.mContext->CreateSampler( Convert( mSpec.mFilter ), Convert( mSpec.mFilter ),
                                                     Convert( mSpec.mWrapping ), Convert( mSpec.mMipFilter ) );

        if( mTextureData->mIsGraphicsOnly ) return;

        InitializeTextureSampler();
    }

} // namespace SE::Graphics
