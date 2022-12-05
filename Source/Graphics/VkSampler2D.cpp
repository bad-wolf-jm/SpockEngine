#include "VkSampler2D.h"
#include "Buffer.h"
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
        , mSamplingSpec{ aSamplingSpec }
    {
        // Spec.MinificationFilter  = Convert( aSamplingInfo.mSamplingSpec.mFilter );
        // Spec.MagnificationFilter = Convert( aSamplingInfo.mSamplingSpec.mFilter );
        // Spec.MipmapMode          = Convert( aSamplingInfo.mSamplingSpec.mMipFilter );
        // Spec.WrappingMode        = Convert( aSamplingInfo.mSamplingSpec.mWrapping );
        // Spec.Format              = aCubeMapData.mSpec.mFormat;
        // Spec.Sampled             = true;
        // Spec.IsCudaVisible       = aCudaVisible;
        // Spec.SampleCount         = 1;
        // Spec.Usage = { TextureUsageFlags::SAMPLED, TextureUsageFlags::TRANSFER_SOURCE, TextureUsageFlags::TRANSFER_DESTINATION };

        mVkImageView = mGraphicContext.mContext->CreateImageView(
            aImageObject->mVkObject, aLayerCount, VK_IMAGE_VIEW_TYPE_2D, aImageFormat, aAspectMask,
            VkComponentMapping{ VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                                VK_COMPONENT_SWIZZLE_IDENTITY } );
        mVkImageSampler = mGraphicContext.mContext->CreateSampler(
            Convert( aSamplingInfo.mSamplingSpec.mFilter ), Convert( aSamplingInfo.mSamplingSpec.mFilter ),
            Convert( aSamplingInfo.mSamplingSpec.mWrapping ), Convert( aSamplingInfo.mSamplingSpec.mMipFilter ) );

        // mTextureView = New<sVkImageViewObject>( mGraphicContext.mContext, mTextureImageObject, 1, VK_IMAGE_VIEW_TYPE_2D,
        //                                         ToVkFormat( Spec.Format ), (VkImageAspectFlags)Spec.AspectMask,
        //                                         VkComponentMapping{ VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
        //                                                             VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY }
        //                                                             );

        // mTextureSamplerObject = New<sVkImageSamplerObject>( mGraphicContext.mContext, Spec.MinificationFilter,
        //                                                     Spec.MagnificationFilter, Spec.WrappingMode, Spec.MipmapMode );
    }

} // namespace SE::Graphics
