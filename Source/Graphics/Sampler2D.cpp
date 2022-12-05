#include "Sampler2D.h"
#include "Buffer.h"
#include "Core/Core.h"
#include "Core/Memory.h"
#include "Core/Vulkan/VkCoreMacros.h"

#include "Core/Logging.h"

namespace SE::Graphics
{
    using namespace Internal;

    Sampler2D::Sampler2D( GraphicContext &aGraphicContext, TextureDescription &aBufferDescription, TextureData &a_BufferData )
        : mGraphicContext( aGraphicContext )
        , Spec( aBufferDescription )
    {
        Buffer lStagingBuffer( mGraphicContext, reinterpret_cast<uint8_t *>( a_BufferData.Data ), a_BufferData.ByteSize,
                                eBufferBindType::UNKNOWN, true, false, true, false );

        mTextureImageObject =
            New<sVkImageObject>( mGraphicContext.mContext, static_cast<uint32_t>( Spec.MipLevels[0].Width ),
                                 static_cast<uint32_t>( Spec.MipLevels[0].Height ), 1, static_cast<uint32_t>( Spec.MipLevels.size() ),
                                 1, VK_SAMPLE_COUNT_VALUE( aBufferDescription.SampleCount ), aBufferDescription.IsCudaVisible, false,
                                 ToVkFormat( Spec.Format ), ToVkMemoryFlag( Spec ), (VkImageUsageFlags)Spec.Usage );

        TransitionImageLayout( VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL );
        CopyBufferToImage( lStagingBuffer );
        TransitionImageLayout( VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL );

        CreateImageView();
        CreateImageSampler();
    }

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
    Sampler2D::Sampler2D( GraphicContext &aGraphicContext, TextureData2D &aCubeMapData, TextureSampler2D &aSamplingInfo,
                          bool aCudaVisible )
        : mGraphicContext( aGraphicContext )
        , mSpec{ aCubeMapData.mSpec }
        , mSamplingSpec{ aSamplingInfo.mSamplingSpec }
    {
        Spec.MinificationFilter  = Convert( aSamplingInfo.mSamplingSpec.mFilter );
        Spec.MagnificationFilter = Convert( aSamplingInfo.mSamplingSpec.mFilter );
        Spec.MipmapMode          = Convert( aSamplingInfo.mSamplingSpec.mMipFilter );
        Spec.WrappingMode        = Convert( aSamplingInfo.mSamplingSpec.mWrapping );
        Spec.Format              = aCubeMapData.mSpec.mFormat;
        Spec.Sampled             = true;
        Spec.IsCudaVisible       = aCudaVisible;
        Spec.SampleCount         = 1;
        Spec.Usage = { TextureUsageFlags::SAMPLED, TextureUsageFlags::TRANSFER_SOURCE, TextureUsageFlags::TRANSFER_DESTINATION };

        sImageData &lImageData = aCubeMapData.GetImageData();
        Buffer lStagingBuffer( mGraphicContext, lImageData.mPixelData, lImageData.mByteSize, eBufferBindType::UNKNOWN, true, false,
                                true, false );

        Spec.MipLevels = { { static_cast<uint32_t>( lImageData.mWidth ), static_cast<uint32_t>( lImageData.mHeight ), 0, 0 } };
        Spec.Format    = lImageData.mFormat;

        mTextureImageObject =
            New<sVkImageObject>( mGraphicContext.mContext, static_cast<uint32_t>( Spec.MipLevels[0].Width ),
                                 static_cast<uint32_t>( Spec.MipLevels[0].Height ), 1, static_cast<uint32_t>( Spec.MipLevels.size() ),
                                 1, VK_SAMPLE_COUNT_VALUE( Spec.SampleCount ), Spec.IsCudaVisible, false, ToVkFormat( Spec.Format ),
                                 ToVkMemoryFlag( Spec ), (VkImageUsageFlags)Spec.Usage );

        TransitionImageLayout( VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL );
        CopyBufferToImage( lStagingBuffer );
        TransitionImageLayout( VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL );

        mTextureSamplerObject = New<sVkImageSamplerObject>( mGraphicContext.mContext, Spec.MinificationFilter,
                                                            Spec.MagnificationFilter, Spec.WrappingMode, Spec.MipmapMode );
    }

} // namespace SE::Graphics
