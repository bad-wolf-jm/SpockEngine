#include "Texture2D.h"
#include "Buffer.h"
#include "Core/Core.h"
#include "Core/Memory.h"
#include "Core/Vulkan/VkCoreMacros.h"

#include "Core/Logging.h"

namespace SE::Graphics
{

    static VkMemoryPropertyFlags ToVkMemoryFlag( TextureDescription const &aBufferDescription )
    {
        VkMemoryPropertyFlags l_Flags = 0;
        if( aBufferDescription.IsHostVisible )
            l_Flags |= ( VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT );
        else
            l_Flags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

        return l_Flags;
    }

    Texture2D::Texture2D( GraphicContext &aGraphicContext, TextureDescription &aBufferDescription, TextureData &a_BufferData )
        : mGraphicContext( aGraphicContext )
        , Spec( aBufferDescription )
    {
        Buffer l_StagingBuffer( mGraphicContext, reinterpret_cast<uint8_t *>( a_BufferData.Data ), a_BufferData.ByteSize,
                                eBufferBindType::UNKNOWN, true, false, true, false );

        mTextureImageObject = New<Internal::sVkImageObject>(
            mGraphicContext.mContext, static_cast<uint32_t>( Spec.MipLevels[0].Width ),
            static_cast<uint32_t>( Spec.MipLevels[0].Height ), 1, static_cast<uint32_t>( Spec.MipLevels.size() ), 1,
            VK_SAMPLE_COUNT_VALUE( aBufferDescription.SampleCount ), aBufferDescription.IsCudaVisible, false,
            ToVkFormat( Spec.Format ), ToVkMemoryFlag( Spec ), (VkImageUsageFlags)Spec.Usage );

        TransitionImageLayout( VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL );
        CopyBufferToImage( l_StagingBuffer );
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
    Texture2D::Texture2D( GraphicContext &aGraphicContext, TextureData2D &aCubeMapData, TextureSampler2D &aSamplingInfo,
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

        sImageData &a_ImageData = aCubeMapData.GetImageData();
        Buffer l_StagingBuffer( mGraphicContext, a_ImageData.mPixelData, a_ImageData.mByteSize, eBufferBindType::UNKNOWN, true, false,
                                true, false );

        Spec.MipLevels = { { static_cast<uint32_t>( a_ImageData.mWidth ), static_cast<uint32_t>( a_ImageData.mHeight ), 0, 0 } };
        Spec.Format    = a_ImageData.mFormat;

        mTextureImageObject = New<Internal::sVkImageObject>(
            mGraphicContext.mContext, static_cast<uint32_t>( Spec.MipLevels[0].Width ),
            static_cast<uint32_t>( Spec.MipLevels[0].Height ), 1, static_cast<uint32_t>( Spec.MipLevels.size() ), 1,
            VK_SAMPLE_COUNT_VALUE( Spec.SampleCount ), Spec.IsCudaVisible, false, ToVkFormat( Spec.Format ), ToVkMemoryFlag( Spec ),
            (VkImageUsageFlags)Spec.Usage );

        TransitionImageLayout( VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL );
        CopyBufferToImage( l_StagingBuffer );
        TransitionImageLayout( VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL );

        CreateImageView();
        CreateImageSampler();
    }

    Texture2D::Texture2D( GraphicContext &aGraphicContext, TextureDescription &aBufferDescription )
        : mGraphicContext( aGraphicContext )
        , Spec( aBufferDescription )
    {
        mTextureImageObject = New<Internal::sVkImageObject>(
            mGraphicContext.mContext, static_cast<uint32_t>( Spec.MipLevels[0].Width ),
            static_cast<uint32_t>( Spec.MipLevels[0].Height ), 1, static_cast<uint32_t>( Spec.MipLevels.size() ), 1,
            VK_SAMPLE_COUNT_VALUE( aBufferDescription.SampleCount ), aBufferDescription.IsCudaVisible, false,
            ToVkFormat( Spec.Format ), ToVkMemoryFlag( Spec ), (VkImageUsageFlags)Spec.Usage );

        CreateImageView();
        CreateImageSampler();
    }

    Texture2D::Texture2D( GraphicContext &aGraphicContext, TextureDescription &aBufferDescription, VkImage aImage )
        : Spec( aBufferDescription )
        , mGraphicContext( aGraphicContext )
    {
        mTextureImageObject = New<Internal::sVkImageObject>( aGraphicContext.mContext, aImage );
        CreateImageView();
        CreateImageSampler();
    }

    Texture2D::Texture2D( GraphicContext &aGraphicContext, TextureDescription &aBufferDescription,
                          Ref<Internal::sVkFramebufferImage> aFramebufferImage )
        : Spec( aBufferDescription )
        , mGraphicContext( aGraphicContext )
    {
        mTextureImageObject   = aFramebufferImage->mImage;
        mTextureView          = aFramebufferImage->mImageView;
        mTextureSamplerObject = aFramebufferImage->mImageSampler;
    }

    void Texture2D::CopyBufferToImage( Buffer &aBuffer )
    {
        Ref<Internal::sVkCommandBufferObject> l_CommandBufferObject = mGraphicContext.BeginSingleTimeCommands();

        std::vector<Internal::sImageRegion> l_BufferCopyRegions;
        uint32_t                            offset = 0;

        for( uint32_t i = 0; i < Spec.MipLevels.size(); i++ )
        {
            Internal::sImageRegion bufferCopyRegion{};
            bufferCopyRegion.mBaseLayer     = 0;
            bufferCopyRegion.mLayerCount    = 1;
            bufferCopyRegion.mBaseMipLevel  = i;
            bufferCopyRegion.mMipLevelCount = 1;
            bufferCopyRegion.mWidth         = Spec.MipLevels[i].Width;
            bufferCopyRegion.mHeight        = Spec.MipLevels[i].Height;
            bufferCopyRegion.mDepth         = 1;
            bufferCopyRegion.mOffset        = offset;

            l_BufferCopyRegions.push_back( bufferCopyRegion );
            offset += static_cast<uint32_t>( Spec.MipLevels[i].Size );
        }

        Internal::sImageRegion imageCopyRegion{};
        imageCopyRegion.mBaseMipLevel  = 0;
        imageCopyRegion.mMipLevelCount = Spec.MipLevels.size();
        imageCopyRegion.mLayerCount    = 1;

        l_CommandBufferObject->CopyBuffer( aBuffer.mVkObject, mTextureImageObject, l_BufferCopyRegions, imageCopyRegion );

        mGraphicContext.EndSingleTimeCommands( l_CommandBufferObject );
    }

    void Texture2D::TransitionImageLayout( VkImageLayout aOldLayout, VkImageLayout aNewLayout )
    {
        Ref<Internal::sVkCommandBufferObject> l_CommandBufferObject = mGraphicContext.BeginSingleTimeCommands();

        l_CommandBufferObject->ImageMemoryBarrier( mTextureImageObject, aOldLayout, aNewLayout, Spec.MipLevels.size(), 1 );

        mGraphicContext.EndSingleTimeCommands( l_CommandBufferObject );
    }

    void Texture2D::CreateImageView()
    {
        mTextureView =
            New<Internal::sVkImageViewObject>( mGraphicContext.mContext, mTextureImageObject, 1, VK_IMAGE_VIEW_TYPE_2D,
                                               ToVkFormat( Spec.Format ), (VkImageAspectFlags)Spec.AspectMask,
                                               VkComponentMapping{ VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                                                                   VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY } );
    }

    void Texture2D::CreateImageSampler()
    {
        if( !Spec.Sampled ) return;
        mTextureSamplerObject = New<Internal::sVkImageSamplerObject>( mGraphicContext.mContext, Spec.MinificationFilter,
                                                                      Spec.MagnificationFilter, Spec.WrappingMode, Spec.MipmapMode );
    }

    void Texture2D::GetTextureData( TextureData2D &aTextureData )
    {
        uint32_t lByteSize = Spec.MipLevels[0].Width * Spec.MipLevels[0].Height * sizeof( uint32_t );
        Buffer   lStagingBuffer( mGraphicContext, eBufferBindType::UNKNOWN, true, false, false, true, lByteSize );

        std::vector<Internal::sImageRegion> l_BufferCopyRegions;
        uint32_t                            lBufferByteOffset = 0;
        for( uint32_t i = 0; i < Spec.MipLevels.size(); i++ )
        {
            Internal::sImageRegion bufferCopyRegion{};
            bufferCopyRegion.mBaseLayer     = 0;
            bufferCopyRegion.mLayerCount    = 1;
            bufferCopyRegion.mBaseMipLevel  = i;
            bufferCopyRegion.mMipLevelCount = 1;
            bufferCopyRegion.mWidth         = Spec.MipLevels[i].Width;
            bufferCopyRegion.mHeight        = Spec.MipLevels[i].Height;
            bufferCopyRegion.mDepth         = 1;
            bufferCopyRegion.mOffset        = lBufferByteOffset;

            l_BufferCopyRegions.push_back( bufferCopyRegion );
            lBufferByteOffset += static_cast<uint32_t>( Spec.MipLevels[i].Size );
        }

        TransitionImageLayout( VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL );
        Ref<Internal::sVkCommandBufferObject> l_CommandBufferObject = mGraphicContext.BeginSingleTimeCommands();
        l_CommandBufferObject->CopyImage( mTextureImageObject, lStagingBuffer.mVkObject, l_BufferCopyRegions, 0 );
        mGraphicContext.EndSingleTimeCommands( l_CommandBufferObject );
        TransitionImageLayout( VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL );

        uint8_t *lPixelData = lStagingBuffer.Map<uint8_t>( lByteSize, 0 );

        sImageData lImageDataStruct{};
        lImageDataStruct.mFormat    = Spec.Format;
        lImageDataStruct.mWidth     = Spec.MipLevels[0].Width;
        lImageDataStruct.mHeight    = Spec.MipLevels[0].Height;
        lImageDataStruct.mByteSize  = lByteSize;
        lImageDataStruct.mPixelData = lPixelData;

        Core::TextureData::sCreateInfo lTextureCreateInfo{};
        lTextureCreateInfo.mMipLevels = 1;
        aTextureData                  = TextureData2D( lTextureCreateInfo, lImageDataStruct );
    }

    sTextureSamplingInfo Texture2D::GetTextureSampling()
    {
        return mSamplingSpec;
    }
} // namespace SE::Graphics
