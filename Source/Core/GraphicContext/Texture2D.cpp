#include "Texture2D.h"
#include "Buffer.h"
#include "Core/Core.h"
#include "Core/Memory.h"
#include "Core/Vulkan/VkCoreMacros.h"

#include "Core/Logging.h"

namespace LTSE::Graphics
{

    static VkMemoryPropertyFlags ToVkMemoryFlag( TextureDescription const &a_BufferDescription )
    {
        VkMemoryPropertyFlags l_Flags = 0;
        if( a_BufferDescription.IsHostVisible )
            l_Flags |= ( VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT );
        else
            l_Flags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

        return l_Flags;
    }

    Texture2D::Texture2D( GraphicContext &a_GraphicContext, TextureDescription &a_BufferDescription, TextureData &a_BufferData )
        : mGraphicContext( a_GraphicContext )
        , Spec( a_BufferDescription )
    {
        Buffer l_StagingBuffer( mGraphicContext, reinterpret_cast<uint8_t *>( a_BufferData.Data ), a_BufferData.ByteSize,
            eBufferBindType::UNKNOWN, true, false, true, false );

        m_TextureImageObject = New<Internal::sVkImageObject>( mGraphicContext.mContext,
            static_cast<uint32_t>( Spec.MipLevels[0].Width ), static_cast<uint32_t>( Spec.MipLevels[0].Height ), 1,
            static_cast<uint32_t>( Spec.MipLevels.size() ), 1, VK_SAMPLE_COUNT_VALUE( a_BufferDescription.SampleCount ), false,
            ToVkFormat( Spec.Format ), ToVkMemoryFlag( Spec ), (VkImageUsageFlags)Spec.Usage );

        TransitionImageLayout( VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL );
        CopyBufferToImage( l_StagingBuffer );
        TransitionImageLayout( VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL );

        CreateImageView();
        CreateImageSampler();
    }

    Texture2D::Texture2D( GraphicContext &a_GraphicContext, TextureDescription &a_BufferDescription, sImageData &a_ImageData )
        : mGraphicContext( a_GraphicContext )
        , Spec( a_BufferDescription )
    {
        Buffer l_StagingBuffer(
            mGraphicContext, a_ImageData.mPixelData, a_ImageData.mByteSize, eBufferBindType::UNKNOWN, true, false, true, false );

        Spec.MipLevels = { { static_cast<uint32_t>( a_ImageData.mWidth ), static_cast<uint32_t>( a_ImageData.mHeight ), 0, 0 } };
        Spec.Format    = a_ImageData.mFormat;

        m_TextureImageObject = New<Internal::sVkImageObject>( mGraphicContext.mContext,
            static_cast<uint32_t>( Spec.MipLevels[0].Width ), static_cast<uint32_t>( Spec.MipLevels[0].Height ), 1,
            static_cast<uint32_t>( Spec.MipLevels.size() ), 1, VK_SAMPLE_COUNT_VALUE( a_BufferDescription.SampleCount ), false,
            ToVkFormat( Spec.Format ), ToVkMemoryFlag( Spec ), (VkImageUsageFlags)Spec.Usage );

        TransitionImageLayout( VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL );
        CopyBufferToImage( l_StagingBuffer );
        TransitionImageLayout( VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL );

        CreateImageView();
        CreateImageSampler();
    }

    Texture2D::Texture2D( GraphicContext &a_GraphicContext, TextureDescription &a_BufferDescription, gli::texture2d &a_ImageData )
        : mGraphicContext( a_GraphicContext )
        , Spec( a_BufferDescription )
    {
        Buffer l_StagingBuffer( mGraphicContext, reinterpret_cast<uint8_t *>( a_ImageData.data() ), a_ImageData.size(),
            eBufferBindType::UNKNOWN, true, false, true, false );

        for( uint32_t l_MipLevel = 0; l_MipLevel < a_ImageData.levels(); l_MipLevel++ )
        {
            Spec.MipLevels.push_back( { static_cast<uint32_t>( a_ImageData[l_MipLevel].extent().x ),
                static_cast<uint32_t>( a_ImageData[l_MipLevel].extent().x ), l_MipLevel, a_ImageData[l_MipLevel].size() } );
        }

        Spec.Format = a_BufferDescription.Format;

        m_TextureImageObject = New<Internal::sVkImageObject>( mGraphicContext.mContext,
            static_cast<uint32_t>( Spec.MipLevels[0].Width ), static_cast<uint32_t>( Spec.MipLevels[0].Height ), 1,
            static_cast<uint32_t>( Spec.MipLevels.size() ), 1, VK_SAMPLE_COUNT_VALUE( a_BufferDescription.SampleCount ), false,
            ToVkFormat( Spec.Format ), ToVkMemoryFlag( Spec ), (VkImageUsageFlags)Spec.Usage );

        TransitionImageLayout( VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL );
        CopyBufferToImage( l_StagingBuffer );
        TransitionImageLayout( VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL );

        CreateImageView();
        CreateImageSampler();
    }

    static SamplerFilter Convert( eSamplerFilter aValue )
    {
        switch( aValue )
        {
        case eSamplerFilter::NEAREST:
            return SamplerFilter::NEAREST;
        case eSamplerFilter::LINEAR:
        default:
            return SamplerFilter::LINEAR;
        }
    }

    static SamplerMipmap Convert( eSamplerMipmap aValue )
    {
        switch( aValue )
        {
        case eSamplerMipmap::NEAREST:
            return SamplerMipmap::NEAREST;
        case eSamplerMipmap::LINEAR:
        default:
            return SamplerMipmap::LINEAR;
        }
    }

    static SamplerWrapping Convert( eSamplerWrapping aValue )
    {
        switch( aValue )
        {
        case eSamplerWrapping::REPEAT:
            return SamplerWrapping::REPEAT;
        case eSamplerWrapping::MIRRORED_REPEAT:
            return SamplerWrapping::MIRRORED_REPEAT;
        case eSamplerWrapping::CLAMP_TO_EDGE:
            return SamplerWrapping::CLAMP_TO_EDGE;
        case eSamplerWrapping::CLAMP_TO_BORDER:
            return SamplerWrapping::CLAMP_TO_BORDER;
        case eSamplerWrapping::MIRROR_CLAMP_TO_BORDER:
            return SamplerWrapping::MIRROR_CLAMP_TO_BORDER;
        default:
            return SamplerWrapping::REPEAT;
        }
    }

    /** @brief */
    Texture2D::Texture2D( GraphicContext &a_GraphicContext, TextureData2D &a_CubeMapData, TextureSampler2D &aSamplingInfo )
        : mGraphicContext( a_GraphicContext )
    {
        Spec.MinificationFilter  = Convert( aSamplingInfo.mSamplingSpec.mMinification );
        Spec.MagnificationFilter = Convert( aSamplingInfo.mSamplingSpec.mMagnification );
        Spec.MipmapMode          = Convert( aSamplingInfo.mSamplingSpec.mMip );
        Spec.WrappingMode        = Convert( aSamplingInfo.mSamplingSpec.mWrapping );
        Spec.Format              = a_CubeMapData.mSpec.mFormat;
        Spec.Sampled             = true;
        Spec.SampleCount         = 1;
        Spec.Usage = { TextureUsageFlags::SAMPLED, TextureUsageFlags::TRANSFER_SOURCE, TextureUsageFlags::TRANSFER_DESTINATION };

        sImageData &a_ImageData = a_CubeMapData.GetImageData();
        Buffer      l_StagingBuffer(
                 mGraphicContext, a_ImageData.mPixelData, a_ImageData.mByteSize, eBufferBindType::UNKNOWN, true, false, true, false );

        Spec.MipLevels = { { static_cast<uint32_t>( a_ImageData.mWidth ), static_cast<uint32_t>( a_ImageData.mHeight ), 0, 0 } };
        Spec.Format    = a_ImageData.mFormat;

        m_TextureImageObject = New<Internal::sVkImageObject>( mGraphicContext.mContext,
            static_cast<uint32_t>( Spec.MipLevels[0].Width ), static_cast<uint32_t>( Spec.MipLevels[0].Height ), 1,
            static_cast<uint32_t>( Spec.MipLevels.size() ), 1, VK_SAMPLE_COUNT_VALUE( Spec.SampleCount ), false,
            ToVkFormat( Spec.Format ), ToVkMemoryFlag( Spec ), (VkImageUsageFlags)Spec.Usage );

        TransitionImageLayout( VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL );
        CopyBufferToImage( l_StagingBuffer );
        TransitionImageLayout( VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL );

        CreateImageView();
        CreateImageSampler();
    }

    Texture2D::Texture2D( GraphicContext &a_GraphicContext, TextureDescription &a_BufferDescription )
        : mGraphicContext( a_GraphicContext )
        , Spec( a_BufferDescription )
    {
        m_TextureImageObject = New<Internal::sVkImageObject>( mGraphicContext.mContext,
            static_cast<uint32_t>( Spec.MipLevels[0].Width ), static_cast<uint32_t>( Spec.MipLevels[0].Height ), 1,
            static_cast<uint32_t>( Spec.MipLevels.size() ), 1, VK_SAMPLE_COUNT_VALUE( a_BufferDescription.SampleCount ), false,
            ToVkFormat( Spec.Format ), ToVkMemoryFlag( Spec ), (VkImageUsageFlags)Spec.Usage );

        CreateImageView();
        CreateImageSampler();
    }

    Texture2D::Texture2D( GraphicContext &a_GraphicContext, TextureDescription &a_BufferDescription, VkImage a_Image )
        : Spec( a_BufferDescription )
        , mGraphicContext( a_GraphicContext )
    {
        m_TextureImageObject = New<Internal::sVkImageObject>( a_GraphicContext.mContext, a_Image );
        CreateImageView();
        CreateImageSampler();
    }

    Texture2D::Texture2D( GraphicContext &a_GraphicContext, TextureDescription &a_BufferDescription,
        Ref<Internal::sVkFramebufferImage> a_FramebufferImage )
        : Spec( a_BufferDescription )
        , mGraphicContext( a_GraphicContext )
    {
        m_TextureImageObject   = a_FramebufferImage->mImage;
        m_TextureView          = a_FramebufferImage->mImageView;
        m_TextureSamplerObject = a_FramebufferImage->mImageSampler;
    }

    void Texture2D::CopyBufferToImage( Buffer &a_Buffer )
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

        l_CommandBufferObject->CopyBuffer( a_Buffer.mVkObject, m_TextureImageObject, l_BufferCopyRegions, imageCopyRegion );

        mGraphicContext.EndSingleTimeCommands( l_CommandBufferObject );
    }

    void Texture2D::TransitionImageLayout( VkImageLayout oldLayout, VkImageLayout newLayout )
    {
        Ref<Internal::sVkCommandBufferObject> l_CommandBufferObject = mGraphicContext.BeginSingleTimeCommands();

        l_CommandBufferObject->ImageMemoryBarrier( m_TextureImageObject, oldLayout, newLayout, Spec.MipLevels.size(), 1 );

        mGraphicContext.EndSingleTimeCommands( l_CommandBufferObject );
    }

    void Texture2D::CreateImageView()
    {
        m_TextureView = New<Internal::sVkImageViewObject>( mGraphicContext.mContext, m_TextureImageObject, 1, VK_IMAGE_VIEW_TYPE_2D,
            ToVkFormat( Spec.Format ), (VkImageAspectFlags)Spec.AspectMask,
            VkComponentMapping{ (VkComponentSwizzle)Spec.ComponentSwizzle[0], (VkComponentSwizzle)Spec.ComponentSwizzle[1],
                (VkComponentSwizzle)Spec.ComponentSwizzle[2], (VkComponentSwizzle)Spec.ComponentSwizzle[3] } );
    }

    void Texture2D::CreateImageSampler()
    {
        if( !Spec.Sampled ) return;
        m_TextureSamplerObject = New<Internal::sVkImageSamplerObject>( mGraphicContext.mContext, (VkFilter)Spec.MinificationFilter,
            (VkFilter)Spec.MagnificationFilter, (VkSamplerAddressMode)Spec.WrappingMode, (VkSamplerMipmapMode)Spec.MipmapMode );
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
        l_CommandBufferObject->CopyImage( m_TextureImageObject, lStagingBuffer.mVkObject, l_BufferCopyRegions, 0 );
        mGraphicContext.EndSingleTimeCommands( l_CommandBufferObject );
        TransitionImageLayout( VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL );

        uint8_t *lPixelData = lStagingBuffer.Map<uint8_t>( lByteSize, 0 );

        sImageData lImageDataStruct{};
        lImageDataStruct.mFormat    = eColorFormat::RGBA8_UNORM;
        lImageDataStruct.mWidth     = Spec.MipLevels[0].Width;
        lImageDataStruct.mHeight    = Spec.MipLevels[0].Height;
        lImageDataStruct.mByteSize  = lByteSize;
        lImageDataStruct.mPixelData = lPixelData;

        Core::TextureData::sCreateInfo lTextureCreateInfo{};
        lTextureCreateInfo.mMipLevels = 1;
        aTextureData                  = TextureData2D( lTextureCreateInfo, lImageDataStruct );
    }

} // namespace LTSE::Graphics
