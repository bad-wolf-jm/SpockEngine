#include "VkTexture2D.h"
#include "Buffer.h"
#include "Core/Core.h"
#include "Core/Memory.h"
#include "Core/Vulkan/VkCoreMacros.h"

#include "Core/Logging.h"

namespace SE::Graphics
{
    using namespace Internal;

    static

        static VkMemoryPropertyFlags
        ToVkMemoryFlag( TextureDescription const &aBufferDescription )
    {
        VkMemoryPropertyFlags lFlags = 0;
        if( aBufferDescription.IsHostVisible )
            lFlags |= ( VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT );
        else
            lFlags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

        return lFlags;
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
    VkTexture2D::VkTexture2D( GraphicContext &aGraphicContext, TextureData2D &aTextureData, TextureSampler2D &aSamplingInfo )
        : mGraphicContext( aGraphicContext )
        , mSpec{ aTextureData.mSpec }
    {

        mVkImage = mGraphicContext.mContext->CreateImage( aWidth, aHeight, aDepth, aMipLevels, aLayers, aSampleCount, true,
                                                          false, aFormat, aProperties, aUsage );

        mVkMemory = mGraphicContext.mContext->AllocateMemory( mVkObject, 0, false, true, &mMemorySize );

        mGraphicContext.mContext->BindMemory( mVkObject, mVkMemory );

        // sImageData &lImageData = aTextureData.GetImageData();
        // Buffer lStagingBuffer( mGraphicContext, lImageData.mPixelData, lImageData.mByteSize, eBufferBindType::UNKNOWN, true, false,
        //                        true, false );

        // Spec.MipLevels = { { static_cast<uint32_t>( lImageData.mWidth ), static_cast<uint32_t>( lImageData.mHeight ), 0, 0 } };
        // Spec.Format    = lImageData.mFormat;

        // mTextureImageObject = New<sVkImageObject>(
        //     mGraphicContext.mContext, static_cast<uint32_t>( mSpec.mWidth ), static_cast<uint32_t>( mSpec.mHeight ), 1,
        //     static_cast<uint32_t>( mSpec.mMipLevels ), 1, VK_SAMPLE_COUNT_VALUE( Spec.SampleCount ), Spec.IsCudaVisible, false,
        //     ToVkFormat( Spec.Format ), ToVkMemoryFlag( Spec ), (VkImageUsageFlags)Spec.Usage );

        // TransitionImageLayout( VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL );
        // CopyBufferToImage( lStagingBuffer );
        // TransitionImageLayout( VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL );

    }

    VkTexture2D::VkTexture2D( GraphicContext &aGraphicContext, Core::TextureData::sCreateInfo &aTextureImageDescription )
        : mGraphicContext( aGraphicContext )
        , mSpec( aTextureImageDescription )
    {
        mTextureImageObject = New<sVkImageObject>(
            mGraphicContext.mContext, static_cast<uint32_t>( mSpec.mWidth ), static_cast<uint32_t>( mSpec.mHeight ), 1,
            static_cast<uint32_t>( mSpec.mMipLevels ), 1, VK_SAMPLE_COUNT_VALUE( aBufferDescription.SampleCount ), true, false,
            ToVkFormat( Spec.Format ), ToVkMemoryFlag( mSpec ), (VkImageUsageFlags)Spec.Usage );

    }

    void VkTexture2D::CopyBufferToImage( Buffer &aBuffer )
    {
        Ref<sVkCommandBufferObject> lCommandBufferObject = mGraphicContext.BeginSingleTimeCommands();

        std::vector<sImageRegion> lBufferCopyRegions;
        uint32_t                  lOffset = 0;

        for( uint32_t i = 0; i < mSpec.mMipLevels; i++ )
        {
            sImageRegion lBufferCopyRegion{};
            lBufferCopyRegion.mBaseLayer     = 0;
            lBufferCopyRegion.mLayerCount    = 1;
            lBufferCopyRegion.mBaseMipLevel  = i;
            lBufferCopyRegion.mMipLevelCount = 1;
            lBufferCopyRegion.mWidth         = mSpec.mWidth >> i;
            lBufferCopyRegion.mHeight        = mSpec.mHeight >> i;
            lBufferCopyRegion.mDepth         = 1;
            lBufferCopyRegion.mOffset        = lOffset;

            lBufferCopyRegions.push_back( lBufferCopyRegion );
            lOffset += static_cast<uint32_t>( ( mSpec.mWidth >> i ) * ( mSpec.mHeight >> i ) * sizeof( uint32_t ) );
        }

        sImageRegion imageCopyRegion{};
        imageCopyRegion.mBaseMipLevel  = 0;
        imageCopyRegion.mMipLevelCount = mSpec.mMipLevels;
        imageCopyRegion.mLayerCount    = 1;

        lCommandBufferObject->CopyBuffer( aBuffer.mVkObject, mTextureImageObject, lBufferCopyRegions, imageCopyRegion );

        mGraphicContext.EndSingleTimeCommands( lCommandBufferObject );
    }

    void VkTexture2D::TransitionImageLayout( VkImageLayout aOldLayout, VkImageLayout aNewLayout )
    {
        Ref<sVkCommandBufferObject> lCommandBufferObject = mGraphicContext.BeginSingleTimeCommands();

        lCommandBufferObject->ImageMemoryBarrier( mTextureImageObject, aOldLayout, aNewLayout, mSpec.mMipLevels, 1 );

        mGraphicContext.EndSingleTimeCommands( lCommandBufferObject );
    }

    // void VkTexture2D::CreateImageView()
    // {
    //     constexpr VkComponentMapping IDENTITY_SWIZZLE{ VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
    //                                                    VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY };

    //     mTextureView = New<sVkImageViewObject>( mGraphicContext.mContext, mTextureImageObject, 1, VK_IMAGE_VIEW_TYPE_2D,
    //                                             ToVkFormat( mSpec.mFormat ), VK_IMAGE_ASPECT_COLOR_BIT, IDENTITY_SWIZZLE );
    // }

    void VkTexture2D::GetTextureData( TextureData2D &aTextureData )
    {
        uint32_t lByteSize = mSpec.mWidth * mSpec.mHeight * sizeof( uint32_t );
        Buffer   lStagingBuffer( mGraphicContext, eBufferBindType::UNKNOWN, true, false, false, true, lByteSize );

        std::vector<sImageRegion> lBufferCopyRegions;
        uint32_t                  lBufferByteOffset = 0;
        for( uint32_t i = 0; i < mSpec.mMipLevels; i++ )
        {
            sImageRegion lBufferCopyRegion{};
            lBufferCopyRegion.mBaseLayer     = 0;
            lBufferCopyRegion.mLayerCount    = 1;
            lBufferCopyRegion.mBaseMipLevel  = i;
            lBufferCopyRegion.mMipLevelCount = 1;
            lBufferCopyRegion.mWidth         = Spec.MipLevels[i].Width;
            lBufferCopyRegion.mHeight        = Spec.MipLevels[i].Height;
            lBufferCopyRegion.mDepth         = 1;
            lBufferCopyRegion.mOffset        = lBufferByteOffset;

            lBufferCopyRegions.push_back( lBufferCopyRegion );
            lBufferByteOffset += static_cast<uint32_t>( Spec.MipLevels[i].Size );
        }

        TransitionImageLayout( VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL );
        Ref<sVkCommandBufferObject> lCommandBufferObject = mGraphicContext.BeginSingleTimeCommands();
        lCommandBufferObject->CopyImage( mTextureImageObject, lStagingBuffer.mVkObject, lBufferCopyRegions, 0 );
        mGraphicContext.EndSingleTimeCommands( lCommandBufferObject );
        TransitionImageLayout( VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL );

        uint8_t *lPixelData = lStagingBuffer.Map<uint8_t>( lByteSize, 0 );

        sImageData lImageDataStruct{};
        lImageDataStruct.mFormat    = Spec.Format;
        lImageDataStruct.mWidth     = mSpec.mWidth;
        lImageDataStruct.mHeight    = mSpec.mHeight;
        lImageDataStruct.mByteSize  = lByteSize;
        lImageDataStruct.mPixelData = lPixelData;

        Core::TextureData::sCreateInfo lTextureCreateInfo{};
        lTextureCreateInfo.mMipLevels = 1;
        aTextureData                  = TextureData2D( lTextureCreateInfo, lImageDataStruct );
    }
} // namespace SE::Graphics
