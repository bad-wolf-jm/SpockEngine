#include "TextureCubeMap.h"
#include "Buffer.h"
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

    TextureCubeMap::TextureCubeMap( GraphicContext &aGraphicContext, TextureDescription &aBufferDescription )
        : mGraphicContext( aGraphicContext )
        , Spec( aBufferDescription )
    {
        mTextureImageObject = New<Internal::sVkImageObject>(
            mGraphicContext.mContext, static_cast<uint32_t>( Spec.MipLevels[0].Width ),
            static_cast<uint32_t>( Spec.MipLevels[0].Height ), 1, static_cast<uint32_t>( Spec.MipLevels.size() ), 6,
            VK_SAMPLE_COUNT_VALUE( aBufferDescription.SampleCount ), aBufferDescription.IsCudaVisible, true, ToVkFormat( Spec.Format ),
            ToVkMemoryFlag( Spec ), (VkImageUsageFlags)Spec.Usage );

        CreateImageView();
        CreateImageSampler();
    }

    TextureCubeMap::TextureCubeMap( GraphicContext &aGraphicContext, TextureDescription &aBufferDescription,
                                    gli::texture_cube &aCubeMapData )
        : mGraphicContext( aGraphicContext )
        , Spec( aBufferDescription )
    {

        for( uint32_t l_MipLevel = 0; l_MipLevel < aCubeMapData.levels(); l_MipLevel++ )
        {
            Spec.MipLevels.push_back( { static_cast<uint32_t>( aCubeMapData[0][l_MipLevel].extent().x ),
                                        static_cast<uint32_t>( aCubeMapData[0][l_MipLevel].extent().y ), l_MipLevel,
                                        aCubeMapData[0][l_MipLevel].size() } );
        }

        Buffer l_StagingBuffer( mGraphicContext, reinterpret_cast<uint8_t *>( aCubeMapData.data() ), aCubeMapData.size(),
                                eBufferBindType::UNKNOWN, true, false, true, false );

        mTextureImageObject = New<Internal::sVkImageObject>(
            mGraphicContext.mContext, static_cast<uint32_t>( Spec.MipLevels[0].Width ),
            static_cast<uint32_t>( Spec.MipLevels[0].Height ), 1, static_cast<uint32_t>( Spec.MipLevels.size() ), 6,
            VK_SAMPLE_COUNT_VALUE( aBufferDescription.SampleCount ), aBufferDescription.IsCudaVisible, true, ToVkFormat( Spec.Format ),
            ToVkMemoryFlag( Spec ), (VkImageUsageFlags)Spec.Usage );

        TransitionImageLayout( VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL );
        CopyBufferToImage( l_StagingBuffer, aCubeMapData );
        TransitionImageLayout( VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL );

        CreateImageView();
        CreateImageSampler();
    }

    void TextureCubeMap::CopyBufferToImage( Buffer &aBuffer, gli::texture_cube &aCubeMapData )
    {
        Ref<Internal::sVkCommandBufferObject> l_CommandBufferObject = mGraphicContext.BeginSingleTimeCommands();

        std::vector<Internal::sImageRegion> l_BufferCopyRegions;
        uint32_t                            offset = 0;

        for( uint32_t face = 0; face < 6; face++ )
        {
            for( uint32_t level = 0; level < Spec.MipLevels.size(); level++ )
            {

                Internal::sImageRegion bufferCopyRegion{};
                bufferCopyRegion.mBaseLayer     = 0;
                bufferCopyRegion.mLayerCount    = 1;
                bufferCopyRegion.mBaseMipLevel  = level;
                bufferCopyRegion.mMipLevelCount = 1;
                bufferCopyRegion.mWidth         = static_cast<uint32_t>( aCubeMapData[face][level].extent().x );
                bufferCopyRegion.mHeight        = static_cast<uint32_t>( aCubeMapData[face][level].extent().y );
                bufferCopyRegion.mDepth         = 1;
                bufferCopyRegion.mOffset        = offset;

                l_BufferCopyRegions.push_back( bufferCopyRegion );
                offset += aCubeMapData[face][level].size();
            }
        }

        Internal::sImageRegion imageCopyRegion{};
        imageCopyRegion.mBaseMipLevel  = 0;
        imageCopyRegion.mMipLevelCount = Spec.MipLevels.size();
        imageCopyRegion.mLayerCount    = 6;

        l_CommandBufferObject->CopyBuffer( aBuffer.mVkObject, mTextureImageObject, l_BufferCopyRegions, imageCopyRegion );

        mGraphicContext.EndSingleTimeCommands( l_CommandBufferObject );
    }

    void TextureCubeMap::TransitionImageLayout( VkImageLayout aOldLayout, VkImageLayout aNewLayout )
    {
        Ref<Internal::sVkCommandBufferObject> l_CommandBufferObject = mGraphicContext.BeginSingleTimeCommands();

        l_CommandBufferObject->ImageMemoryBarrier( mTextureImageObject, aOldLayout, aNewLayout, Spec.MipLevels.size(), 6 );

        mGraphicContext.EndSingleTimeCommands( l_CommandBufferObject );
    }

    void TextureCubeMap::CreateImageView()
    {
        mTextureView =
            New<Internal::sVkImageViewObject>( mGraphicContext.mContext, mTextureImageObject, 6, VK_IMAGE_VIEW_TYPE_CUBE,
                                               ToVkFormat( Spec.Format ), (VkImageAspectFlags)Spec.AspectMask,
                                               VkComponentMapping{ VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                                                                   VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY } );
    }

    void TextureCubeMap::CreateImageSampler()
    {
        if( !Spec.Sampled ) return;
        mTextureSamplerObject = New<Internal::sVkImageSamplerObject>(
            mGraphicContext.mContext, (VkFilter)Spec.MinificationFilter, (VkFilter)Spec.MagnificationFilter,
            (VkSamplerAddressMode)Spec.WrappingMode, (VkSamplerMipmapMode)Spec.MipmapMode );
    }
} // namespace SE::Graphics
