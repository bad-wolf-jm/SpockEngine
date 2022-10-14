#include "TextureCubeMap.h"
#include "Buffer.h"
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

    TextureCubeMap::TextureCubeMap( GraphicContext &a_GraphicContext, TextureDescription &a_BufferDescription )
        : mGraphicContext( a_GraphicContext )
        , Spec( a_BufferDescription )
    {
        m_TextureImageObject =
            New<Internal::sVkImageObject>( mGraphicContext.mContext, static_cast<uint32_t>( Spec.MipLevels[0].Width ), static_cast<uint32_t>( Spec.MipLevels[0].Height ), 1,
                                           static_cast<uint32_t>( Spec.MipLevels.size() ), 6, VK_SAMPLE_COUNT_VALUE( a_BufferDescription.SampleCount ), true,
                                           ToVkFormat( Spec.Format ), ToVkMemoryFlag( Spec ), (VkImageUsageFlags)Spec.Usage );

        CreateImageView();
        CreateImageSampler();
    }

    TextureCubeMap::TextureCubeMap( GraphicContext &a_GraphicContext, TextureDescription &a_BufferDescription, gli::texture_cube &a_CubeMapData )
        : mGraphicContext( a_GraphicContext )
        , Spec( a_BufferDescription )
    {

        for( uint32_t l_MipLevel = 0; l_MipLevel < a_CubeMapData.levels(); l_MipLevel++ )
        {
            Spec.MipLevels.push_back( { static_cast<uint32_t>( a_CubeMapData[0][l_MipLevel].extent().x ), static_cast<uint32_t>( a_CubeMapData[0][l_MipLevel].extent().y ),
                                        l_MipLevel, a_CubeMapData[0][l_MipLevel].size() } );
        }

        Buffer l_StagingBuffer( mGraphicContext, reinterpret_cast<uint8_t *>( a_CubeMapData.data() ), a_CubeMapData.size(), eBufferBindType::UNKNOWN, true, false, true, false );

        m_TextureImageObject =
            New<Internal::sVkImageObject>( mGraphicContext.mContext, static_cast<uint32_t>( Spec.MipLevels[0].Width ), static_cast<uint32_t>( Spec.MipLevels[0].Height ), 1,
                                           static_cast<uint32_t>( Spec.MipLevels.size() ), 6, VK_SAMPLE_COUNT_VALUE( a_BufferDescription.SampleCount ), true,
                                           ToVkFormat( Spec.Format ), ToVkMemoryFlag( Spec ), (VkImageUsageFlags)Spec.Usage );

        TransitionImageLayout( VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL );
        CopyBufferToImage( l_StagingBuffer, a_CubeMapData );
        TransitionImageLayout( VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL );

        CreateImageView();
        CreateImageSampler();
    }

    void TextureCubeMap::CopyBufferToImage( Buffer &a_Buffer, gli::texture_cube &a_CubeMapData )
    {
        Ref<Internal::sVkCommandBufferObject> l_CommandBufferObject = mGraphicContext.BeginSingleTimeCommands();

        std::vector<Internal::sImageRegion> l_BufferCopyRegions;
        uint32_t offset = 0;

        for( uint32_t face = 0; face < 6; face++ )
        {
            for( uint32_t level = 0; level < Spec.MipLevels.size(); level++ )
            {

                Internal::sImageRegion bufferCopyRegion{};
                bufferCopyRegion.mBaseLayer     = 0;
                bufferCopyRegion.mLayerCount    = 1;
                bufferCopyRegion.mBaseMipLevel  = level;
                bufferCopyRegion.mMipLevelCount = 1;
                bufferCopyRegion.mWidth         = static_cast<uint32_t>( a_CubeMapData[face][level].extent().x );
                bufferCopyRegion.mHeight        = static_cast<uint32_t>( a_CubeMapData[face][level].extent().y );
                bufferCopyRegion.mDepth         = 1;
                bufferCopyRegion.mOffset        = offset;

                l_BufferCopyRegions.push_back( bufferCopyRegion );
                offset += a_CubeMapData[face][level].size();
            }
        }

        Internal::sImageRegion imageCopyRegion{};
        imageCopyRegion.mBaseMipLevel  = 0;
        imageCopyRegion.mMipLevelCount = Spec.MipLevels.size();
        imageCopyRegion.mLayerCount    = 6;

        l_CommandBufferObject->CopyBuffer( a_Buffer.mVkObject, m_TextureImageObject, l_BufferCopyRegions, imageCopyRegion );

        mGraphicContext.EndSingleTimeCommands( l_CommandBufferObject );
    }

    void TextureCubeMap::TransitionImageLayout( VkImageLayout oldLayout, VkImageLayout newLayout )
    {
        Ref<Internal::sVkCommandBufferObject> l_CommandBufferObject = mGraphicContext.BeginSingleTimeCommands();

        l_CommandBufferObject->ImageMemoryBarrier( m_TextureImageObject, oldLayout, newLayout, Spec.MipLevels.size(), 6 );

        mGraphicContext.EndSingleTimeCommands( l_CommandBufferObject );
    }

    void TextureCubeMap::CreateImageView()
    {
        m_TextureView = New<Internal::sVkImageViewObject>( mGraphicContext.mContext, m_TextureImageObject, 6, VK_IMAGE_VIEW_TYPE_CUBE, ToVkFormat( Spec.Format ),
                                                           (VkImageAspectFlags)Spec.AspectMask,
                                                           VkComponentMapping{ (VkComponentSwizzle)Spec.ComponentSwizzle[0], (VkComponentSwizzle)Spec.ComponentSwizzle[1],
                                                                               (VkComponentSwizzle)Spec.ComponentSwizzle[2], (VkComponentSwizzle)Spec.ComponentSwizzle[3] } );
    }

    void TextureCubeMap::CreateImageSampler()
    {
        if( !Spec.Sampled )
            return;
        m_TextureSamplerObject = New<Internal::sVkImageSamplerObject>( mGraphicContext.mContext, (VkFilter)Spec.MinificationFilter, (VkFilter)Spec.MagnificationFilter,
                                                                       (VkSamplerAddressMode)Spec.WrappingMode, (VkSamplerMipmapMode)Spec.MipmapMode );
    }
} // namespace LTSE::Graphics
