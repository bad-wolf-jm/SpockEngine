#include "VkTexture2D.h"

#include "Core/Core.h"
#include "Core/Logging.h"
#include "Core/Memory.h"

#include "Graphics/Vulkan/VkCoreMacros.h"

#ifdef CUDA_INTEROP
#    include "Core/CUDA/Array/CudaBuffer.h"
#    include "Core/CUDA/CudaAssert.h"
#    include "Core/CUDA/Texture/Conversion.h"
#endif
#include "VkCommand.h"
#include "VkGpuBuffer.h"

namespace SE::Graphics
{
    /** @brief */
    VkTexture2D::VkTexture2D( Ref<IGraphicContext> aGraphicContext, TextureData2D &aTextureData, uint8_t aSampleCount,
                              bool aIsHostVisible, bool aIsGraphicsOnly, bool aIsTransferSource )
        : ITexture2D( aGraphicContext, aTextureData.mSpec, aSampleCount, aIsHostVisible, aIsGraphicsOnly, aIsTransferSource, false )
    {
        if( mSpec.mIsDepthTexture ) mSpec.mFormat = GraphicContext<VkGraphicContext>()->GetDepthFormat();

        CreateImage();
        AllocateMemory();
        BindMemory();
        ConfigureExternalMemoryHandle();

        sImageData &lImageData = aTextureData.GetImageData();
        auto lStagingBuffer = New<VkGpuBuffer>( GraphicContext<VkGraphicContext>(), lImageData.mPixelData.data(), lImageData.mByteSize,
                                                eBufferType::UNKNOWN, true, false, true, false );
        TransitionImageLayout( VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL );
        SetPixelData( lStagingBuffer );
        TransitionImageLayout( VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL );
    }

    VkTexture2D::VkTexture2D( Ref<IGraphicContext> aGraphicContext, Core::sTextureCreateInfo &aTextureImageDescription,
                              uint8_t aSampleCount, bool aIsHostVisible, bool aIsGraphicsOnly, bool aIsTransferSource,
                              bool aIsTransferDestination )
        : ITexture2D( aGraphicContext, aTextureImageDescription, aSampleCount, aIsHostVisible, aIsGraphicsOnly, aIsTransferSource,
                      aIsTransferDestination )
    {
        if( mSpec.mIsDepthTexture ) mSpec.mFormat = GraphicContext<VkGraphicContext>()->GetDepthFormat();

        CreateImage();
        AllocateMemory();
        BindMemory();
        ConfigureExternalMemoryHandle();
    }

    VkTexture2D::VkTexture2D( Ref<IGraphicContext> aGraphicContext, Core::sTextureCreateInfo &aTextureImageDescription,
                              VkImage aExternalImage )
        : ITexture2D( aGraphicContext, aTextureImageDescription, 1, false, true, false, false )
        , mVkImage{ aExternalImage }
    {
        if( mSpec.mIsDepthTexture ) mSpec.mFormat = GraphicContext<VkGraphicContext>()->GetDepthFormat();
    }

    VkTexture2D::~VkTexture2D()
    {
        GraphicContext<VkGraphicContext>()->DestroyImage( mVkImage );
        GraphicContext<VkGraphicContext>()->FreeMemory( mVkMemory );
    }

    void VkTexture2D::CreateImage()
    {
        mVkImage = GraphicContext<VkGraphicContext>()->CreateImage(
            mSpec.mWidth, mSpec.mHeight, mSpec.mDepth, mSpec.mMipLevels, mSpec.mLayers, VK_SAMPLE_COUNT_VALUE( mSampleCount ),
            !mIsGraphicsOnly, false,
            mSpec.mIsDepthTexture ? ToVkFormat( mGraphicContext->GetDepthFormat() ) : ToVkFormat( mSpec.mFormat ), MemoryProperties(),
            ImageUsage() );
    }

    void VkTexture2D::AllocateMemory()
    {
        mVkMemory = GraphicContext<VkGraphicContext>()->AllocateMemory( mVkImage, 0, mIsHostVisible, !mIsGraphicsOnly, &mMemorySize );
    }

    void VkTexture2D::BindMemory() { GraphicContext<VkGraphicContext>()->BindMemory( mVkImage, mVkMemory ); }

    VkMemoryPropertyFlags VkTexture2D::MemoryProperties()
    {
        VkMemoryPropertyFlags lProperties = 0;
        if( mIsHostVisible )
            lProperties |= ( VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT );
        else
            lProperties |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

        return lProperties;
    }

    VkImageUsageFlags VkTexture2D::ImageUsage()
    {
        VkImageUsageFlags lUsage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        if( mIsTransferSource ) lUsage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        if( mIsTransferDestination ) lUsage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;

        if( mSpec.mIsDepthTexture )
            lUsage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        else
            lUsage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

        return lUsage;
    }

    void VkTexture2D::ConfigureExternalMemoryHandle()
    {
#ifdef CUDA_INTEROP
        if( mIsGraphicsOnly ) return;

        cudaExternalMemoryHandleDesc lCudaExternalMemoryHandleDesc{};
        lCudaExternalMemoryHandleDesc.type  = cudaExternalMemoryHandleTypeOpaqueWin32;
        lCudaExternalMemoryHandleDesc.size  = mMemorySize;
        lCudaExternalMemoryHandleDesc.flags = 0;
        lCudaExternalMemoryHandleDesc.handle.win32.handle =
            (HANDLE)GraphicContext<VkGraphicContext>()->GetSharedMemoryHandle( mVkMemory );
        CUDA_ASSERT( cudaImportExternalMemory( &mExternalMemoryHandle, &lCudaExternalMemoryHandleDesc ) );

        cudaExternalMemoryMipmappedArrayDesc lExternalMemoryMipmappedArrayDesc{};
        lExternalMemoryMipmappedArrayDesc.formatDesc = Cuda::ToCudaChannelDesc( mSpec.mFormat );

        lExternalMemoryMipmappedArrayDesc.extent.width  = mSpec.mWidth;
        lExternalMemoryMipmappedArrayDesc.extent.height = mSpec.mHeight;
        lExternalMemoryMipmappedArrayDesc.extent.depth  = 0;
        lExternalMemoryMipmappedArrayDesc.numLevels     = mSpec.mMipLevels;
        lExternalMemoryMipmappedArrayDesc.flags         = 0;
        CUDA_ASSERT( cudaExternalMemoryGetMappedMipmappedArray( &mInternalCudaMipmappedArray, mExternalMemoryHandle,
                                                                &lExternalMemoryMipmappedArrayDesc ) );
        CUDA_ASSERT( cudaGetMipmappedArrayLevel( &mInternalCudaArray, mInternalCudaMipmappedArray, 0 ) );
#endif
    }

    void VkTexture2D::SetPixelData( Ref<IGraphicBuffer> aBuffer )
    {
        Ref<sVkCommandBufferObject> lCommandBufferObject = SE::Core::New<sVkCommandBufferObject>( GraphicContext<VkGraphicContext>() );
        lCommandBufferObject->Begin( VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT );

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

        sImageRegion lImageCopyRegion{};
        lImageCopyRegion.mBaseMipLevel  = 0;
        lImageCopyRegion.mMipLevelCount = mSpec.mMipLevels;
        lImageCopyRegion.mLayerCount    = 1;

        lCommandBufferObject->CopyBuffer( std::reinterpret_pointer_cast<VkGpuBuffer>( aBuffer )->mVkBuffer, mVkImage,
                                          lBufferCopyRegions, lImageCopyRegion );

        lCommandBufferObject->End();
        lCommandBufferObject->SubmitTo( GraphicContext<VkGraphicContext>()->GetTransferQueue() );
        GraphicContext<VkGraphicContext>()->WaitIdle( GraphicContext<VkGraphicContext>()->GetTransferQueue() );
    }

    void VkTexture2D::TransitionImageLayout( VkImageLayout aOldLayout, VkImageLayout aNewLayout )
    {
        Ref<sVkCommandBufferObject> lCommandBufferObject = SE::Core::New<sVkCommandBufferObject>( GraphicContext<VkGraphicContext>() );
        lCommandBufferObject->Begin( VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT );
        lCommandBufferObject->ImageMemoryBarrier( mVkImage, aOldLayout, aNewLayout, mSpec.mMipLevels, 1 );
        lCommandBufferObject->End();
        lCommandBufferObject->SubmitTo( GraphicContext<VkGraphicContext>()->GetTransferQueue() );
        GraphicContext<VkGraphicContext>()->WaitIdle( GraphicContext<VkGraphicContext>()->GetTransferQueue() );
    }

    void VkTexture2D::GetPixelData( TextureData2D &aTextureData )
    {
        uint32_t    lByteSize = mSpec.mWidth * mSpec.mHeight * sizeof( uint32_t );
        VkGpuBuffer lStagingBuffer( GraphicContext<VkGraphicContext>(), eBufferType::UNKNOWN, true, false, false, true, lByteSize );

        std::vector<sImageRegion> lBufferCopyRegions;
        uint32_t                  lBufferByteOffset = 0;
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
            lBufferCopyRegion.mOffset        = lBufferByteOffset;

            lBufferCopyRegions.push_back( lBufferCopyRegion );
            lBufferByteOffset += static_cast<uint32_t>( ( mSpec.mWidth >> i ) * ( mSpec.mHeight >> i ) * sizeof( uint32_t ) );
        }

        TransitionImageLayout( VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL );
        Ref<sVkCommandBufferObject> lCommandBufferObject = SE::Core::New<sVkCommandBufferObject>( GraphicContext<VkGraphicContext>() );
        lCommandBufferObject->Begin( VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT );
        lCommandBufferObject->CopyImage( mVkImage, lStagingBuffer.mVkBuffer, lBufferCopyRegions, 0 );
        lCommandBufferObject->End();
        lCommandBufferObject->SubmitTo( GraphicContext<VkGraphicContext>()->GetTransferQueue() );
        GraphicContext<VkGraphicContext>()->WaitIdle( GraphicContext<VkGraphicContext>()->GetTransferQueue() );
        TransitionImageLayout( VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL );

        uint8_t *lPixelData = lStagingBuffer.Map<uint8_t>( lByteSize, 0 );

        sImageData lImageDataStruct{};
        lImageDataStruct.mFormat    = mSpec.mFormat;
        lImageDataStruct.mWidth     = mSpec.mWidth;
        lImageDataStruct.mHeight    = mSpec.mHeight;
        lImageDataStruct.mByteSize  = lByteSize;
        lImageDataStruct.mPixelData = std::vector<uint8_t>( lPixelData, lPixelData + lByteSize );

        aTextureData = TextureData2D( mSpec, lImageDataStruct );
    }
} // namespace SE::Graphics
