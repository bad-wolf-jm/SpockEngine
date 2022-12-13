#include "VkTexture2D.h"

#include "Core/Core.h"
#include "Core/Logging.h"
#include "Core/Memory.h"

#include "Graphics/Vulkan/VkCoreMacros.h"

#include "Core/CUDA/Array/CudaBuffer.h"
#include "Core/CUDA/CudaAssert.h"

#include "VkGpuBuffer.h"

namespace SE::Graphics
{
    using namespace Internal;

    /** @brief */
    VkTexture2D::VkTexture2D( GraphicContext &aGraphicContext, TextureData2D &mTextureData, uint8_t aSampleCount, bool aIsHostVisible,
                              bool aIsGraphicsOnly, bool aIsTransferSource )
        : mGraphicContext( aGraphicContext )
        , mSpec{ mTextureData.mSpec }
        , mSampleCount{ VK_SAMPLE_COUNT_VALUE( aSampleCount ) }
        , mIsHostVisible{ aIsHostVisible }
        , mIsGraphicsOnly{ aIsGraphicsOnly }
        , mIsTransferSource{ aIsTransferSource }
        , mIsTransferDestination{ false }
    {
        if( mSpec.mIsDepthTexture ) mSpec.mFormat = ToLtseFormat( mGraphicContext.mContext->GetDepthFormat() );

        CreateImage();
        AllocateMemory();
        BindMemory();
        ConfigureExternalMemoryHandle();

        sImageData &lImageData = mTextureData.GetImageData();
        VkGpuBuffer lStagingBuffer( mGraphicContext, lImageData.mPixelData.data(), lImageData.mByteSize, eBufferBindType::UNKNOWN,
                                    true, false, true, false );
        TransitionImageLayout( VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL );
        SetPixelData( lStagingBuffer );
        TransitionImageLayout( VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL );
    }

    VkTexture2D::VkTexture2D( GraphicContext &aGraphicContext, Core::sTextureCreateInfo &aTextureImageDescription,
                              uint8_t aSampleCount, bool aIsHostVisible, bool aIsGraphicsOnly, bool aIsTransferSource,
                              bool aIsTransferDestination )
        : mGraphicContext( aGraphicContext )
        , mSpec( aTextureImageDescription )
        , mSampleCount{ VK_SAMPLE_COUNT_VALUE( aSampleCount ) }
        , mIsHostVisible{ aIsHostVisible }
        , mIsGraphicsOnly{ aIsGraphicsOnly }
        , mIsTransferSource{ aIsTransferSource }
        , mIsTransferDestination{ aIsTransferDestination }
    {
        if( mSpec.mIsDepthTexture ) mSpec.mFormat = ToLtseFormat( mGraphicContext.mContext->GetDepthFormat() );

        CreateImage();
        AllocateMemory();
        BindMemory();
        ConfigureExternalMemoryHandle();
    }

    VkTexture2D::VkTexture2D( GraphicContext &aGraphicContext, Core::sTextureCreateInfo &aTextureImageDescription,
                              VkImage aExternalImage )
        : mGraphicContext( aGraphicContext )
        , mSpec( aTextureImageDescription )
        , mSampleCount{ VK_SAMPLE_COUNT_VALUE( 1 ) }
        , mIsHostVisible{ false }
        , mIsGraphicsOnly{ true }
        , mIsTransferSource{ false }
        , mIsTransferDestination{ false }
        , mVkImage{ aExternalImage }
    {
        if( mSpec.mIsDepthTexture ) mSpec.mFormat = ToLtseFormat( mGraphicContext.mContext->GetDepthFormat() );
    }

    void VkTexture2D::CreateImage()
    {
        mVkImage = mGraphicContext.mContext->CreateImage(
            mSpec.mWidth, mSpec.mHeight, mSpec.mDepth, mSpec.mMipLevels, mSpec.mLayers, mSampleCount, !mIsGraphicsOnly, false,
            mSpec.mIsDepthTexture ? mGraphicContext.mContext->GetDepthFormat() : ToVkFormat( mSpec.mFormat ), MemoryProperties(),
            ImageUsage() );
    }

    void VkTexture2D::AllocateMemory()
    {
        mVkMemory = mGraphicContext.mContext->AllocateMemory( mVkImage, 0, mIsHostVisible, !mIsGraphicsOnly, &mMemorySize );
    }

    void VkTexture2D::BindMemory() { mGraphicContext.mContext->BindMemory( mVkImage, mVkMemory ); }

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
        if( mIsGraphicsOnly ) return;

        cudaExternalMemoryHandleDesc lCudaExternalMemoryHandleDesc{};
        lCudaExternalMemoryHandleDesc.type                = cudaExternalMemoryHandleTypeOpaqueWin32;
        lCudaExternalMemoryHandleDesc.size                = mMemorySize;
        lCudaExternalMemoryHandleDesc.flags               = 0;
        lCudaExternalMemoryHandleDesc.handle.win32.handle = (HANDLE)mGraphicContext.mContext->GetSharedMemoryHandle( mVkMemory );
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
    }

    void VkTexture2D::SetPixelData( VkGpuBuffer &aBuffer )
    {
        // Ref<sVkCommandBufferObject> lCommandBufferObject = mGraphicContext.BeginSingleTimeCommands();
        Ref<Internal::sVkCommandBufferObject> lCommandBufferObject =
            SE::Core::New<Internal::sVkCommandBufferObject>( mGraphicContext.mContext );
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

        lCommandBufferObject->CopyBuffer( aBuffer.mVkBuffer, mVkImage, lBufferCopyRegions, lImageCopyRegion );

        // mGraphicContext.EndSingleTimeCommands( lCommandBufferObject );
        lCommandBufferObject->End();
        lCommandBufferObject->SubmitTo( mGraphicContext.mContext->GetGraphicsQueue() );
        mGraphicContext.mContext->WaitIdle( mGraphicContext.mContext->GetGraphicsQueue() );
    }

    void VkTexture2D::TransitionImageLayout( VkImageLayout aOldLayout, VkImageLayout aNewLayout )
    {
        Ref<Internal::sVkCommandBufferObject> lCommandBufferObject =
            SE::Core::New<Internal::sVkCommandBufferObject>( mGraphicContext.mContext );
        lCommandBufferObject->Begin( VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT );
        // Ref<sVkCommandBufferObject> lCommandBufferObject = mGraphicContext.BeginSingleTimeCommands();
        lCommandBufferObject->ImageMemoryBarrier( mVkImage, aOldLayout, aNewLayout, mSpec.mMipLevels, 1 );
        // mGraphicContext.EndSingleTimeCommands( lCommandBufferObject );
        lCommandBufferObject->End();
        lCommandBufferObject->SubmitTo( mGraphicContext.mContext->GetGraphicsQueue() );
        mGraphicContext.mContext->WaitIdle( mGraphicContext.mContext->GetGraphicsQueue() );
    }

    void VkTexture2D::GetTextureData( TextureData2D &mTextureData )
    {
        uint32_t    lByteSize = mSpec.mWidth * mSpec.mHeight * sizeof( uint32_t );
        VkGpuBuffer lStagingBuffer( mGraphicContext, eBufferBindType::UNKNOWN, true, false, false, true, lByteSize );

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
        Ref<Internal::sVkCommandBufferObject> lCommandBufferObject =
            SE::Core::New<Internal::sVkCommandBufferObject>( mGraphicContext.mContext );
        lCommandBufferObject->Begin( VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT );
        // Ref<sVkCommandBufferObject> lCommandBufferObject = mGraphicContext.BeginSingleTimeCommands();
        lCommandBufferObject->CopyImage( mVkImage, lStagingBuffer.mVkBuffer, lBufferCopyRegions, 0 );
        // mGraphicContext.EndSingleTimeCommands( lCommandBufferObject );
        lCommandBufferObject->End();
        lCommandBufferObject->SubmitTo( mGraphicContext.mContext->GetGraphicsQueue() );
        mGraphicContext.mContext->WaitIdle( mGraphicContext.mContext->GetGraphicsQueue() );
        TransitionImageLayout( VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL );

        uint8_t *lPixelData = lStagingBuffer.Map<uint8_t>( lByteSize, 0 );

        sImageData lImageDataStruct{};
        lImageDataStruct.mFormat    = mSpec.mFormat;
        lImageDataStruct.mWidth     = mSpec.mWidth;
        lImageDataStruct.mHeight    = mSpec.mHeight;
        lImageDataStruct.mByteSize  = lByteSize;
        lImageDataStruct.mPixelData = std::vector<uint8_t>( lPixelData, lPixelData + lByteSize );

        Core::sTextureCreateInfo lTextureCreateInfo{};
        lTextureCreateInfo.mMipLevels = 1;
        mTextureData                  = TextureData2D( lTextureCreateInfo, lImageDataStruct );
    }
} // namespace SE::Graphics
