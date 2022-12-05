#include "VkBuffer.h"

#include <fmt/core.h>

#ifdef APIENTRY
#    undef APIENTRY
#endif

// clang-format off
#include <windows.h>
#include <vulkan/vulkan_win32.h>
// clang-format on

namespace SE::Graphics
{
    static constexpr VkDeviceSize gBufferMemoryAlignment = 256;

    VkBuffer::VkBuffer( GraphicContext &aGraphicContext, bool aIsHostVisible, bool aIsGraphicsOnly, bool aIsTransferSource,
                        bool aIsTransferDestination, size_t aSize )
        : VkBuffer::VkBuffer( aGraphicContext, eBufferBindType::UNKNOWN, aIsHostVisible, aIsGraphicsOnly, aIsTransferSource,
                              aIsTransferDestination, aSize )
    {
    }

    VkBuffer::VkBuffer( GraphicContext &aGraphicContext, eBufferBindType aType, bool aIsHostVisible, bool aIsGraphicsOnly,
                        bool aIsTransferSource, bool aIsTransferDestination, size_t aSize )
        : mGraphicContext{ aGraphicContext }
        , mSize{ aSize }
        , mType{ aType }
        , mIsHostVisible{ aIsHostVisible }
        , mIsTransferSource{ aIsTransferSource }
        , mIsTransferDestination{ aIsTransferDestination }
        , mIsGraphicsOnly{ aIsGraphicsOnly }

    {
        mSizeAligned = ( ( mSize - 1 ) / gBufferMemoryAlignment + 1 ) * gBufferMemoryAlignment;

        VkBufferUsageFlags lBufferFlags = (VkBufferUsageFlags)aType;
        if( mIsTransferSource ) lBufferFlags |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        if( mIsTransferDestination ) lBufferFlags |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;

        mVkObject = mGraphicContext.mContext->CreateBuffer( lBufferFlags, mSizeAligned, mIsHostVisible, !mIsGraphicsOnly );
        mVkMemory = mGraphicContext.mContext->AllocateMemory( mVkObject, mSizeAligned, mIsHostVisible, !mIsGraphicsOnly );
        mGraphicContext.mContext->BindMemory( mVkObject, mVkMemory );

        if( !mIsGraphicsOnly )
        {
            cudaExternalMemoryHandleDesc lExternalMemoryHandleDesc{};
            lExternalMemoryHandleDesc.type                = cudaExternalMemoryHandleTypeOpaqueWin32;
            lExternalMemoryHandleDesc.size                = mSizeAligned;
            lExternalMemoryHandleDesc.handle.win32.handle = (HANDLE)mGraphicContext.mContext->GetSharedMemoryHandle( mVkMemory );
            CUDA_ASSERT( cudaImportExternalMemory( &mExternalMemoryHandle, &lExternalMemoryHandleDesc ) );

            cudaExternalMemoryBufferDesc lExternalMemBufferDesc{};
            lExternalMemBufferDesc.offset = aOffset;
            lExternalMemBufferDesc.flags  = 0;
            lExternalMemBufferDesc.size   = mSizeAligned;
            CUDA_ASSERT( cudaExternalMemoryGetMappedBuffer( &mDevicePointer, mExternalMemoryHandle, &lExternalMemBufferDesc ) );
        }
    }

    VkBuffer::~VkBuffer()
    {
        mGraphicContext.mContext->DestroyBuffer( mVkObject );
        mGraphicContext.mContext->FreeMemory( mVkMemory );

        GPUMemory::Dispose();

        if( mExternalMemoryHandle ) CUDA_ASSERT( cudaDestroyExternalMemory( mExternalMemoryHandle ) );
        mExternalMemoryHandle = 0;
    }

    void VkBuffer::Upload( void *aData, size_t aSize, size_t aOffset )
    {
        if( ( aOffset + aSize ) > mSize )
            throw std::runtime_error(
                fmt::format( "Attempted to copy an array of size {} into a buffer if size {}", aSize, mSize ).c_str() );

        if( !mIsHostVisible )
        {
            auto lStagingBuffer = VkBuffer( mGraphicContext, eBufferBindType::UNKNOWN, true, false, true, false, aSize );
            lStagingBuffer.Upload( aData, aSize, 0 );

            Ref<Internal::sVkCommandBufferObject> l_CommandBufferObject = mGraphicContext.BeginSingleTimeCommands();
            l_CommandBufferObject->CopyBuffer( lStagingBuffer.mVkObject, 0, lStagingBuffer.mSize, mVkObject, aOffset );
            mGraphicContext.EndSingleTimeCommands( l_CommandBufferObject );
        }
        else
        {
            uint8_t *lMappedMemory = Map<uint8_t>( aSize, aOffset );
            if( !lMappedMemory ) throw std::runtime_error( "Could not map memory" );

            memcpy( reinterpret_cast<void *>( lMappedMemory ), aData, aSize );
            Unmap();
        }
    }

    void VkBuffer::Copy( Ref<VkBuffer> aSource, size_t aOffset )
    {
        Ref<Internal::sVkCommandBufferObject> l_CommandBufferObject = mGraphicContext.BeginSingleTimeCommands();
        l_CommandBufferObject->CopyBuffer( aSource->mVkObject, 0, aSource->mSize, mVkObject, aOffset );
        mGraphicContext.EndSingleTimeCommands( l_CommandBufferObject );
    }

    void VkBuffer::Resize( size_t aNewSizeInBytes )
    {
        if( aNewSizeInBytes <= mSize ) return;

        mGraphicContext.mContext->DestroyBuffer( mVkObject );
        mGraphicContext.mContext->FreeMemory( mVkMemory );

        mSize = lBufferSizeAligned;

        VkBufferUsageFlags lBufferFlags = (VkBufferUsageFlags)mType;
        if( mIsTransferSource ) lBufferFlags |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        if( mIsTransferDestination ) lBufferFlags |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;

        mVkObject = mGraphicContext.mContext->CreateBuffer( lBufferFlags, mSize, mIsHostVisible, !mIsGraphicsOnly );
        mVkMemory = mGraphicContext.mContext->AllocateMemory( mVkObject, mSize, mIsHostVisible, !mIsGraphicsOnly );
        mGraphicContext.mContext->BindMemory( mVkObject, mVkMemory );
    }

} // namespace SE::Graphics