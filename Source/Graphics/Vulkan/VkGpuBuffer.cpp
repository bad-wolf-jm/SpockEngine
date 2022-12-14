#include "VkGpuBuffer.h"

#include <fmt/core.h>

#ifdef APIENTRY
#    undef APIENTRY
#endif

// clang-format off
#include <windows.h>
#include <vulkan/vulkan_win32.h>
// clang-format on

#include "VkCommand.h"

namespace SE::Graphics
{
    static constexpr VkDeviceSize gBufferMemoryAlignment = 256;

    VkGpuBuffer::VkGpuBuffer( Ref<VkGraphicContext> aGraphicContext, bool aIsHostVisible, bool aIsGraphicsOnly, bool aIsTransferSource,
                              bool aIsTransferDestination, size_t aSize )
        : VkGpuBuffer::VkGpuBuffer( aGraphicContext, eBufferBindType::UNKNOWN, aIsHostVisible, aIsGraphicsOnly, aIsTransferSource,
                                    aIsTransferDestination, aSize )
    {
    }

    VkGpuBuffer::VkGpuBuffer( Ref<VkGraphicContext> aGraphicContext, eBufferBindType aType, bool aIsHostVisible, bool aIsGraphicsOnly,
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

        mVkBuffer = mGraphicContext->CreateBuffer( lBufferFlags, mSizeAligned, mIsHostVisible, !mIsGraphicsOnly );
        mVkMemory = mGraphicContext->AllocateMemory( mVkBuffer, mSizeAligned, mIsHostVisible, !mIsGraphicsOnly );
        mGraphicContext->BindMemory( mVkBuffer, mVkMemory );

        if( !mIsGraphicsOnly )
        {
            cudaExternalMemoryHandleDesc lExternalMemoryHandleDesc{};
            lExternalMemoryHandleDesc.type                = cudaExternalMemoryHandleTypeOpaqueWin32;
            lExternalMemoryHandleDesc.size                = mSizeAligned;
            lExternalMemoryHandleDesc.handle.win32.handle = (HANDLE)mGraphicContext->GetSharedMemoryHandle( mVkMemory );
            CUDA_ASSERT( cudaImportExternalMemory( &mExternalMemoryHandle, &lExternalMemoryHandleDesc ) );

            cudaExternalMemoryBufferDesc lExternalMemBufferDesc{};
            lExternalMemBufferDesc.offset = 0;
            lExternalMemBufferDesc.flags  = 0;
            lExternalMemBufferDesc.size   = mSizeAligned;
            CUDA_ASSERT(
                cudaExternalMemoryGetMappedBuffer( (void **)&mDevicePointer, mExternalMemoryHandle, &lExternalMemBufferDesc ) );
        }
    }

    VkGpuBuffer::~VkGpuBuffer()
    {
        mGraphicContext->DestroyBuffer( mVkBuffer );
        mGraphicContext->FreeMemory( mVkMemory );

        GPUMemory::Dispose();

        if( mExternalMemoryHandle ) CUDA_ASSERT( cudaDestroyExternalMemory( mExternalMemoryHandle ) );
        mExternalMemoryHandle = 0;
    }

    void VkGpuBuffer::Upload( void *aData, size_t aSize, size_t aOffset )
    {
        if( ( aOffset + aSize ) > mSize )
            throw std::runtime_error(
                fmt::format( "Attempted to copy an array of size {} into a buffer if size {}", aSize, mSize ).c_str() );

        if( !mIsHostVisible )
        {
            auto lStagingBuffer = VkGpuBuffer( mGraphicContext, eBufferBindType::UNKNOWN, true, false, true, false, aSize );
            lStagingBuffer.Upload( aData, aSize, 0 );

            Ref<sVkCommandBufferObject> lCommandBuffer = SE::Core::New<sVkCommandBufferObject>( mGraphicContext );
            lCommandBuffer->Begin( VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT );
            lCommandBuffer->CopyBuffer( lStagingBuffer.mVkBuffer, 0, lStagingBuffer.mSize, mVkBuffer, aOffset );
            lCommandBuffer->End();
            lCommandBuffer->SubmitTo( mGraphicContext->GetGraphicsQueue() );
            mGraphicContext->WaitIdle( mGraphicContext->GetGraphicsQueue() );
        }
        else
        {
            uint8_t *lMappedMemory = Map<uint8_t>( aSize, aOffset );
            if( !lMappedMemory ) throw std::runtime_error( "Could not map memory" );

            memcpy( reinterpret_cast<void *>( lMappedMemory ), aData, aSize );
            Unmap();
        }
    }

    void VkGpuBuffer::Copy( Ref<VkGpuBuffer> aSource, size_t aOffset )
    {

        Ref<sVkCommandBufferObject> lCommandBuffer = SE::Core::New<sVkCommandBufferObject>( mGraphicContext );
        lCommandBuffer->Begin( VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT );
        lCommandBuffer->CopyBuffer( aSource->mVkBuffer, 0, aSource->mSize, mVkBuffer, aOffset );
        lCommandBuffer->End();
        lCommandBuffer->SubmitTo( mGraphicContext->GetGraphicsQueue() );
        mGraphicContext->WaitIdle( mGraphicContext->GetGraphicsQueue() );
    }

    void VkGpuBuffer::Resize( size_t aNewSizeInBytes )
    {
        if( aNewSizeInBytes <= mSize ) return;

        mGraphicContext->DestroyBuffer( mVkBuffer );
        mGraphicContext->FreeMemory( mVkMemory );

        mSize        = aNewSizeInBytes;
        mSizeAligned = ( ( mSize - 1 ) / gBufferMemoryAlignment + 1 ) * gBufferMemoryAlignment;

        VkBufferUsageFlags lBufferFlags = (VkBufferUsageFlags)mType;
        if( mIsTransferSource ) lBufferFlags |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        if( mIsTransferDestination ) lBufferFlags |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;

        mVkBuffer = mGraphicContext->CreateBuffer( lBufferFlags, mSize, mIsHostVisible, !mIsGraphicsOnly );
        mVkMemory = mGraphicContext->AllocateMemory( mVkBuffer, mSize, mIsHostVisible, !mIsGraphicsOnly );
        mGraphicContext->BindMemory( mVkBuffer, mVkMemory );
    }

} // namespace SE::Graphics