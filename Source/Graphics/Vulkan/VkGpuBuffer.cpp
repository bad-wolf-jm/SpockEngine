#include <memory>

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

    static VkBufferUsageFlags GetVkBufferType( eBufferType aType )
    {
        switch( aType )
        {
        case eBufferType::VERTEX_BUFFER:
            return VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        case eBufferType::INDEX_BUFFER:
            return VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
        case eBufferType::STORAGE_BUFFER:
            return VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        case eBufferType::UNIFORM_BUFFER:
            return VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        case eBufferType::UNKNOWN:
        default:
            return VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        };
    }

    VkGpuBuffer::VkGpuBuffer( ref_t<VkGraphicContext> aGraphicContext, bool aIsHostVisible, bool aIsGraphicsOnly, bool aIsTransferSource,
                              bool aIsTransferDestination, size_t aSize )
        : VkGpuBuffer::VkGpuBuffer( aGraphicContext, eBufferType::UNKNOWN, aIsHostVisible, aIsGraphicsOnly, aIsTransferSource,
                                    aIsTransferDestination, aSize )
    {
        // Allocate( mSize );
    }

    VkGpuBuffer::VkGpuBuffer( ref_t<VkGraphicContext> aGraphicContext, eBufferType aType, bool aIsHostVisible, bool aIsGraphicsOnly,
                              bool aIsTransferSource, bool aIsTransferDestination, size_t aSize )
        : IGraphicBuffer( aGraphicContext, aType, aIsHostVisible, aIsGraphicsOnly, aIsTransferSource, aIsTransferDestination, aSize )
        , mVkGraphicContext{ aGraphicContext }
    {
        Allocate( mSize );
    }

    VkGpuBuffer::~VkGpuBuffer()
    {
        mVkGraphicContext->DestroyBuffer( mVkBuffer );
        mVkGraphicContext->FreeMemory( mVkMemory );

        // GPUMemory::Dispose();
        if( mExternalMemoryHandle )
            CUDA_ASSERT( cudaDestroyExternalMemory( mExternalMemoryHandle ) );
        mExternalMemoryHandle = 0;
    }

    void VkGpuBuffer::Allocate( size_t aSizeInBytes )
    {
        mSizeAligned = ( ( mSize - 1 ) / gBufferMemoryAlignment + 1 ) * gBufferMemoryAlignment;

        VkBufferUsageFlags lBufferFlags = GetVkBufferType( mType );
        ;
        if( mIsTransferSource )
            lBufferFlags |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        if( mIsTransferDestination )
            lBufferFlags |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;

        mVkBuffer = mVkGraphicContext->CreateBuffer( lBufferFlags, mSizeAligned, mIsHostVisible, !mIsGraphicsOnly );
        mVkMemory = mVkGraphicContext->AllocateMemory( mVkBuffer, mSizeAligned, mIsHostVisible, !mIsGraphicsOnly );
        mVkGraphicContext->BindMemory( mVkBuffer, mVkMemory );

        if( !mIsGraphicsOnly )
        {
            cudaExternalMemoryHandleDesc lExternalMemoryHandleDesc{};
            lExternalMemoryHandleDesc.type                = cudaExternalMemoryHandleTypeOpaqueWin32;
            lExternalMemoryHandleDesc.size                = mSizeAligned;
            lExternalMemoryHandleDesc.handle.win32.handle = (HANDLE)mVkGraphicContext->GetSharedMemoryHandle( mVkMemory );
            CUDA_ASSERT( cudaImportExternalMemory( &mExternalMemoryHandle, &lExternalMemoryHandleDesc ) );

            cudaExternalMemoryBufferDesc lExternalMemBufferDesc{};
            lExternalMemBufferDesc.offset = 0;
            lExternalMemBufferDesc.flags  = 0;
            lExternalMemBufferDesc.size   = mSizeAligned;
            CUDA_ASSERT(
                cudaExternalMemoryGetMappedBuffer( (void **)&mDevicePointer, mExternalMemoryHandle, &lExternalMemBufferDesc ) );
        }
    }

    void VkGpuBuffer::DoUpload( void *aData, size_t aSize, size_t aOffset )
    {
        if( ( aOffset + aSize ) > mSize )
            throw std::runtime_error(
                fmt::format( "Attempted to copy an array of size {} into a buffer if size {}", aSize, mSize ).c_str() );

        if( !mIsHostVisible )
        {
            auto lStagingBuffer = VkGpuBuffer( mVkGraphicContext, eBufferType::UNKNOWN, true, false, true, false, aSize );
            lStagingBuffer.DoUpload( aData, aSize, 0 );

            ref_t<sVkCommandBufferObject> lCommandBuffer = SE::Core::New<sVkCommandBufferObject>( mVkGraphicContext );
            lCommandBuffer->Begin( VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT );
            lCommandBuffer->CopyBuffer( lStagingBuffer.mVkBuffer, 0, lStagingBuffer.mSize, mVkBuffer, aOffset );
            lCommandBuffer->End();
            lCommandBuffer->SubmitTo( mVkGraphicContext->GetTransferQueue() );
            mVkGraphicContext->WaitIdle( mVkGraphicContext->GetTransferQueue() );
        }
        else
        {
            uint8_t *lMappedMemory = Map<uint8_t>( aSize, aOffset );
            if( !lMappedMemory )
                throw std::runtime_error( "Could not map memory" );

            memcpy( reinterpret_cast<void *>( lMappedMemory ), aData, aSize );
            Unmap();
        }
    }

    void VkGpuBuffer::Copy( ref_t<IGraphicBuffer> aSource, size_t aOffset )
    {
        auto lSource = std::reinterpret_pointer_cast<VkGpuBuffer>( aSource );

        ref_t<sVkCommandBufferObject> lCommandBuffer = SE::Core::New<sVkCommandBufferObject>( mVkGraphicContext );
        lCommandBuffer->Begin( VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT );
        lCommandBuffer->CopyBuffer( lSource->mVkBuffer, 0, lSource->mSize, mVkBuffer, aOffset );
        lCommandBuffer->End();
        lCommandBuffer->SubmitTo( mVkGraphicContext->GetTransferQueue() );
        mVkGraphicContext->WaitIdle( mVkGraphicContext->GetTransferQueue() );
    }

    void VkGpuBuffer::Resize( size_t aNewSizeInBytes )
    {
        if( aNewSizeInBytes <= mSize )
            return;

        mVkGraphicContext->DestroyBuffer( mVkBuffer );
        mVkGraphicContext->FreeMemory( mVkMemory );

        mSize        = aNewSizeInBytes;
        mSizeAligned = ( ( mSize - 1 ) / gBufferMemoryAlignment + 1 ) * gBufferMemoryAlignment;

        VkBufferUsageFlags lBufferFlags = GetVkBufferType( mType );
        if( mIsTransferSource )
            lBufferFlags |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        if( mIsTransferDestination )
            lBufferFlags |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;

        mVkBuffer = mVkGraphicContext->CreateBuffer( lBufferFlags, mSizeAligned, mIsHostVisible, !mIsGraphicsOnly );
        mVkMemory = mVkGraphicContext->AllocateMemory( mVkBuffer, mSizeAligned, mIsHostVisible, !mIsGraphicsOnly );
        mVkGraphicContext->BindMemory( mVkBuffer, mVkMemory );
    }

} // namespace SE::Graphics