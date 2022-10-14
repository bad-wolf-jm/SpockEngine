#include "Buffer.h"

#include <fmt/core.h>
#include <set>
#include <unordered_set>

// #include "VkCoreMacros.h"

#ifdef APIENTRY
#    undef APIENTRY
#endif

// clang-format off
#include <windows.h>
#include <vulkan/vulkan_win32.h>
// clang-format on

namespace LTSE::Graphics
{
    Buffer::Buffer( GraphicContext &aGraphicContext, eBufferBindType aType, bool aIsHostVisible, bool aIsCudaShareable, bool aIsTransferSource, bool aIsTransferDestination,
                    size_t aSize )
        : mGraphicContext{ aGraphicContext }
        , mSize{ aSize }
        , mType{ aType }
        , mIsHostVisible{ aIsHostVisible }
        , mIsTransferSource{ aIsTransferSource }
        , mIsTransferDestination{ aIsTransferDestination }
        , mIsCudaShareable{ aIsCudaShareable }

    {
        VkBufferUsageFlags lBufferFlags = (VkBufferUsageFlags)aType;
        if( mIsTransferSource )
            lBufferFlags |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

        if( mIsTransferDestination )
            lBufferFlags |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;

        mVkObject = mGraphicContext.mContext->CreateBuffer( lBufferFlags, mSize, mIsHostVisible, mIsCudaShareable );
        mVkMemory = mGraphicContext.mContext->AllocateMemory( mVkObject, mSize, mIsHostVisible, mIsCudaShareable );
        mGraphicContext.mContext->BindMemory( mVkObject, mVkMemory );
    }

    void *Buffer::GetMemoryHandle() { return mGraphicContext.mContext->GetSharedMemoryHandle( mVkMemory ); }

    Buffer::~Buffer()
    {
        mGraphicContext.mContext->DestroyBuffer( mVkObject );
        mGraphicContext.mContext->FreeMemory( mVkMemory );
    }

    void Buffer::Upload( void *aData, size_t aSize, size_t aOffset )
    {
        if( ( aOffset + aSize ) > mSize )
            throw std::runtime_error( fmt::format( "Attempted to copy an array of size {} into a buffer if size {}", aSize, mSize ).c_str() );

        if( !mIsHostVisible )
        {
            auto lStagingBuffer = Buffer( mGraphicContext, eBufferBindType::UNKNOWN, true, false, true, false, aSize );
            lStagingBuffer.Upload( aData, aSize, 0 );

            Ref<Internal::sVkCommandBufferObject> l_CommandBufferObject = mGraphicContext.BeginSingleTimeCommands();
            l_CommandBufferObject->CopyBuffer( lStagingBuffer.mVkObject, 0, lStagingBuffer.mSize, mVkObject, aOffset );
            mGraphicContext.EndSingleTimeCommands( l_CommandBufferObject );
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

    static VkDeviceSize g_BufferMemoryAlignment = 256;
    void Buffer::Resize( size_t a_NewSizeInBytes )
    {
        if( a_NewSizeInBytes <= mSize )
            return;

        VkDeviceSize vertex_buffer_size_aligned = ( ( a_NewSizeInBytes - 1 ) / g_BufferMemoryAlignment + 1 ) * g_BufferMemoryAlignment;

        mGraphicContext.mContext->DestroyBuffer( mVkObject );
        mGraphicContext.mContext->FreeMemory( mVkMemory );

        mSize                           = vertex_buffer_size_aligned;
        VkBufferUsageFlags lBufferFlags = (VkBufferUsageFlags)mType;
        if( mIsTransferSource )
            lBufferFlags |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

        if( mIsTransferDestination )
            lBufferFlags |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;

        mVkObject = mGraphicContext.mContext->CreateBuffer( lBufferFlags, mSize, mIsHostVisible, mIsCudaShareable );
        mVkMemory = mGraphicContext.mContext->AllocateMemory( mVkObject, mSize, mIsHostVisible, mIsCudaShareable );
        mGraphicContext.mContext->BindMemory( mVkObject, mVkMemory );
    }

} // namespace LTSE::Graphics