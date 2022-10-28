#pragma once

#include <memory>

#include "Core/Memory.h"
#include "Core/Platform/ViewportClient.h"

// #include "VkContext.h"
#include "GraphicContext.h"

// #include <vulkan/vulkan.h>

namespace LTSE::Graphics
{
    using namespace LTSE::Core;
    using namespace LTSE::Graphics::Internal;

    struct Buffer
    {
        eBufferBindType mType                  = eBufferBindType::UNKNOWN;
        bool            mIsHostVisible         = true;
        bool            mIsTransferSource      = true;
        bool            mIsTransferDestination = true;
        bool            mIsCudaShareable       = true;

        VkBuffer       mVkObject = VK_NULL_HANDLE;
        VkDeviceMemory mVkMemory = VK_NULL_HANDLE;

        Buffer()           = default;
        Buffer( Buffer & ) = default;

        Buffer( GraphicContext &aGraphicContext, eBufferBindType aType, bool aIsHostVisible, bool aIsCudaShareable,
            bool aIsTransferSource, bool aIsTransferDestination, size_t aSize );

        template <typename _Ty>
        Buffer( GraphicContext &aGraphicContext, std::vector<_Ty> aData, eBufferBindType aType, bool aIsHostVisible,
            bool aIsCudaShareable, bool aIsTransferSource, bool aIsTransferDestination )
            : Buffer( aGraphicContext, aType, aIsHostVisible, aIsCudaShareable, aIsTransferSource, aIsTransferDestination,
                  aData.size() * sizeof( _Ty ) )
        {
            Upload( aData );
        }

        template <typename _Ty>
        Buffer( GraphicContext &aGraphicContext, _Ty *aData, size_t aSize, eBufferBindType aType, bool aIsHostVisible,
            bool aIsCudaShareable, bool aIsTransferSource, bool aIsTransferDestination )
            : Buffer( aGraphicContext, aType, aIsHostVisible, aIsCudaShareable, aIsTransferSource, aIsTransferDestination,
                  aSize * sizeof( _Ty ) )
        {
            Upload( aData, aSize );
        }

        ~Buffer();

        void *Buffer::GetMemoryHandle();

        template <typename _MapType>
        _MapType *Map( size_t aSize, size_t aOffset )
        {
            return mGraphicContext.mContext->MapMemory<_MapType>( mVkMemory, aSize, aOffset );
        }
        void Unmap() { mGraphicContext.mContext->UnmapMemory( mVkMemory ); }

        template <typename _MapType>
        void Upload( std::vector<_MapType> aData )
        {
            Upload( aData.data(), aData.size(), 0 );
        }

        template <typename _MapType>
        void Upload( std::vector<_MapType> aData, size_t aOffset )
        {
            Upload( aData.data(), aData.size(), aOffset );
        }

        template <typename _MapType>
        void Upload( _MapType *aData, size_t aSize )
        {
            Upload( aData, aSize, 0 );
        }

        template <typename _MapType>
        void Upload( _MapType *aData, size_t aSize, size_t aOffset )
        {
            Upload( reinterpret_cast<void *>( aData ), aSize * sizeof( _MapType ), aOffset );
        }

        void Upload( void *aData, size_t aSize, size_t aOffset );
        void Copy( Ref<Buffer> aSource, size_t aOffset );

        template <typename _Ty>
        void Write( _Ty a_Value, size_t offset = 0 )
        {
            Upload( reinterpret_cast<void *>( &a_Value ), sizeof( _Ty ), offset );
        }

        template <typename T>
        size_t SizeAs()
        {
            return mSize / sizeof( T );
        }

        void Resize( size_t a_NewSizeInBytes );

      private:
        GraphicContext mGraphicContext;

        size_t mSize = 0;
    };
} // namespace LTSE::Graphics
