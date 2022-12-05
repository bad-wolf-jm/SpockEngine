#pragma once

#include <memory>

#include "Core/Memory.h"

// #include "VkContext.h"
#include "Core/CUDA/Array/CudaBuffer.h"
#include "GraphicContext.h"

// #include <vulkan/vulkan.h>

namespace SE::Graphics
{
    using namespace SE::Core;
    using namespace SE::Graphics::Internal;

    class VkBuffer : public GPUMemory
    {
        eBufferBindType mType                  = eBufferBindType::UNKNOWN;
        bool            mIsHostVisible         = true;
        bool            mIsTransferSource      = true;
        bool            mIsTransferDestination = true;
        bool            mIsGraphicsOnly        = true;

        VkBuffer       mVkObject = VK_NULL_HANDLE;
        VkDeviceMemory mVkMemory = VK_NULL_HANDLE;

        VkBuffer()             = default;
        VkBuffer( VkBuffer & ) = default;

        VkBuffer( GraphicContext &aGraphicContext, eBufferBindType aType, bool aIsHostVisible, bool aIsGraphicsOnly,
                  bool aIsTransferSource, bool aIsTransferDestination, size_t aSize );

        VkBuffer( GraphicContext &aGraphicContext, bool aIsHostVisible, bool aIsGraphicsOnly,
                  bool aIsTransferSource, bool aIsTransferDestination, size_t aSize );

        template <typename _Ty>
        VkBuffer( GraphicContext &aGraphicContext, std::vector<_Ty> aData, eBufferBindType aType, bool aIsHostVisible,
                  bool aIsGraphicsOnly, bool aIsTransferSource, bool aIsTransferDestination )
            : VkBuffer( aGraphicContext, aType, aIsHostVisible, aIsGraphicsOnly, aIsTransferSource, aIsTransferDestination,
                        aData.size() * sizeof( _Ty ) )
        {
            Upload( aData );
        }

        template <typename _Ty>
        VkBuffer( GraphicContext &aGraphicContext, _Ty *aData, size_t aSize, eBufferBindType aType, bool aIsHostVisible,
                  bool aIsGraphicsOnly, bool aIsTransferSource, bool aIsTransferDestination )
            : VkBuffer( aGraphicContext, aType, aIsHostVisible, aIsGraphicsOnly, aIsTransferSource, aIsTransferDestination,
                        aSize * sizeof( _Ty ) )
        {
            Upload( aData, aSize );
        }

        ~VkBuffer();

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
        void Copy( Ref<VkBuffer> aSource, size_t aOffset );

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

        void Resize( size_t aNewSizeInBytes );

      private:
        GraphicContext mGraphicContext;

        size_t               mSize                 = 0;
        VkDeviceSize         mSizeAligned          = 0;
        cudaExternalMemory_t mExternalMemoryHandle = 0;
    };
} // namespace SE::Graphics
