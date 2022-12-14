#pragma once

#include <memory>

#include "Core/Memory.h"

#include "Core/CUDA/Array/CudaBuffer.h"
// #include "Core/GraphicContext/GraphicContext.h"

#include "VkGraphicContext.h"

namespace SE::Graphics
{
    using namespace SE::Core;

    class VkGpuBuffer : public Cuda::GPUMemory
    {
      public:
        eBufferBindType mType                  = eBufferBindType::UNKNOWN;
        bool            mIsHostVisible         = true;
        bool            mIsTransferSource      = true;
        bool            mIsTransferDestination = true;
        bool            mIsGraphicsOnly        = true;

        VkBuffer       mVkBuffer = VK_NULL_HANDLE;
        VkDeviceMemory mVkMemory = VK_NULL_HANDLE;

        VkGpuBuffer()                = default;
        VkGpuBuffer( VkGpuBuffer & ) = default;

        VkGpuBuffer( Ref<VkGraphicContext> aGraphicContext, eBufferBindType aType, bool aIsHostVisible, bool aIsGraphicsOnly,
                     bool aIsTransferSource, bool aIsTransferDestination, size_t aSize );

        VkGpuBuffer( Ref<VkGraphicContext> aGraphicContext, bool aIsHostVisible, bool aIsGraphicsOnly, bool aIsTransferSource,
                     bool aIsTransferDestination, size_t aSize );

        template <typename _Ty>
        VkGpuBuffer( Ref<VkGraphicContext> aGraphicContext, std::vector<_Ty> aData, eBufferBindType aType, bool aIsHostVisible,
                     bool aIsGraphicsOnly, bool aIsTransferSource, bool aIsTransferDestination )
            : VkGpuBuffer( aGraphicContext, aType, aIsHostVisible, aIsGraphicsOnly, aIsTransferSource, aIsTransferDestination,
                           aData.size() * sizeof( _Ty ) )
        {
            Upload( aData );
        }

        template <typename _Ty>
        VkGpuBuffer( Ref<VkGraphicContext> aGraphicContext, _Ty *aData, size_t aSize, eBufferBindType aType, bool aIsHostVisible,
                     bool aIsGraphicsOnly, bool aIsTransferSource, bool aIsTransferDestination )
            : VkGpuBuffer( aGraphicContext, aType, aIsHostVisible, aIsGraphicsOnly, aIsTransferSource, aIsTransferDestination,
                           aSize * sizeof( _Ty ) )
        {
            Upload( aData, aSize );
        }

        ~VkGpuBuffer();

        template <typename _MapType>
        _MapType *Map( size_t aSize, size_t aOffset )
        {
            return mGraphicContext->MapMemory<_MapType>( mVkMemory, aSize, aOffset );
        }
        void Unmap() { mGraphicContext->UnmapMemory( mVkMemory ); }

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
        void Copy( Ref<VkGpuBuffer> aSource, size_t aOffset );

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
        Ref<VkGraphicContext> mGraphicContext;

        size_t               mSize                 = 0;
        VkDeviceSize         mSizeAligned          = 0;
        cudaExternalMemory_t mExternalMemoryHandle = 0;
    };
} // namespace SE::Graphics
