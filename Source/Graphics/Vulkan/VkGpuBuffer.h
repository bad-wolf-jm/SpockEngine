#pragma once

#include <memory>

#include "Core/Memory.h"

#include "Core/CUDA/Array/CudaBuffer.h"

#include "Graphics/Interface/IGraphicBuffer.h"
#include "VkGraphicContext.h"

namespace SE::Graphics
{
    using namespace SE::Core;

    class VkGpuBuffer : public IGraphicBuffer
    {
      public:
        VkBuffer       mVkBuffer = VK_NULL_HANDLE;
        VkDeviceMemory mVkMemory = VK_NULL_HANDLE;

        VkGpuBuffer()                = default;
        VkGpuBuffer( VkGpuBuffer & ) = default;

        VkGpuBuffer( ref_t<VkGraphicContext> aGraphicContext, eBufferType aType, bool aIsHostVisible, bool aIsGraphicsOnly,
                     bool aIsTransferSource, bool aIsTransferDestination, size_t aSize );

        VkGpuBuffer( ref_t<VkGraphicContext> aGraphicContext, bool aIsHostVisible, bool aIsGraphicsOnly, bool aIsTransferSource,
                     bool aIsTransferDestination, size_t aSize );

        template <typename _Ty>
        VkGpuBuffer( ref_t<VkGraphicContext> aGraphicContext, vec_t<_Ty> aData, eBufferType aType, bool aIsHostVisible,
                     bool aIsGraphicsOnly, bool aIsTransferSource, bool aIsTransferDestination )
            : VkGpuBuffer( aGraphicContext, aData.data(), aData.size(), aType, aIsHostVisible, aIsGraphicsOnly, aIsTransferSource,
                           aIsTransferDestination )
        {
        }

        template <typename _Ty>
        VkGpuBuffer( ref_t<VkGraphicContext> aGraphicContext, _Ty *aData, size_t aSize, eBufferType aType, bool aIsHostVisible,
                     bool aIsGraphicsOnly, bool aIsTransferSource, bool aIsTransferDestination )
            : VkGpuBuffer( aGraphicContext, aType, aIsHostVisible, aIsGraphicsOnly, aIsTransferSource, aIsTransferDestination,
                           aSize * sizeof( _Ty ) )
        {
            Allocate( mSize );
            Upload( aData, aSize );
        }

        ~VkGpuBuffer();

        template <typename _MapType>
        _MapType *Map( size_t aSize, size_t aOffset )
        {
            return mVkGraphicContext->MapMemory<_MapType>( mVkMemory, aSize, aOffset );
        }
        void Unmap() { mVkGraphicContext->UnmapMemory( mVkMemory ); }

        void Allocate( size_t aSizeInBytes );
        void Resize( size_t aNewSizeInBytes );
        void Copy( ref_t<IGraphicBuffer> aSource, size_t aOffset );
        void DoUpload( void *aData, size_t aSize, size_t aOffset );

      private:
        ref_t<VkGraphicContext> mVkGraphicContext;

        VkDeviceSize         mSizeAligned          = 0;
        cudaExternalMemory_t mExternalMemoryHandle = 0;
    };
} // namespace SE::Graphics
