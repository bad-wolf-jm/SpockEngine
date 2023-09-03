#pragma once

#include <memory>

#include "Core/Memory.h"

#include "IGraphicContext.h"
#include "IGraphicResource.h"

#include "Core/CUDA/Array/PointerView.h"

namespace SE::Graphics
{
    using namespace SE::Core;

    enum class eBufferType : uint32_t
    {
        VERTEX_BUFFER  = 0,
        INDEX_BUFFER   = 1,
        STORAGE_BUFFER = 2,
        UNIFORM_BUFFER = 3,
        UNKNOWN        = 4
    };

    class IGraphicBuffer : public IGraphicResource, public Cuda::Internal::sGPUDevicePointerView
    {
      public:
        eBufferType mType = eBufferType::UNKNOWN;

        IGraphicBuffer()                   = default;
        IGraphicBuffer( IGraphicBuffer & ) = default;

        IGraphicBuffer( ref_t<IGraphicContext> aGraphicContext, eBufferType aType, bool aIsHostVisible, bool aIsGraphicsOnly,
                        bool aIsTransferSource, bool aIsTransferDestination, size_t aSize )
            : IGraphicResource( aGraphicContext, aIsHostVisible, aIsGraphicsOnly, aIsTransferSource, aIsTransferDestination, aSize )
            , Cuda::Internal::sGPUDevicePointerView( aSize, nullptr )
            , mType{ aType }
        {
        }

        IGraphicBuffer( ref_t<IGraphicContext> aGraphicContext, bool aIsHostVisible, bool aIsGraphicsOnly, bool aIsTransferSource,
                        bool aIsTransferDestination, size_t aSize )
            : IGraphicBuffer( aGraphicContext, eBufferType::UNKNOWN, aIsHostVisible, aIsGraphicsOnly, aIsTransferSource,
                              aIsTransferDestination, aSize )
        {
        }

        template <typename _Ty>
        IGraphicBuffer( ref_t<IGraphicContext> aGraphicContext, std::vector<_Ty> aData, eBufferType aType, bool aIsHostVisible,
                        bool aIsGraphicsOnly, bool aIsTransferSource, bool aIsTransferDestination )
            : IGraphicBuffer( aGraphicContext, aData.data(), aData.size(), aType, aIsHostVisible, aIsGraphicsOnly, aIsTransferSource,
                              aIsTransferDestination )
        {
        }

        template <typename _Ty>
        IGraphicBuffer( ref_t<IGraphicContext> aGraphicContext, _Ty *aData, size_t aSize, eBufferType aType, bool aIsHostVisible,
                        bool aIsGraphicsOnly, bool aIsTransferSource, bool aIsTransferDestination )
            : IGraphicBuffer( aGraphicContext, aType, aIsHostVisible, aIsGraphicsOnly, aIsTransferSource, aIsTransferDestination,
                              aSize * sizeof( _Ty ) )
        {
        }

        ~IGraphicBuffer() = default;

        template <typename _MapType>
        void Upload( std::vector<_MapType> aData )
        {
            Upload( aData, 0 );
        }

        template <typename _Ty>
        void Upload( std::vector<_Ty> aData, size_t aOffset )
        {
            Upload( aData.data(), aData.size(), aOffset );
        }

        template <typename _Ty>
        void Upload( _Ty *aData, size_t aSize )
        {
            Upload( aData, aSize, 0 );
        }

        template <typename _Ty>
        void Upload( _Ty *aData, size_t aSize, size_t aOffset )
        {
            DoUpload( reinterpret_cast<void *>( aData ), aSize * sizeof( _Ty ), aOffset * sizeof( _Ty ) );
        }

        template <typename _Ty>
        void Write( _Ty aValue, size_t aIndex = 0 )
        {
            DoUpload( reinterpret_cast<void *>( &aValue ), sizeof( _Ty ), aIndex * sizeof( _Ty ) );
        }

        virtual void Allocate( size_t aSizeInBytes )                       = 0;
        virtual void Resize( size_t aNewSizeInBytes )                      = 0;
        virtual void Copy( ref_t<IGraphicBuffer> aSource, size_t aOffset )   = 0;
        virtual void DoUpload( void *aData, size_t aSize, size_t aOffset ) = 0;
    };
} // namespace SE::Graphics
