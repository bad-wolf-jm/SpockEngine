#pragma once

#include <memory>

#include "Core/Memory.h"

#include "IGraphicContext.h"
#include "IGraphicResource.h"

#include "Core/CUDA/Array/PointerView.h"

namespace SE::Graphics
{
    using namespace SE::Core;
    using namespace SE::Graphics::Internal;

    enum class eBufferType : uint32_t
    {
        VERTEX_BUFFER  = 0,
        INDEX_BUFFER   = 1,
        STORAGE_BUFFER = 2,
        UNIFORM_BUFFER = 3,
        UNKNOWN        = 4
    };

    class IBuffer : public IGraphicResource
    {
      public:
        eBufferType mType = eBufferBindType::UNKNOWN;

        IBuffer()            = default;
        IBuffer( IBuffer & ) = default;

        IBuffer( Ref<IGraphicContext> aGraphicContext, eBufferType aType, bool aIsHostVisible, bool aIsGraphicsOnly,
                 bool aIsTransferSource, bool aIsTransferDestination, size_t aSize )
            : IGraphicResource( aGraphicContext, aIsHostVisible, aIsGraphicsOnly, aIsTransferSource, aIsTransferDestination, aSize )
            , mType{ aType }
        {
            Allocate( mSize );
        }

        IBuffer( Ref<IGraphicContext> aGraphicContext, bool aIsHostVisible, bool aIsGraphicsOnly, bool aIsTransferSource,
                 bool aIsTransferDestination, size_t aSize )
            : IBuffer( aGraphicContext, eBufferType::UNKNOWN, aIsHostVisible, aIsGraphicsOnly, aIsTransferSource,
                       aIsTransferDestination, aSize )

        {
        }

        template <typename _Ty>
        IBuffer( Ref<IGraphicContext> aGraphicContext, std::vector<_Ty> aData, eBufferBindType aType, bool aIsHostVisible,
                 bool aIsGraphicsOnly, bool aIsTransferSource, bool aIsTransferDestination )
            : IBuffer( aGraphicContext, aData.data(), aData.size(), aType, aIsHostVisible, aIsGraphicsOnly, aIsTransferSource,
                       aIsTransferDestination, aData.size() * sizeof( _Ty ) )
        {
        }

        template <typename _Ty>
        IBuffer( Ref<IGraphicContext> aGraphicContext, _Ty *aData, size_t aSize, eBufferBindType aType, bool aIsHostVisible,
                 bool aIsGraphicsOnly, bool aIsTransferSource, bool aIsTransferDestination )
            : IBuffer( aGraphicContext, aType, aIsHostVisible, aIsGraphicsOnly, aIsTransferSource, aIsTransferDestination,
                       aSize * sizeof( _Ty ) )
        {
            Allocate( mSize );
            Upload( aData, aSize );
        }

        ~IBuffer() = default;

        template <typename _MapType>
        void Upload( std::vector<_MapType> aData )
        {
            Upload( aData, 0 );
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
            Upload( reinterpret_cast<void *>( aData ), aSize * sizeof( _MapType ), aOffset * sizeof( _MapType ) );
        }

        template <typename _Ty>
        void Write( _Ty aValue, size_t aIndex = 0 )
        {
            Upload( reinterpret_cast<void *>( &aValue ), sizeof( _Ty ), aIndex * sizeof( _Ty ) );
        }

        virtual void Allocate( size_t aSizeInBytes )                     = 0;
        virtual void Resize( size_t aNewSizeInBytes )                    = 0;
        virtual void Copy( Ref<IBuffer> aSource, size_t aOffset )        = 0;
        virtual void Upload( void *aData, size_t aSize, size_t aOffset ) = 0;
    };
} // namespace SE::Graphics
