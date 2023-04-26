#pragma once

#include "Interface/IGraphicBuffer.h"
#include "Interface/IGraphicContext.h"

namespace SE::Graphics
{
    enum class eGraphicsAPI : uint8_t
    {
        VULKAN,
        OPENGL,
        DIRECTX
    };

    Ref<IGraphicBuffer> CreateBuffer( Ref<IGraphicContext> aGraphicContext, eBufferType aType, bool aIsHostVisible,
                                      bool aIsGraphicsOnly, bool aIsTransferSource, bool aIsTransferDestination, size_t aSize );

    Ref<IGraphicBuffer> CreateBuffer( Ref<IGraphicContext> aGraphicContext, bool aIsHostVisible, bool aIsGraphicsOnly,
                                      bool aIsTransferSource, bool aIsTransferDestination, size_t aSize );

    template <typename _Ty>
    Ref<IGraphicBuffer> CreateBuffer( Ref<IGraphicContext> aGraphicContext, std::vector<_Ty> aData, eBufferType aType,
                                      bool aIsHostVisible, bool aIsGraphicsOnly, bool aIsTransferSource, bool aIsTransferDestination )
    {
        return CreateBuffer( aGraphicContext, aData.data(), aData.size(), aType, aIsHostVisible, aIsGraphicsOnly, aIsTransferSource,
                             aIsTransferDestination );
    }

    template <typename _Ty>
    Ref<IGraphicBuffer> CreateBuffer( Ref<IGraphicContext> aGraphicContext, _Ty *aData, size_t aSize, eBufferType aType,
                                      bool aIsHostVisible, bool aIsGraphicsOnly, bool aIsTransferSource, bool aIsTransferDestination )
    {
        return CreateBuffer( aGraphicContext, (uint8_t *)aData, aSize * sizeof( _Ty ), aType, aIsHostVisible, aIsGraphicsOnly,
                             aIsTransferSource, aIsTransferDestination );
    }

    template <>
    Ref<IGraphicBuffer> CreateBuffer( Ref<IGraphicContext> aGraphicContext, uint8_t *aData, size_t aSize, eBufferType aType,
                                      bool aIsHostVisible, bool aIsGraphicsOnly, bool aIsTransferSource, bool aIsTransferDestination )
    {
        auto lNewBuffer =
            CreateBuffer( aGraphicContext, aType, aIsHostVisible, aIsGraphicsOnly, aIsTransferSource, aIsTransferDestination, aSize );
        lNewBuffer->Upload( aData, aSize );

        return lNewBuffer;
    }
} // namespace SE::Graphics