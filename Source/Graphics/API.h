#pragma once

#include "Interface/IGraphicBuffer.h"
#include "Interface/IGraphicContext.h"
#include "Interface/IGraphicsPipeline.h"
#include "Interface/IRenderContext.h"
#include "Interface/ISwapChain.h"

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
                                      bool aIsHostVisible, bool aIsGraphicsOnly, bool aIsTransferSource, bool aIsTransferDestination );

    Ref<IGraphicsPipeline> CreateGraphicsPipeline( Ref<IGraphicContext> aGraphicContext, Ref<IRenderContext> aRenderContext,
                                                   ePrimitiveTopology aTopology );

    Ref<ISwapChain> CreateSwapChain( Ref<IGraphicContext> aGraphicContext, Ref<IWindow> aWindow );

    Ref<IRenderContext> CreateRenderContext( Ref<IGraphicContext> aGraphicContext, Ref<ISwapChain> aWindow );
    Ref<IRenderContext> CreateRenderContext( Ref<IGraphicContext> aGraphicContext, Ref<IRenderTarget> aWindow );

} // namespace SE::Graphics