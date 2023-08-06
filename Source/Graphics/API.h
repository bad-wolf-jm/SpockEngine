#pragma once

#include "Interface/IDescriptorSetLayout.h"
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

    ref_t<IGraphicContext> CreateGraphicContext( uint32_t aSampleCount );

    ref_t<IGraphicBuffer> CreateBuffer( ref_t<IGraphicContext> aGraphicContext, eBufferType aType, bool aIsHostVisible,
                                        bool aIsGraphicsOnly, bool aIsTransferSource, bool aIsTransferDestination, size_t aSize );

    ref_t<IGraphicBuffer> CreateBuffer( ref_t<IGraphicContext> aGraphicContext, bool aIsHostVisible, bool aIsGraphicsOnly,
                                        bool aIsTransferSource, bool aIsTransferDestination, size_t aSize );

    template <typename _Ty>
    ref_t<IGraphicBuffer> CreateBuffer( ref_t<IGraphicContext> aGraphicContext, vector_t<_Ty> aData, eBufferType aType,
                                        bool aIsHostVisible, bool aIsGraphicsOnly, bool aIsTransferSource,
                                        bool aIsTransferDestination )
    {
        return CreateBuffer( aGraphicContext, aData.data(), aData.size(), aType, aIsHostVisible, aIsGraphicsOnly, aIsTransferSource,
                             aIsTransferDestination );
    }

    template <typename _Ty>
    ref_t<IGraphicBuffer> CreateBuffer( ref_t<IGraphicContext> aGraphicContext, _Ty *aData, size_t aSize, eBufferType aType,
                                        bool aIsHostVisible, bool aIsGraphicsOnly, bool aIsTransferSource,
                                        bool aIsTransferDestination )
    {
        return CreateBuffer( aGraphicContext, (uint8_t *)aData, aSize * sizeof( _Ty ), aType, aIsHostVisible, aIsGraphicsOnly,
                             aIsTransferSource, aIsTransferDestination );
    }

    template <>
    ref_t<IGraphicBuffer> CreateBuffer( ref_t<IGraphicContext> aGraphicContext, uint8_t *aData, size_t aSize, eBufferType aType,
                                        bool aIsHostVisible, bool aIsGraphicsOnly, bool aIsTransferSource,
                                        bool aIsTransferDestination );

    ref_t<ITexture2D> CreateTexture2D( ref_t<IGraphicContext> aGraphicContext, sTextureCreateInfo &aTextureImageDescription,
                                       uint8_t aSampleCount, bool aIsHostVisible, bool aIsGraphicsOnly, bool aIsTransferSource,
                                       bool aIsTransferDestination );

    ref_t<ITexture2D> CreateTexture2D( ref_t<IGraphicContext> aGraphicContext, TextureData2D &aTextureData );

    ref_t<ITexture2D> CreateTexture2D( ref_t<IGraphicContext> aGraphicContext, TextureData2D &aTextureData, uint8_t aSampleCount,
                                       bool aIsHostVisible, bool aIsGraphicsOnly, bool aIsTransferSource );

    ref_t<ISampler2D> CreateSampler2D( ref_t<IGraphicContext> aGraphicContext, ref_t<ITexture2D> aTextureData,
                                       sTextureSamplingInfo const &aSamplingSpec );

    ref_t<IGraphicsPipeline> CreateGraphicsPipeline( ref_t<IGraphicContext> aGraphicContext, ref_t<IRenderContext> aRenderContext,
                                                     ePrimitiveTopology aTopology );

    ref_t<ISwapChain> CreateSwapChain( ref_t<IGraphicContext> aGraphicContext, ref_t<IWindow> aWindow );

    ref_t<IRenderContext> CreateRenderContext( ref_t<IGraphicContext> aGraphicContext, ref_t<ISwapChain> aWindow );
    ref_t<IRenderContext> CreateRenderContext( ref_t<IGraphicContext> aGraphicContext, ref_t<IRenderTarget> aWindow );

    ref_t<IDescriptorSetLayout> CreateDescriptorSetLayout( ref_t<IGraphicContext> aGraphicContext, bool aUnbounded = false,
                                                           uint32_t aCount = 1 );

} // namespace SE::Graphics