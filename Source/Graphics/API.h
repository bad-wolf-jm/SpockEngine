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

    Ref<IGraphicContext> CreateGraphicContext( uint32_t aSampleCount );

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

    Ref<ITexture2D> CreateTexture2D( Ref<IGraphicContext> aGraphicContext, sTextureCreateInfo &aTextureImageDescription,
                                     uint8_t aSampleCount, bool aIsHostVisible, bool aIsGraphicsOnly, bool aIsTransferSource,
                                     bool aIsTransferDestination );

    Ref<ITexture2D> CreateTexture2D( Ref<IGraphicContext> aGraphicContext, TextureData2D &aTextureData );

    Ref<ITexture2D> CreateTexture2D( Ref<IGraphicContext> aGraphicContext, TextureData2D &aTextureData, uint8_t aSampleCount,
                                     bool aIsHostVisible, bool aIsGraphicsOnly, bool aIsTransferSource );

    Ref<ISampler2D> CreateSampler2D( Ref<IGraphicContext> aGraphicContext, Ref<ITexture2D> aTextureData, uint32_t aLayer,
                                     sTextureSamplingInfo const &aSamplingSpec );

    Ref<ISampler2D> CreateSampler2D( Ref<IGraphicContext> aGraphicContext, Ref<ITexture2D> aTextureData,
                                     sTextureSamplingInfo const &aSamplingSpec );

    Ref<ISampler2D> CreateSampler2D( Ref<IGraphicContext> aGraphicContext, Ref<ITexture2D> aTextureData );

    Ref<ISamplerCubeMap> CreateSamplerCubeMap( Ref<IGraphicContext> aGraphicContext, Ref<ITexture2D> aTextureData,
                                               sTextureSamplingInfo const &aSamplingSpec );

    Ref<ISamplerCubeMap> CreateSamplerCubeMap( Ref<IGraphicContext> aGraphicContext, Ref<ITexture2D> aTextureData );

    Ref<IGraphicsPipeline> CreateGraphicsPipeline( Ref<IGraphicContext> aGraphicContext, Ref<IRenderContext> aRenderContext,
                                                   ePrimitiveTopology aTopology );

    Ref<ISwapChain>    CreateSwapChain( Ref<IGraphicContext> aGraphicContext, Ref<IWindow> aWindow );
    Ref<IRenderTarget> CreateRenderTarget( Ref<IGraphicContext> aGraphicContext, sRenderTargetDescription const &aSpec );

    Ref<IRenderContext> CreateRenderContext( Ref<IGraphicContext> aGraphicContext, Ref<ISwapChain> aWindow );
    Ref<IRenderContext> CreateRenderContext( Ref<IGraphicContext> aGraphicContext, Ref<IRenderTarget> aWindow );

    Ref<IDescriptorSetLayout> CreateDescriptorSetLayout( Ref<IGraphicContext> aGraphicContext, bool aUnbounded = false,
                                                         uint32_t aCount = 1 );

    Ref<IShaderProgram> CreateShaderProgram( Ref<IGraphicContext> aGraphicContext, eShaderStageTypeFlags aShaderType, int aVersion,
                                             std::string const &aName );
    Ref<IShaderProgram> CreateShaderProgram( Ref<IGraphicContext> aGraphicContext, eShaderStageTypeFlags aShaderType, int aVersion,
                                             std::string const &aName, fs::path const &aCacheRoot );

} // namespace SE::Graphics