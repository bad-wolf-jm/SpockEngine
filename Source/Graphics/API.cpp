#include "API.h"

#include "Vulkan/VkDescriptorSetLayout.h"
#include "Vulkan/VkGpuBuffer.h"
#include "Vulkan/VkGraphicsPipeline.h"
#include "Vulkan/VkRenderContext.h"
#include "Vulkan/VkShaderProgram.h"
#include "Vulkan/VkSwapChain.h"
#include "Vulkan/VkSwapChainRenderContext.h"

namespace SE::Graphics
{
    static eGraphicsAPI gApi = eGraphicsAPI::VULKAN;
    static fs::path     gShaderCacheRoot;

    void SetShaderCacheFolder( fs::path aPath )
    {
        gShaderCacheRoot = aPath;
    }

    ref_t<IGraphicContext> CreateGraphicContext( uint32_t aSampleCount )
    {
        switch( gApi )
        {
        case eGraphicsAPI::VULKAN:
            return New<VkGraphicContext>( aSampleCount, true );
        case eGraphicsAPI::OPENGL:
        case eGraphicsAPI::DIRECTX:
        default:
            return nullptr;
        }
    }

    ref_t<IGraphicBuffer> CreateBuffer( ref_t<IGraphicContext> aGraphicContext, eBufferType aType, bool aIsHostVisible,
                                      bool aIsGraphicsOnly, bool aIsTransferSource, bool aIsTransferDestination, size_t aSize )
    {
        switch( gApi )
        {
        case eGraphicsAPI::VULKAN:
            return New<VkGpuBuffer>( Cast<VkGraphicContext>( aGraphicContext ), aType, aIsHostVisible, aIsGraphicsOnly,
                                     aIsTransferSource, aIsTransferDestination, aSize );
        case eGraphicsAPI::OPENGL:
        case eGraphicsAPI::DIRECTX:
        default:
            return nullptr;
        }
    }

    ref_t<IGraphicBuffer> CreateBuffer( ref_t<IGraphicContext> aGraphicContext, bool aIsHostVisible, bool aIsGraphicsOnly,
                                      bool aIsTransferSource, bool aIsTransferDestination, size_t aSize )
    {
        return CreateBuffer( Cast<VkGraphicContext>( aGraphicContext ), eBufferType::UNKNOWN, aIsHostVisible, aIsGraphicsOnly,
                             aIsTransferSource, aIsTransferDestination, aSize );
    }

    template <>
    ref_t<IGraphicBuffer> CreateBuffer( ref_t<IGraphicContext> aGraphicContext, uint8_t *aData, size_t aSize, eBufferType aType,
                                      bool aIsHostVisible, bool aIsGraphicsOnly, bool aIsTransferSource, bool aIsTransferDestination )
    {
        auto lNewBuffer =
            CreateBuffer( aGraphicContext, aType, aIsHostVisible, aIsGraphicsOnly, aIsTransferSource, aIsTransferDestination, aSize );
        lNewBuffer->Upload( aData, aSize );

        return lNewBuffer;
    }

    ref_t<ITexture2D> CreateTexture2D( ref_t<IGraphicContext> aGraphicContext, sTextureCreateInfo &aTextureImageDescription,
                                     uint8_t aSampleCount, bool aIsHostVisible, bool aIsGraphicsOnly, bool aIsTransferSource,
                                     bool aIsTransferDestination )
    {
        switch( gApi )
        {
        case eGraphicsAPI::VULKAN:
            return New<VkTexture2D>( aGraphicContext, aTextureImageDescription, aSampleCount, aIsHostVisible, aIsGraphicsOnly,
                                     aIsTransferSource, aIsTransferDestination );
        case eGraphicsAPI::OPENGL:
        case eGraphicsAPI::DIRECTX:
        default:
            return nullptr;
        }
    }

    ref_t<ITexture2D> CreateTexture2D( ref_t<IGraphicContext> aGraphicContext, TextureData2D &aTextureData )
    {
        switch( gApi )
        {
        case eGraphicsAPI::VULKAN:
            return New<VkTexture2D>( aGraphicContext, aTextureData );
        case eGraphicsAPI::OPENGL:
        case eGraphicsAPI::DIRECTX:
        default:
            return nullptr;
        }
    }

    ref_t<ITexture2D> CreateTexture2D( ref_t<IGraphicContext> aGraphicContext, TextureData2D &aTextureData, uint8_t aSampleCount,
                                     bool aIsHostVisible, bool aIsGraphicsOnly, bool aIsTransferSource )
    {
        switch( gApi )
        {
        case eGraphicsAPI::VULKAN:
            return New<VkTexture2D>( aGraphicContext, aTextureData, aSampleCount, aIsHostVisible, aIsGraphicsOnly, aIsTransferSource );
        case eGraphicsAPI::OPENGL:
        case eGraphicsAPI::DIRECTX:
        default:
            return nullptr;
        }
    }

    ref_t<ISampler2D> CreateSampler2D( ref_t<IGraphicContext> aGraphicContext, ref_t<ITexture2D> aTextureData, uint32_t aLayer,
                                     sTextureSamplingInfo const &aSamplingSpec )
    {
        switch( gApi )
        {
        case eGraphicsAPI::VULKAN:
            return New<VkSampler2D>( aGraphicContext, Cast<VkTexture2D>( aTextureData ), aLayer, aSamplingSpec );
        case eGraphicsAPI::OPENGL:
        case eGraphicsAPI::DIRECTX:
        default:
            return nullptr;
        }
    }

    ref_t<ISampler2D> CreateSampler2D( ref_t<IGraphicContext> aGraphicContext, ref_t<ITexture2D> aTextureData,
                                     sTextureSamplingInfo const &aSamplingSpec )
    {
        switch( gApi )
        {
        case eGraphicsAPI::VULKAN:
            return New<VkSampler2D>( aGraphicContext, Cast<VkTexture2D>( aTextureData ), aSamplingSpec );
        case eGraphicsAPI::OPENGL:
        case eGraphicsAPI::DIRECTX:
        default:
            return nullptr;
        }
    }

    ref_t<ISampler2D> CreateSampler2D( ref_t<IGraphicContext> aGraphicContext, ref_t<ITexture2D> aTextureData )
    {
        return CreateSampler2D( aGraphicContext, aTextureData, sTextureSamplingInfo{} );
    }

    ref_t<ISamplerCubeMap> CreateSamplerCubeMap( ref_t<IGraphicContext> aGraphicContext, ref_t<ITexture2D> aTextureData,
                                               sTextureSamplingInfo const &aSamplingSpec )
    {
        switch( gApi )
        {
        case eGraphicsAPI::VULKAN:
            return New<VkSamplerCubeMap>( Cast<VkGraphicContext>( aGraphicContext ), Cast<VkTexture2D>( aTextureData ),
                                          aSamplingSpec );
        case eGraphicsAPI::OPENGL:
        case eGraphicsAPI::DIRECTX:
        default:
            return nullptr;
        }
    }

    ref_t<ISamplerCubeMap> CreateSamplerCubeMap( ref_t<IGraphicContext> aGraphicContext, ref_t<ITexture2D> aTextureData )
    {
        return CreateSamplerCubeMap( aGraphicContext, aTextureData, sTextureSamplingInfo{} );
    }

    ref_t<IGraphicsPipeline> CreateGraphicsPipeline( ref_t<IGraphicContext> aGraphicContext, ref_t<IRenderContext> aRenderContext,
                                                   ePrimitiveTopology aTopology )
    {
        switch( gApi )
        {
        case eGraphicsAPI::VULKAN:
            return New<VkGraphicsPipeline>( Cast<VkGraphicContext>( aGraphicContext ), Cast<VkRenderContext>( aRenderContext ),
                                            aTopology );
        case eGraphicsAPI::OPENGL:
        case eGraphicsAPI::DIRECTX:
        default:
            return nullptr;
        }
    }

    ref_t<ISwapChain> CreateSwapChain( ref_t<IGraphicContext> aGraphicContext, ref_t<IWindow> aWindow )
    {
        switch( gApi )
        {
        case eGraphicsAPI::VULKAN:
            return New<VkSwapChain>( aGraphicContext, aWindow );
        case eGraphicsAPI::OPENGL:
        case eGraphicsAPI::DIRECTX:
        default:
            return nullptr;
        }
    }

    ref_t<IRenderTarget> CreateRenderTarget( ref_t<IGraphicContext> aGraphicContext, sRenderTargetDescription const &aSpec )
    {
        switch( gApi )
        {
        case eGraphicsAPI::VULKAN:
            return New<VkRenderTarget>( Cast<VkGraphicContext>( aGraphicContext ), aSpec );
        case eGraphicsAPI::OPENGL:
        case eGraphicsAPI::DIRECTX:
        default:
            return nullptr;
        }
    }

    ref_t<IRenderContext> CreateRenderContext( ref_t<IGraphicContext> aGraphicContext, ref_t<ISwapChain> aWindow )
    {
        switch( gApi )
        {
        case eGraphicsAPI::VULKAN:
            return New<VkSwapChainRenderContext>( aGraphicContext, aWindow );
        case eGraphicsAPI::OPENGL:
        case eGraphicsAPI::DIRECTX:
        default:
            return nullptr;
        }
    }

    ref_t<IRenderContext> CreateRenderContext( ref_t<IGraphicContext> aGraphicContext, ref_t<IRenderTarget> aWindow )
    {
        switch( gApi )
        {
        case eGraphicsAPI::VULKAN:
            return New<VkRenderContext>( aGraphicContext, aWindow );
        case eGraphicsAPI::OPENGL:
        case eGraphicsAPI::DIRECTX:
        default:
            return nullptr;
        }
    }

    ref_t<IDescriptorSetLayout> CreateDescriptorSetLayout( ref_t<IGraphicContext> aGraphicContext, bool aUnbounded, uint32_t aCount )
    {
        switch( gApi )
        {
        case eGraphicsAPI::VULKAN:
            return New<VkDescriptorSetLayoutObject>( aGraphicContext, aUnbounded, aCount );
        case eGraphicsAPI::OPENGL:
        case eGraphicsAPI::DIRECTX:
        default:
            return nullptr;
        }
    }

    ref_t<IShaderProgram> CreateShaderProgram( ref_t<IGraphicContext> aGraphicContext, eShaderStageTypeFlags aShaderType, int aVersion,
                                             string_t const &aName )
    {
        switch( gApi )
        {
        case eGraphicsAPI::VULKAN:
            return New<VkShaderProgram>( aGraphicContext, aShaderType, aVersion, aName, gShaderCacheRoot );
        case eGraphicsAPI::OPENGL:
        case eGraphicsAPI::DIRECTX:
        default:
            return nullptr;
        }
    }
} // namespace SE::Graphics