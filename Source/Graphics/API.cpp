#include "API.h"

#include "Vulkan/VkGpuBuffer.h"
#include "Vulkan/VkGraphicsPipeline.h"
#include "Vulkan/VkSwapChain.h"
#include "Vulkan/VkRenderContext.h"
#include "Vulkan/VkRenderContext.h"
#include "Vulkan/VkSwapChainRenderContext.h"

namespace SE::Graphics
{
    static eGraphicsAPI gApi = eGraphicsAPI::VULKAN;

    Ref<IGraphicBuffer> CreateBuffer( Ref<IGraphicContext> aGraphicContext, eBufferType aType, bool aIsHostVisible,
                                      bool aIsGraphicsOnly, bool aIsTransferSource, bool aIsTransferDestination, size_t aSize )
    {
        switch( gApi )
        {
        case eGraphicsAPI::VULKAN:
            return New<VkGpuBuffer>( Cast<VkGraphicContext>( aGraphicContext ), aType, aIsHostVisible, aIsGraphicsOnly,
                                     aIsTransferSource, aIsTransferDestination, aSize );
        case eGraphicsAPI::OPENGL:
        case eGraphicsAPI::DIRECTX:
        default: return nullptr;
        }
    }

    Ref<IGraphicBuffer> CreateBuffer( Ref<IGraphicContext> aGraphicContext, bool aIsHostVisible, bool aIsGraphicsOnly,
                                      bool aIsTransferSource, bool aIsTransferDestination, size_t aSize )
    {
        return CreateBuffer( Cast<VkGraphicContext>( aGraphicContext ), eBufferType::UNKNOWN, aIsHostVisible, aIsGraphicsOnly,
                             aIsTransferSource, aIsTransferDestination, aSize );
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

    Ref<IGraphicsPipeline> CreateGraphicsPipeline( Ref<IGraphicContext> aGraphicContext, Ref<IRenderContext> aRenderContext,
                                                   ePrimitiveTopology aTopology )
    {
        switch( gApi )
        {
        case eGraphicsAPI::VULKAN:
            return New<VkGraphicsPipeline>( Cast<VkGraphicContext>( aGraphicContext ), Cast<VkRenderContext>( aRenderContext ),
                                            aTopology );
        case eGraphicsAPI::OPENGL:
        case eGraphicsAPI::DIRECTX:
        default: return nullptr;
        }
    }

    Ref<ISwapChain> CreateSwapChain( Ref<IGraphicContext> aGraphicContext, Ref<IWindow> aWindow )
    {
        switch( gApi )
        {
        case eGraphicsAPI::VULKAN: return New<VkSwapChain>( aGraphicContext, aWindow );
        case eGraphicsAPI::OPENGL:
        case eGraphicsAPI::DIRECTX:
        default: return nullptr;
        }
    }

    Ref<IRenderContext> CreateRenderContext( Ref<IGraphicContext> aGraphicContext, Ref<ISwapChain> aWindow )
    {
        switch( gApi )
        {
        case eGraphicsAPI::VULKAN: return New<VkSwapChainRenderContext>( aGraphicContext, aWindow );
        case eGraphicsAPI::OPENGL:
        case eGraphicsAPI::DIRECTX:
        default: return nullptr;
        }
    }

    Ref<IRenderContext> CreateRenderContext( Ref<IGraphicContext> aGraphicContext, Ref<IRenderTarget> aWindow )
    {
        switch( gApi )
        {
        case eGraphicsAPI::VULKAN: return New<VkRenderContext>( aGraphicContext, aWindow );
        case eGraphicsAPI::OPENGL:
        case eGraphicsAPI::DIRECTX:
        default: return nullptr;
        }
    }

} // namespace SE::Graphics