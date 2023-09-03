#pragma once

#include <memory>

#include <vulkan/vulkan.h>

#include "Core/Memory.h"
#include "Core/Types.h"

#include "VkGpuBuffer.h"
#include "VkGraphicContext.h"

#include "Graphics/Interface/ITexture2D.h"

namespace SE::Graphics
{
    using namespace SE::Core;

    /** @brief */
    class VkTexture2D : public ITexture2D
    {
        friend class VkSampler2D;
        friend class VkSamplerCubeMap;
        friend class VkRenderTarget;

      public:
        /** @brief */
        VkTexture2D( ref_t<IGraphicContext> aGraphicContext, Core::sTextureCreateInfo &aTextureImageDescription, uint8_t aSampleCount,
                     bool aIsHostVisible, bool aIsGraphicsOnly, bool aIsTransferSource, bool aIsTransferDestination );

        /** @brief */
        VkTexture2D( ref_t<IGraphicContext> aGraphicContext, TextureData2D &aTextureData )
            : VkTexture2D( aGraphicContext, aTextureData, 1, false, true, true )
        {
        }

        /** @brief */
        VkTexture2D( ref_t<IGraphicContext> aGraphicContext, TextureData2D &aTextureData, uint8_t aSampleCount, bool aIsHostVisible,
                     bool aIsGraphicsOnly, bool aIsTransferSource );

        /** @brief */
        VkTexture2D( ref_t<IGraphicContext> aGraphicContext, Core::sTextureCreateInfo &aTextureImageDescription,
                     VkImage aExternalImage );

        /** @brief */
        ~VkTexture2D();

        void GetPixelData( TextureData2D &mTextureData );
        void SetPixelData( ref_t<IGraphicBuffer> a_Buffer );
        void TransitionImageLayout( VkImageLayout aOldLayout, VkImageLayout aNewLayout );

      private:
        void ConfigureExternalMemoryHandle();

        VkMemoryPropertyFlags MemoryProperties();
        VkImageUsageFlags     ImageUsage();

        void CreateImage();
        void AllocateMemory();
        void BindMemory();

      private:
        VkImage        mVkImage    = VK_NULL_HANDLE;
        VkDeviceMemory mVkMemory   = VK_NULL_HANDLE;
        size_t         mMemorySize = 0;

        cudaExternalMemory_t mExternalMemoryHandle = nullptr;
    };
} // namespace SE::Graphics
