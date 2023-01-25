#pragma once

#include <memory>

#include <vulkan/vulkan.h>

#include "Core/Memory.h"
#include "Core/Types.h"

#include "VkGpuBuffer.h"
#include "VkGraphicContext.h"

#include "Graphics/Interface/ITextureCubeMap.h"

namespace SE::Graphics
{
    using namespace SE::Core;

    /** @brief */
    class VkTextureCubeMap : public ITextureCubeMap
    {
        friend class VkSamplerCubeMap;
        friend class VkRenderTarget;

      public:
        /** @brief */
        VkTextureCubeMap( Ref<VkGraphicContext> aGraphicContext, Core::sTextureCreateInfo &aTextureImageDescription,
                          uint8_t aSampleCount, bool aIsHostVisible, bool aIsGraphicsOnly, bool aIsTransferSource,
                          bool aIsTransferDestination );

        /** @brief */
        VkTextureCubeMap( Ref<VkGraphicContext> aGraphicContext, TextureDataCubeMap &aCubeMapData )
            : VkTextureCubeMap( aGraphicContext, aCubeMapData, 1, false, true, true )
        {
        }

        /** @brief */
        VkTextureCubeMap( Ref<VkGraphicContext> aGraphicContext, TextureDataCubeMap &aCubeMapData, uint8_t aSampleCount,
                          bool aIsHostVisible, bool aIsGraphicsOnly, bool aIsTransferSource );

        /** @brief */
        VkTextureCubeMap( Ref<VkGraphicContext> aGraphicContext, Core::sTextureCreateInfo &aTextureImageDescription,
                          VkImage aExternalImage );

        /** @brief */
        ~VkTextureCubeMap();

        void GetPixelData( TextureDataCubeMap &mTextureData );
        void GetPixelData( TextureDataCubeMap &mTextureData, eCubeFace aFace );
        void SetPixelData( Ref<IGraphicBuffer> aBuffer );
        void SetPixelData( eCubeFace aFace, Ref<IGraphicBuffer> aBuffer );
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
