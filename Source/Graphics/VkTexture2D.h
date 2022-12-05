#pragma once

#include <memory>

#include <gli/gli.hpp>
#include <vulkan/vulkan.h>

#include "Core/CUDA/Texture/TextureData.h"
#include "Core/Memory.h"
#include "Core/Types.h"

#include "Core/CUDA/Texture/ColorFormat.h"
#include "Core/CUDA/Texture/TextureTypes.h"
#include "Core/CUDA/Texture/VkTexture2D.h"

#include "Buffer.h"
#include "GraphicContext.h"

namespace SE::Graphics
{
    using namespace SE::Core;

    /** @brief */
    class VkTexture2D : public Cuda::Texture2D
    {
      public:
        Core::TextureData::sCreateInfo mSpec;

        /** @brief */
        VkTexture2D( GraphicContext &aGraphicContext, Core::TextureData::sCreateInfo &aTextureImageDescription, bool aIsHostVisible,
                     bool aIsGraphicsOnly, bool aIsTransferSource, bool aIsTransferDestination );

        /** @brief */
        VkTexture2D( GraphicContext &aGraphicContext, TextureData2D &aCubeMapData, bool aIsHostVisible, bool aIsGraphicsOnly,
                     bool aIsTransferSource );

        /** @brief */
        ~VkTexture2D() = default;

      private:
        // void CreateImageView();
        void ConfigureExternalMemoryHandle();

        VkMemoryPropertyFlags MemoryProperties();
        VkImageUsageFlags     ImageUsage();
        VkImage               CreateImage();
        VkDeviceMemory        AllocateMemory();
        void                  BindMemory();

        void CopyBufferToImage( VkGpuBuffer &a_Buffer );
        void TransitionImageLayout( VkImageLayout aOldLayout, VkImageLayout aNewLayout );

      private:
        GraphicContext mGraphicContext{};

        VkSampleCountFlagBits mSampleCount           = false;
        bool                  mIsHostVisible         = false;
        bool                  mIsGraphicsOnly        = false;
        bool                  mIsTransferSource      = false;
        bool                  mIsTransferDestination = false;

        VkImage        mVkImage    = VK_NULL_HANDLE;
        VkDeviceMemory mVkMemory   = VK_NULL_HANDLE;
        size_t         mMemorySize = 0;

        cudaExternalMemory_t mExternalMemoryHandle = nullptr;
    };
} // namespace SE::Graphics
