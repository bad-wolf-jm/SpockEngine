#pragma once

#include <memory>

#include <gli/gli.hpp>
#include <vulkan/vulkan.h>

#include "Core/Memory.h"
#include "Core/Types.h"

#include "Core/CUDA/Texture/ColorFormat.h"
#include "Core/CUDA/Texture/Texture2D.h"
#include "Core/CUDA/Texture/TextureData.h"
#include "Core/CUDA/Texture/TextureTypes.h"

#include "Core/GraphicContext/GraphicContext.h"

#include "VkGpuBuffer.h"

namespace SE::Graphics
{
    using namespace SE::Core;

    /** @brief */
    class VkTexture2D : public Cuda::Texture2D
    {
        friend class VkSampler2D;

      public:
        Core::TextureData::sCreateInfo mSpec;

        /** @brief */
        VkTexture2D( GraphicContext &aGraphicContext, Core::TextureData::sCreateInfo &aTextureImageDescription, uint8_t aSampleCount,
                     bool aIsHostVisible, bool aIsGraphicsOnly, bool aIsTransferSource, bool aIsTransferDestination );

        /** @brief */
        VkTexture2D( GraphicContext &aGraphicContext, TextureData2D &aCubeMapData, uint8_t aSampleCount, bool aIsHostVisible,
                     bool aIsGraphicsOnly, bool aIsTransferSource );

        /** @brief */
        ~VkTexture2D() = default;
        void GetTextureData( TextureData2D &mTextureData );

      private:
        // void CreateImageView();
        void ConfigureExternalMemoryHandle();

        VkMemoryPropertyFlags MemoryProperties();
        VkImageUsageFlags     ImageUsage();

        void CreateImage();
        void AllocateMemory();
        void BindMemory();

        void SetPixelData( VkGpuBuffer &a_Buffer );
        void TransitionImageLayout( VkImageLayout aOldLayout, VkImageLayout aNewLayout );

      private:
        GraphicContext mGraphicContext{};

        VkSampleCountFlagBits mSampleCount           = VK_SAMPLE_COUNT_1_BIT;
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
