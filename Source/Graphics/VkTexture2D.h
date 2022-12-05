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
        VkTexture2D( GraphicContext &aGraphicContext, Core::TextureData::sCreateInfo &aTextureImageDescription, bool aIsHostVisible );

        /** @brief */
        VkTexture2D( GraphicContext &aGraphicContext, TextureData2D &aCubeMapData );

        /** @brief */
        ~VkTexture2D() = default;

      private:
        // void CreateImageView();
        void CopyBufferToImage( Buffer &a_Buffer );
        void TransitionImageLayout( VkImageLayout oldLayout, VkImageLayout newLayout );

      private:
        GraphicContext mGraphicContext{};

        VkImage        mVkImage    = VK_NULL_HANDLE;
        VkDeviceMemory mVkMemory   = VK_NULL_HANDLE;
        size_t         mMemorySize = 0;

        // Ref<Internal::sVkImageObject>     mTextureImageObject = nullptr;
        // Ref<Internal::sVkImageViewObject> mTextureView        = nullptr;
    };
} // namespace SE::Graphics
