#pragma once

#include <memory>

#include <gli/gli.hpp>
#include <vulkan/vulkan.h>

#include "Core/CUDA/Texture/TextureData.h"
#include "Core/Memory.h"
#include "Core/Types.h"

#include "Core/CUDA/Texture/ColorFormat.h"
#include "Core/CUDA/Texture/Texture2D.h"
#include "Core/CUDA/Texture/TextureTypes.h"

// #include "Core/GraphicContext/GraphicContext.h"

#include "VkGraphicContext.h"
#include "VkTexture2D.h"

namespace SE::Graphics
{
    using namespace SE::Core;
    using namespace SE::Graphics::Internal;

    /** @brief */
    class VkSampler2D : public Cuda::TextureSampler2D
    {
      public:
        /** @brief */
        VkSampler2D( Ref<VkGraphicContext> aGraphicContext, Ref<VkTexture2D> aTextureData, sTextureSamplingInfo const &aSamplingSpec );

        /** @brief */
        ~VkSampler2D() = default;

        Ref<VkTexture2D> GetTexture() { return mTextureData; }
        VkImageView      GetImageView() { return mVkImageView; }
        VkSampler        GetSampler() { return mVkImageSampler; }

      private:
        Ref<VkGraphicContext> mGraphicContext{};
        Ref<VkTexture2D>      mTextureData = nullptr;

        VkImageView mVkImageView    = VK_NULL_HANDLE;
        VkSampler   mVkImageSampler = VK_NULL_HANDLE;

        friend class VkTexture2D;
    };
} // namespace SE::Graphics
