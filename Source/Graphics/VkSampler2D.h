#pragma once

#include <memory>

#include <gli/gli.hpp>
#include <vulkan/vulkan.h>

#include "Core/CUDA/Texture/TextureData.h"
#include "Core/Memory.h"
#include "Core/Types.h"

#include "Core/CUDA/Texture/ColorFormat.h"
#include "Core/CUDA/Texture/TextureTypes.h"

#include "Buffer.h"
#include "GraphicContext.h"

namespace SE::Graphics
{
    using namespace SE::Core;

    /** @brief */
    class VkSampler2D : public Cuda::TextureSampler2D
    {
      public:
        sTextureSamplingInfo mSamplingSpec;

        /** @brief */
        VkSampler2D( GraphicContext &aGraphicContext, Ref<VkTexture2D> aTextureData, sTextureSamplingInfo const &aSamplingSpec );

        /** @brief */
        ~VkSampler2D() = default;

      private:
        GraphicContext   mGraphicContext{};
        Ref<VkTexture2D> mTextureData = nullptr;

        VkImageView mVkImageView    = VK_NULL_HANDLE;
        VkSampler   mVkImageSampler = VK_NULL_HANDLE;
    };
} // namespace SE::Graphics
