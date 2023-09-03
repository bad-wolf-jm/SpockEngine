#pragma once

#include <memory>

#include <gli/gli.hpp>
#include <vulkan/vulkan.h>

#include "Core/Memory.h"
#include "Core/Types.h"

#include "Graphics/Interface/ISamplerCubeMap.h"

#include "VkGraphicContext.h"
#include "VkTexture2D.h"

namespace SE::Graphics
{
    using namespace SE::Core;

    /** @brief */
    class VkSamplerCubeMap : public ISamplerCubeMap
    {
      public:
        /** @brief */
        VkSamplerCubeMap( ref_t<VkGraphicContext> aGraphicContext, ref_t<VkTexture2D> aTextureData,
                          sTextureSamplingInfo const &aSamplingSpec );

        /** @brief */
        ~VkSamplerCubeMap();

        ref_t<ITexture2D> GetTexture()
        {
            return mTextureData;
        }
        VkImageView GetImageView()
        {
            return mVkImageView;
        }
        VkSampler GetSampler()
        {
            return mVkImageSampler;
        }

      private:
        VkImageView mVkImageView    = VK_NULL_HANDLE;
        VkSampler   mVkImageSampler = VK_NULL_HANDLE;

        friend class VkTexture2D;
    };
} // namespace SE::Graphics
