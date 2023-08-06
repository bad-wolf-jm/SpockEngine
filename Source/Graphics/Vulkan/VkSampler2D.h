#pragma once

#include <memory>

#include <gli/gli.hpp>
#include <vulkan/vulkan.h>

#include "Core/Memory.h"
#include "Core/Types.h"

#include "Graphics/Interface/ISampler2D.h"

#include "VkGraphicContext.h"
#include "VkTexture2D.h"

namespace SE::Graphics
{
    using namespace SE::Core;

    /** @brief */
    class VkSampler2D : public ISampler2D
    {
      public:
        /** @brief */
        VkSampler2D( ref_t<IGraphicContext> aGraphicContext, ref_t<VkTexture2D> aTextureData,
                     sTextureSamplingInfo const &aSamplingSpec );

        /** @brief */
        ~VkSampler2D();

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
