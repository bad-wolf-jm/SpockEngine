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
    class Sampler2D : public Cuda::TextureSampler2D
    {
      public:
        sTextureSamplingInfo mSamplingSpec; 

        /** @brief */
        Sampler2D( GraphicContext &aGraphicContext, Texture2D const &aBufferDescription, sTextureSamplingInfo const &mSamplingSpec );

        /** @brief */
        ~Sampler2D() = default;

      private:
        GraphicContext mGraphicContext{};
        Ref<Internal::sVkImageSamplerObject> mTextureSamplerObject = nullptr;
    };
} // namespace SE::Graphics
