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

#include "Buffer.h"
#include "GraphicContext.h"

namespace SE::Graphics
{

    using namespace SE::Core;


    /** @brief */
    class Texture2D : public Cuda::Texture2D
    {
      public:
        Core::TextureData::sCreateInfo mSpec;

        /** @brief */
        Texture2D( GraphicContext &aGraphicContext, Core::TextureData::sCreateInfo &aTextureImageDescription, bool aIsHostVisible );

        /** @brief */
        Texture2D( GraphicContext &aGraphicContext, TextureData2D &aCubeMapData );

        /** @brief */
        ~Texture2D() = default;

      private:
        void CreateImageView();
        void CopyBufferToImage( Buffer &a_Buffer );
        void TransitionImageLayout( VkImageLayout oldLayout, VkImageLayout newLayout );

      private:
        GraphicContext mGraphicContext{};

        Ref<Internal::sVkImageObject>     mTextureImageObject = nullptr;
        Ref<Internal::sVkImageViewObject> mTextureView        = nullptr;
    };
} // namespace SE::Graphics
