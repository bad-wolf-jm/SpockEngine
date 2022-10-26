#pragma once

#include <memory>

#include <vulkan/vulkan.h>

#include "Core/Textures/ColorFormat.h"
#include "Core/Memory.h"
#include "Core/Textures/TextureData.h"

#include <gli/gli.hpp>

#include "Buffer.h"
#include "GraphicContext.h"
#include "Texture2D.h"

namespace LTSE::Graphics
{

    // front, back, up, down, right, left

    enum class CubeMapDirection : size_t
    {
        FRONT  = 0,
        BACK   = 1,
        TOP    = 2,
        BOTTOM = 3,
        RIGHT  = 4,
        LEFT   = 5,
    };

    struct CubeMapFace
    {
        CubeMapDirection Direction;
        TextureData FaceData;
    };

    /** @brief */
    class TextureCubeMap
    {
      public:
        TextureDescription Spec; /**!< */

        /** @brief */
        TextureCubeMap( GraphicContext &a_GraphicContext, TextureDescription &a_BufferDescription );
        TextureCubeMap( GraphicContext &a_GraphicContext, TextureDescription &a_BufferDescription, gli::texture_cube &a_CubeMapData );

        /** @brief */
        ~TextureCubeMap() = default;

        TextureData GetImageData();

        /** @brief */
        inline VkImageView GetImageView() { return m_TextureView->mVkObject; }

        /** @brief */
        inline VkImage GetImage() { return m_TextureImageObject->mVkObject; }

        /** @brief */
        inline VkSampler GetSampler() { return m_TextureSamplerObject->mVkObject; }

        void TransitionImageLayout( VkImageLayout oldLayout, VkImageLayout newLayout );

      private:
        void CopyBufferToImage( Buffer &a_Buffer, gli::texture_cube &a_CubeMapData );
        void CreateImageView();
        void CreateImageSampler();

      private:
        GraphicContext mGraphicContext{};

        Ref<Internal::sVkImageObject> m_TextureImageObject          = nullptr;
        Ref<Internal::sVkImageSamplerObject> m_TextureSamplerObject = nullptr;
        Ref<Internal::sVkImageViewObject> m_TextureView             = nullptr;
    };
} // namespace LTSE::Graphics
