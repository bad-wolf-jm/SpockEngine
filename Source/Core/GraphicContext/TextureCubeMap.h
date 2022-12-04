#pragma once

#include <memory>

#include <vulkan/vulkan.h>

#include "Core/Memory.h"
#include "Core/CUDA/Texture/ColorFormat.h"
#include "Core/CUDA/Texture/TextureData.h"

#include <gli/gli.hpp>

#include "Buffer.h"
#include "GraphicContext.h"
#include "Texture2D.h"

namespace SE::Graphics
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
        TextureData      FaceData;
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
        inline VkImageView GetImageView() { return mTextureView->mVkObject; }

        /** @brief */
        inline VkImage GetImage() { return mTextureImageObject->mVkObject; }

        /** @brief */
        inline VkSampler GetSampler() { return mTextureSamplerObject->mVkObject; }

        void *GetMemoryHandle() { return mGraphicContext.mContext->GetSharedMemoryHandle( mTextureImageObject->mVkMemory ); }

        void TransitionImageLayout( VkImageLayout oldLayout, VkImageLayout newLayout );

      private:
        void CopyBufferToImage( Buffer &a_Buffer, gli::texture_cube &a_CubeMapData );
        void CreateImageView();
        void CreateImageSampler();

      private:
        GraphicContext mGraphicContext{};

        Ref<Internal::sVkImageObject>        mTextureImageObject   = nullptr;
        Ref<Internal::sVkImageSamplerObject> mTextureSamplerObject = nullptr;
        Ref<Internal::sVkImageViewObject>    mTextureView          = nullptr;
    };
} // namespace SE::Graphics
