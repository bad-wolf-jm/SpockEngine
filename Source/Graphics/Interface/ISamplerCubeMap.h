#pragma once

#include <memory>

#include "Core/Memory.h"
#include "Core/Types.h"

#include "Core/CUDA/Texture/ColorFormat.h"
#include "Core/CUDA/Texture/TextureCubeMap.h"
#include "Core/CUDA/Texture/TextureData.h"
#include "Core/CUDA/Texture/TextureTypes.h"

#include "IGraphicContext.h"
#include "ITextureCubeMap.h"

namespace SE::Graphics
{
    using namespace SE::Core;

    /** @brief */
    class ISamplerCubeMap : public Cuda::TextureSamplerCubeMap
    {
      public:
        /** @brief */
        ISamplerCubeMap( ref_t<IGraphicContext> aGraphicContext, ref_t<ITextureCubeMap> aTextureData,
                         sTextureSamplingInfo const &aSamplingSpec );

        /** @brief */
        ~ISamplerCubeMap() = default;

        ref_t<ITextureCubeMap> GetTexture()
        {
            return mTextureData;
        }

      protected:
        ref_t<IGraphicContext> mGraphicContext = nullptr;
        ref_t<ITextureCubeMap> mTextureData    = nullptr;

        friend class ITextureCubeMap;
    };
} // namespace SE::Graphics
