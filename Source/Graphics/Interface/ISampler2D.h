#pragma once

#include <memory>

#include "Core/Memory.h"
#include "Core/Types.h"

#include "Core/CUDA/Texture/ColorFormat.h"
#include "Core/CUDA/Texture/Texture2D.h"
#include "Core/CUDA/Texture/TextureData.h"
#include "Core/CUDA/Texture/TextureTypes.h"

#include "IGraphicContext.h"
#include "ITexture2D.h"

namespace SE::Graphics
{
    using namespace SE::Core;

    /** @brief */
    class ISampler2D : public Cuda::TextureSampler2D
    {
      public:
        /** @brief */
        ISampler2D( ref_t<IGraphicContext> aGraphicContext, ref_t<ITexture2D> aTextureData,
                    sTextureSamplingInfo const &aSamplingSpec );

        /** @brief */
        ~ISampler2D() = default;

        ref_t<ITexture2D> GetTexture()
        {
            return mTextureData;
        }

        template <typename _GCSubtype>
        ref_t<_GCSubtype> GraphicContext()
        {
            return std::reinterpret_pointer_cast<_GCSubtype>( mGraphicContext );
        }

      protected:
        ref_t<IGraphicContext> mGraphicContext = nullptr;
        ref_t<ITexture2D>      mTextureData    = nullptr;

        friend class ITexture2D;
    };
} // namespace SE::Graphics
