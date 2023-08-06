#pragma once

#include "Core/Memory.h"
#include "Core/Types.h"

#include "Core/CUDA/Texture/ColorFormat.h"
#include "Core/CUDA/Texture/Texture2D.h"
#include "Core/CUDA/Texture/TextureData.h"
#include "Core/CUDA/Texture/TextureTypes.h"

#include "IGraphicBuffer.h"
#include "IGraphicContext.h"
#include "IGraphicResource.h"
#include "ITexture.h"

namespace SE::Graphics
{
    using namespace SE::Core;

    /** @brief */
    class ITexture2D : public ITexture, public Cuda::Texture2D
    {
        friend class ISampler2D;
        friend class IRenderTarget;

      public:
        /** @brief */
        ITexture2D( ref_t<IGraphicContext> aGraphicContext, sTextureCreateInfo &aTextureImageDescription, uint8_t aSampleCount,
                    bool aIsHostVisible, bool aIsGraphicsOnly, bool aIsTransferSource, bool aIsTransferDestination );

        /** @brief */
        ITexture2D( ref_t<IGraphicContext> aGraphicContext, TextureData2D &aTextureData )
            : ITexture2D( aGraphicContext, aTextureData, 1, false, true, true )
        {
        }

        /** @brief */
        ITexture2D( ref_t<IGraphicContext> aGraphicContext, TextureData2D &aTextureData, uint8_t aSampleCount, bool aIsHostVisible,
                    bool aIsGraphicsOnly, bool aIsTransferSource );

        /** @brief */
        ~ITexture2D() = default;

        virtual void GetPixelData( TextureData2D &mTextureData )    = 0;
        virtual void SetPixelData( ref_t<IGraphicBuffer> a_Buffer ) = 0;
    };
} // namespace SE::Graphics
