#pragma once

#include "Core/Memory.h"
#include "Core/Types.h"

#include "Core/CUDA/Texture/ColorFormat.h"
#include "Core/CUDA/Texture/TextureCubeMap.h"
#include "Core/CUDA/Texture/TextureData.h"
#include "Core/CUDA/Texture/TextureTypes.h"

#include "IGraphicBuffer.h"
#include "IGraphicContext.h"
#include "IGraphicResource.h"
#include "ITexture.h"

namespace SE::Graphics
{
    using namespace SE::Core;

    using TextureDataCubeMap = TextureData2D;

    /** @brief */
    class ITextureCubeMap : public ITexture, public Cuda::TextureCubeMap
    {
        friend class ISamplerCubeMap;

      public:
        /** @brief */
        ITextureCubeMap( ref_t<IGraphicContext> aGraphicContext, sTextureCreateInfo &aTextureImageDescription, uint8_t aSampleCount,
                         bool aIsHostVisible, bool aIsGraphicsOnly, bool aIsTransferSource, bool aIsTransferDestination );

        /** @brief */
        ITextureCubeMap( ref_t<IGraphicContext> aGraphicContext, TextureDataCubeMap &aCubeMapData )
            : ITextureCubeMap( aGraphicContext, aCubeMapData, 1, false, true, true )
        {
        }

        /** @brief */
        ITextureCubeMap( ref_t<IGraphicContext> aGraphicContext, TextureDataCubeMap &aCubeMapData, uint8_t aSampleCount,
                         bool aIsHostVisible, bool aIsGraphicsOnly, bool aIsTransferSource );

        /** @brief */
        ~ITextureCubeMap() = default;

        virtual void GetPixelData( TextureDataCubeMap &aTextureData )             = 0;
        virtual void GetPixelData( TextureData2D &aTextureData, eCubeFace aFace ) = 0;

        virtual void SetPixelData( ref_t<IGraphicBuffer> aBuffer )                  = 0;
        virtual void SetPixelData( eCubeFace aFace, ref_t<IGraphicBuffer> aBuffer ) = 0;
    };
} // namespace SE::Graphics
