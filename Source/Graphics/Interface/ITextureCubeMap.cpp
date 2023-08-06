#include "ITextureCubeMap.h"

#include "Core/Core.h"
#include "Core/Logging.h"
#include "Core/Memory.h"

namespace SE::Graphics
{
    /** @brief */
    ITextureCubeMap::ITextureCubeMap( ref_t<IGraphicContext> aGraphicContext, TextureDataCubeMap &aCubeMapData, uint8_t aSampleCount,
                                      bool aIsHostVisible, bool aIsGraphicsOnly, bool aIsTransferSource )
        : ITexture( aGraphicContext, eTextureType::TEXTURE_CUBE_MAP, sTextureCreateInfo{}, aSampleCount, aIsHostVisible,
                    aIsGraphicsOnly, aIsTransferSource, false )
    {
        mSpec = mCreateInfo;
    }

    ITextureCubeMap::ITextureCubeMap( ref_t<IGraphicContext> aGraphicContext, Core::sTextureCreateInfo &aTextureImageDescription,
                                      uint8_t aSampleCount, bool aIsHostVisible, bool aIsGraphicsOnly, bool aIsTransferSource,
                                      bool aIsTransferDestination )
        : ITexture( aGraphicContext, eTextureType::TEXTURE_CUBE_MAP, aTextureImageDescription, aSampleCount, aIsHostVisible,
                    aIsGraphicsOnly, aIsTransferSource, aIsTransferDestination )
    {
        mSpec = mCreateInfo;
    }
} // namespace SE::Graphics
