#include "ITextureCubeMap.h"

#include "Core/Core.h"
#include "Core/Logging.h"
#include "Core/Memory.h"

namespace SE::Graphics
{
    /** @brief */
    ITextureCubeMap::ITextureCubeMap( Ref<IGraphicContext> aGraphicContext, TextureDataCubeMap &aCubeMapData, uint8_t aSampleCount,
                                      bool aIsHostVisible, bool aIsGraphicsOnly, bool aIsTransferSource )
        : ITexture( aGraphicContext, eTextureType::TEXTURE_CUBE_MAP, sTextureCreateInfo{}, aIsHostVisible, aSampleCount,
                    aIsGraphicsOnly, aIsTransferSource, false )
    {
        mSpec = mCreateInfo;
    }

    ITextureCubeMap::ITextureCubeMap( Ref<IGraphicContext> aGraphicContext, Core::sTextureCreateInfo &aTextureImageDescription,
                                      uint8_t aSampleCount, bool aIsHostVisible, bool aIsGraphicsOnly, bool aIsTransferSource,
                                      bool aIsTransferDestination )
        : ITexture( aGraphicContext, eTextureType::TEXTURE_CUBE_MAP, sTextureCreateInfo{}, aIsHostVisible, aSampleCount,
                    aIsGraphicsOnly, aIsTransferSource, false )
    {
        mSpec = mCreateInfo;
    }
} // namespace SE::Graphics
