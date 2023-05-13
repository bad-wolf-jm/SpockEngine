#include "ITexture2D.h"

#include "Core/Core.h"
#include "Core/Logging.h"
#include "Core/Memory.h"

namespace SE::Graphics
{
    /** @brief */
    ITexture2D::ITexture2D( Ref<IGraphicContext> aGraphicContext, TextureData2D &mTextureData, uint8_t aSampleCount,
                            bool aIsHostVisible, bool aIsGraphicsOnly, bool aIsTransferSource )
        : ITexture( aGraphicContext, eTextureType::TEXTURE_2D, sTextureCreateInfo{}, aIsHostVisible, aSampleCount, aIsGraphicsOnly,
                    aIsTransferSource, false )
    {
        mSpec = mCreateInfo;
    }

    ITexture2D::ITexture2D( Ref<IGraphicContext> aGraphicContext, Core::sTextureCreateInfo &aTextureImageDescription,
                            uint8_t aSampleCount, bool aIsHostVisible, bool aIsGraphicsOnly, bool aIsTransferSource,
                            bool aIsTransferDestination )
        : ITexture( aGraphicContext, eTextureType::TEXTURE_2D, aTextureImageDescription, aSampleCount, aIsHostVisible, aIsGraphicsOnly,
                    aIsTransferSource, aIsTransferDestination )
    {
        mSpec = mCreateInfo;
    }
} // namespace SE::Graphics
