#include "ITexture2D.h"

#include "Core/Core.h"
#include "Core/Logging.h"
#include "Core/Memory.h"

namespace SE::Graphics
{
    /** @brief */
    ITexture2D::ITexture2D( Ref<IGraphicContext> aGraphicContext, TextureData2D &mTextureData, uint8_t aSampleCount,
                            bool aIsHostVisible, bool aIsGraphicsOnly, bool aIsTransferSource )
        : IGraphicResource( aGraphicContext, aIsHostVisible, aIsGraphicsOnly, aIsTransferSource, false, 0 )
        , mSampleCount{ aSampleCount }
    {
        mSpec = mTextureData.mSpec;
    }

    ITexture2D::ITexture2D( Ref<IGraphicContext> aGraphicContext, Core::sTextureCreateInfo &aTextureImageDescription,
                            uint8_t aSampleCount, bool aIsHostVisible, bool aIsGraphicsOnly, bool aIsTransferSource,
                            bool aIsTransferDestination )
        : IGraphicResource( aGraphicContext, aIsHostVisible, aIsGraphicsOnly, aIsTransferSource, aIsTransferDestination, 0 )
        , mSampleCount{ aSampleCount }
    {
        mSpec = aTextureImageDescription;
    }
} // namespace SE::Graphics
