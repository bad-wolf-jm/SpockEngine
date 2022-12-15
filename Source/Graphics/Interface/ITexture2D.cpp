#include "ITexture2D.h"

#include "Core/Core.h"
#include "Core/Logging.h"
#include "Core/Memory.h"

namespace SE::Graphics
{
    /** @brief */
    ITexture2D::ITexture2D( Ref<IGraphicContext> aGraphicContext, TextureData2D &mTextureData, uint8_t aSampleCount,
                            bool aIsHostVisible, bool aIsGraphicsOnly, bool aIsTransferSource )
        : mGraphicContext( aGraphicContext )
        , mSampleCount{ aSampleCount }
        , mIsHostVisible{ aIsHostVisible }
        , mIsGraphicsOnly{ aIsGraphicsOnly }
        , mIsTransferSource{ aIsTransferSource }
        , mIsTransferDestination{ false }
    {
        mSpec = mTextureData.mSpec;
    }

    ITexture2D::ITexture2D( Ref<IGraphicContext> aGraphicContext, Core::sTextureCreateInfo &aTextureImageDescription,
                            uint8_t aSampleCount, bool aIsHostVisible, bool aIsGraphicsOnly, bool aIsTransferSource,
                            bool aIsTransferDestination )
        : mGraphicContext( aGraphicContext )
        , mSampleCount{ aSampleCount }
        , mIsHostVisible{ aIsHostVisible }
        , mIsGraphicsOnly{ aIsGraphicsOnly }
        , mIsTransferSource{ aIsTransferSource }
        , mIsTransferDestination{ aIsTransferDestination }
    {
        mSpec = aTextureImageDescription;
    }
} // namespace SE::Graphics
