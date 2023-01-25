#include "ITexture.h"

#include "Core/Core.h"
#include "Core/Logging.h"
#include "Core/Memory.h"

namespace SE::Graphics
{
    ITexture::ITexture( Ref<IGraphicContext> aGraphicContext, eTextureType aType, sTextureCreateInfo &aTextureImageDescription,
                        uint8_t aSampleCount, bool aIsHostVisible, bool aIsGraphicsOnly, bool aIsTransferSource,
                        bool aIsTransferDestination )
        : IGraphicResource( aGraphicContext, aIsHostVisible, aIsGraphicsOnly, aIsTransferSource, aIsTransferDestination, 0 )
        , mSampleCount{ aSampleCount }
        , mType{ aType }
        , mCreateInfo{ aTextureImageDescription }
    {
    }
} // namespace SE::Graphics
