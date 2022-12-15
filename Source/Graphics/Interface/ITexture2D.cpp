#include "ITexture2D.h"

#include "Core/Core.h"
#include "Core/Logging.h"
#include "Core/Memory.h"

#include "VkCommand.h"
#include "VkGpuBuffer.h"

namespace SE::Graphics
{
    /** @brief */
    ITexture2D::ITexture2D( Ref<IGraphicContext> aGraphicContext, TextureData2D &mTextureData, uint8_t aSampleCount,
                            bool aIsHostVisible, bool aIsGraphicsOnly, bool aIsTransferSource )
        : mGraphicContext( aGraphicContext )
        , mSpec{ mTextureData.mSpec }
        , mSampleCount{ aSampleCount }
        , mIsHostVisible{ aIsHostVisible }
        , mIsGraphicsOnly{ aIsGraphicsOnly }
        , mIsTransferSource{ aIsTransferSource }
        , mIsTransferDestination{ false }
    {
    }

    ITexture2D::ITexture2D( Ref<IGraphicContext> aGraphicContext, Core::sTextureCreateInfo &aTextureImageDescription,
                            uint8_t aSampleCount, bool aIsHostVisible, bool aIsGraphicsOnly, bool aIsTransferSource,
                            bool aIsTransferDestination )
        : mGraphicContext( aGraphicContext )
        , mSpec( aTextureImageDescription )
        , mSampleCount{ VK_SAMPLE_COUNT_VALUE( aSampleCount ) }
        , mIsHostVisible{ aIsHostVisible }
        , mIsGraphicsOnly{ aIsGraphicsOnly }
        , mIsTransferSource{ aIsTransferSource }
        , mIsTransferDestination{ aIsTransferDestination }
    {
    }
} // namespace SE::Graphics
