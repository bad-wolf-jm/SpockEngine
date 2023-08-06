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

namespace SE::Graphics
{
    using namespace SE::Core;

    /** @brief */
    class ITexture : public IGraphicResource
    {
      public:
        sTextureCreateInfo mCreateInfo{};

      public:
        /** @brief */
        ITexture( ref_t<IGraphicContext> aGraphicContext, eTextureType aType, sTextureCreateInfo &aTextureImageDescription,
                  uint8_t aSampleCount, bool aIsHostVisible, bool aIsGraphicsOnly, bool aIsTransferSource,
                  bool aIsTransferDestination );

        /** @brief */
        ~ITexture() = default;

        eTextureType GetTextureType()
        {
            return mType;
        }

      protected:
        uint8_t      mSampleCount = 1;
        eTextureType mType        = eTextureType::UNKNOWN;
    };
} // namespace SE::Graphics
