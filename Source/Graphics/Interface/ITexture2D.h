#pragma once

#include "Core/Memory.h"
#include "Core/Types.h"

#include "Core/CUDA/Texture/ColorFormat.h"
#include "Core/CUDA/Texture/Texture2D.h"
#include "Core/CUDA/Texture/TextureData.h"
#include "Core/CUDA/Texture/TextureTypes.h"

#include "IGraphicBuffer.h"
#include "IGraphicContext.h"

namespace SE::Graphics
{
    using namespace SE::Core;

    /** @brief */
    class ITexture2D : public Cuda::Texture2D
    {
        friend class ISampler2D;
        friend class IRenderTarget;

      public:
        /** @brief */
        ITexture2D( Ref<IGraphicContext> aGraphicContext, sTextureCreateInfo &aTextureImageDescription, uint8_t aSampleCount,
                    bool aIsHostVisible, bool aIsGraphicsOnly, bool aIsTransferSource, bool aIsTransferDestination );

        /** @brief */
        ITexture2D( Ref<IGraphicContext> aGraphicContext, TextureData2D &aCubeMapData )
            : ITexture2D( aGraphicContext, aCubeMapData, 1, false, true, true )
        {
        }

        /** @brief */
        ITexture2D( Ref<IGraphicContext> aGraphicContext, TextureData2D &aCubeMapData, uint8_t aSampleCount, bool aIsHostVisible,
                    bool aIsGraphicsOnly, bool aIsTransferSource );

        /** @brief */
        ~ITexture2D() = default;

        virtual void GetPixelData( TextureData2D &mTextureData )  = 0;
        virtual void SetPixelData( Ref<IGraphicBuffer> a_Buffer ) = 0;

      protected:
        Ref<IGraphicContext> mGraphicContext{};

        uint8_t mSampleCount           = 1;
        bool    mIsHostVisible         = false;
        bool    mIsGraphicsOnly        = false;
        bool    mIsTransferSource      = false;
        bool    mIsTransferDestination = false;
    };
} // namespace SE::Graphics
