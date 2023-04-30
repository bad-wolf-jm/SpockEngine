#pragma once

#include "Core/Memory.h"
#include "Core/Types.h"

#include "Core/CUDA/Texture/ColorFormat.h"
#include "Core/CUDA/Texture/Texture2D.h"
#include "Core/CUDA/Texture/TextureData.h"
#include "Core/CUDA/Texture/TextureTypes.h"

#include "IGraphicContext.h"
#include "IGraphicResource.h"
#include "IGraphicBuffer.h"

namespace SE::Graphics
{
    using namespace SE::Core;

    /** @brief */
    class ICommandBuffer
    {
      public:
        /** @brief */
        ICommandBuffer(  ) = default;
        ICommandBuffer( Ref<IGraphicContext> aGraphicContext );

        /** @brief */
        ~ICommandBuffer() = default;

      protected:
        Ref<IGraphicContext> mGraphicContext = nullptr;
    };
} // namespace SE::Graphics
