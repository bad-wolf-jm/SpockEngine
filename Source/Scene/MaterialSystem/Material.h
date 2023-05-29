#pragma once
#include "Core/Math/Types.h"
#include "Graphics/API.h"

namespace SE::Graphics
{
    class Material
    {
      public:
        Material() = default;
        Material( Ref<IGraphicContext> aGraphicContext );

        ~Material() = default;

      private:
        Ref<IGraphicContext>   mGraphicContext = nullptr;
        Ref<IGraphicsPipeline> mPipeline       = nullptr;
    }
} // namespace SE::Graphics