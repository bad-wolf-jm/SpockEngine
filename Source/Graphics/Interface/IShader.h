#pragma once

#include "Core/Memory.h"
#include "Core/Types.h"

#include "IGraphicContext.h"

namespace SE::Graphics
{
    using namespace SE::Core;

    enum eShaderType : uint8_t
    {
        UNKNOWN,
        VERTEX,
        GEOMETRY,
        FRAGMENT,
        COMPUTE
    };

    /** @brief */
    class IShader
    {
      public:
        /** @brief */
        IShader( Ref<IGraphicContext> aGraphicContext, eShaderType aType );

        /** @brief */
        ~IShader() = default;

      protected:
        Ref<IGraphicContext> mGraphicContext = nullptr;
        eShaderType          mType           = eShaderType::UNKNOWN;
    };
} // namespace SE::Graphics
