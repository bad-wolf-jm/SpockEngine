#pragma once

#include <memory>

#include <gli/gli.hpp>
#include <vulkan/vulkan.h>

#include "Core/Memory.h"
#include "Core/Types.h"

#include "Graphics/Interface/IShaderProgram.h"

namespace SE::Graphics
{
    using namespace SE::Core;

    /** @brief */
    class VkShaderProgram : public IShaderProgram
    {
      public:
        /** @brief */
        VkShaderProgram( Ref<IGraphicContext> aGraphicContext, eShaderStageTypeFlags aShaderType, int aVersion );

        /** @brief */
        ~VkShaderProgram() = default;

        void Compile();

      private:
    };
} // namespace SE::Graphics
