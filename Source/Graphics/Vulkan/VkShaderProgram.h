#pragma once

#include <memory>

#include <gli/gli.hpp>
#include <vulkan/vulkan.h>

#include "Core/Memory.h"
#include "Core/Types.h"

#include "VkPipeline.h"

#include "Graphics/Interface/IShaderProgram.h"

namespace SE::Graphics
{
    using namespace SE::Core;

    /** @brief */
    class VkShaderProgram : public IShaderProgram
    {
      public:
        /** @brief */
        VkShaderProgram( ref_t<IGraphicContext> aGraphicContext, eShaderStageTypeFlags aShaderType, int aVersion,
                         string_t const &aName, fs::path const &aCacheRoot );

        /** @brief */
        ~VkShaderProgram() = default;

        void DoCompile();
        void BuildProgram();

        ref_t<ShaderModule> GetShaderModule() { return mShaderModule; };

      private:
        ref_t<ShaderModule> mShaderModule{};
    };
} // namespace SE::Graphics
