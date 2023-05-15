#pragma once

#include "Core/Memory.h"
#include "IGraphicBuffer.h"
#include "IGraphicContext.h"
#include "ISampler2D.h"
#include "ISamplerCubeMap.h"

#include <vector>
#include <vulkan/vulkan.h>

#include "Graphics/Vulkan/VkPipeline.h"
#include "IDescriptorSet.h"

namespace SE::Graphics
{
    struct sDescriptorBindingInfo
    {
        uint32_t        mBindingIndex = 0;
        eDescriptorType mType         = eDescriptorType::UNIFORM_BUFFER;
        ShaderStageType mShaderStages = {};
    };

    class IDescriptorSetLayout
    {
      public:
        IDescriptorSetLayout( Ref<IGraphicContext> aGraphicContext, bool aIsUnbounded = false, uint32_t aDescriptorCount = 0 );
        ~IDescriptorSetLayout() = default;

        void         AddBinding( uint32_t aBindingIndex, eDescriptorType aType, ShaderStageType aShaderStages );
        virtual void Build() = 0;

        virtual Ref<IDescriptorSet> Allocate(uint32_t aDescriptorCount) = 0;

      protected:
        Ref<IGraphicContext> mGraphicContext{};

        bool     mIsUnbounded     = false;
        uint32_t mDescriptorCount = 1;

        std::vector<sDescriptorBindingInfo> mDescriptors = {};
    };

} // namespace SE::Graphics