#pragma once

#include "Core/Memory.h"
#include "IDescriptorSet.h"
#include "IGraphicBuffer.h"
#include "IGraphicContext.h"
#include "ISampler2D.h"
#include "ISamplerCubeMap.h"

#include "Enums.h"

#include <vector>
#include <vulkan/vulkan.h>

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
        IDescriptorSetLayout( ref_t<IGraphicContext> aGraphicContext, bool aIsUnbounded = false, uint32_t aDescriptorCount = 0 );
        ~IDescriptorSetLayout() = default;

        void         AddBinding( uint32_t aBindingIndex, eDescriptorType aType, ShaderStageType aShaderStages );
        virtual void Build() = 0;

        virtual ref_t<IDescriptorSet> Allocate( uint32_t aDescriptorCount ) = 0;

      protected:
        ref_t<IGraphicContext> mGraphicContext{};

        bool     mIsUnbounded     = false;
        uint32_t mDescriptorCount = 1;

        vector_t<sDescriptorBindingInfo> mDescriptors = {};
    };

} // namespace SE::Graphics