#pragma once

#include "Core/Memory.h"
#include "IGraphicBuffer.h"
#include "IGraphicContext.h"
#include "ISampler2D.h"
#include "ISamplerCubeMap.h"

#include <vector>
#include <vulkan/vulkan.h>

#include "IDescriptorSet.h"

namespace SE::Graphics
{
    enum class eDescriptorType : uint32_t
    {
        SAMPLER                = 0,
        COMBINED_IMAGE_SAMPLER = 1,
        SAMPLED_IMAGE          = 2,
        STORAGE_IMAGE          = 3,
        UNIFORM_TEXEL_BUFFER   = 4,
        STORAGE_TEXEL_BUFFER   = 5,
        UNIFORM_BUFFER         = 6,
        STORAGE_BUFFER         = 7,
        UNIFORM_BUFFER_DYNAMIC = 8,
        STORAGE_BUFFER_DYNAMIC = 9,
        INPUT_ATTACHMENT       = 10
    };

    enum class eShaderStageTypeFlags : uint32_t
    {
        VERTEX                 = 1,
        GEOMETRY               = 2,
        FRAGMENT               = 4,
        TESSELATION_CONTROL    = 8,
        TESSELATION_EVALUATION = 16,
        COMPUTE                = 32,
        DEFAULT                = 0xFFFFFFFF
    };

    using ShaderStageType = EnumSet<eShaderStageTypeFlags, 0x000001ff>;

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

        virtual Ref<IDescriptorSet> Allocate( uint32_t aDescriptorCount ) = 0;

      protected:
        Ref<IGraphicContext> mGraphicContext{};

        bool     mIsUnbounded     = false;
        uint32_t mDescriptorCount = 1;

        std::vector<sDescriptorBindingInfo> mDescriptors = {};
    };

} // namespace SE::Graphics