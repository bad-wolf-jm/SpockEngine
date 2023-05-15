#pragma once

#include "Core/Memory.h"
#include "IGraphicBuffer.h"
#include "IGraphicContext.h"
#include "ISampler2D.h"
#include "ISamplerCubeMap.h"

#include <vector>
#include <vulkan/vulkan.h>

#include "Graphics/Vulkan/VkPipeline.h"

namespace SE::Graphics
{
    struct sDescriptorBindingInfo
    {
        uint32_t        mBindingIndex = 0;
        eDescriptorType mType         = eDescriptorType::UNIFORM_BUFFER;
        ShaderStageType mShaderStages = {};
    };

    class IDescriptorSet
    {
      public:
        IDescriptorSet( Ref<IGraphicContext> aGraphicContext, bool aIsUnbounded = false, uint32_t aDescriptorCount = 0 );
        ~IDescriptorSet() = default;

        void         Write( Ref<ISampler2D> aBuffer, uint32_t aBinding );
        void         Write( Ref<ISamplerCubeMap> aBuffer, uint32_t aBinding );
        virtual void Write( Ref<IGraphicBuffer> aBuffer, bool aDynamicOffset, uint32_t aOffset, uint32_t aSize,
                            uint32_t aBinding )                                            = 0;
        virtual void Write( std::vector<Ref<ISampler2D>> aBuffer, uint32_t aBinding )      = 0;
        virtual void Write( std::vector<Ref<ISamplerCubeMap>> aBuffer, uint32_t aBinding ) = 0;

        void         AddBinding( uint32_t aBindingIndex, eDescriptorType aType, ShaderStageType aShaderStages );
        virtual void Build() = 0;

      private:
        Ref<IGraphicContext> mGraphicContext{};

        bool     mIsUnbounded     = false;
        uint32_t mDescriptorCount = 1;

        std::vector<sDescriptorBindingInfo> mDescriptors = {};
    };

} // namespace SE::Graphics