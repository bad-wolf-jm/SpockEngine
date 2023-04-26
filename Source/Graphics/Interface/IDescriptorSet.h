#pragma once

#include "Core/Memory.h"
#include "IGraphicContext.h"
#include "ISampler2D.h"
#include "ISamplerCubeMap.h"
#include "IGraphicBuffer.h"

#include <vector>
#include <vulkan/vulkan.h>

namespace SE::Graphics
{
    class IDescriptorSet
    {
      public:
        IDescriptorSet( Ref<IGraphicContext> aGraphicContext, uint32_t aDescriptorCount = 0 );
        ~IDescriptorSet() = default;

        void Write( Ref<ISampler2D> aBuffer, uint32_t aBinding );
        void Write( Ref<ISamplerCubeMap> aBuffer, uint32_t aBinding );

        virtual void Write( Ref<IGraphicBuffer> aBuffer, bool aDynamicOffset, uint32_t aOffset, uint32_t aSize,
                            uint32_t aBinding )                                            = 0;
        virtual void Write( std::vector<Ref<ISampler2D>> aBuffer, uint32_t aBinding )      = 0;
        virtual void Write( std::vector<Ref<ISamplerCubeMap>> aBuffer, uint32_t aBinding ) = 0;

      private:
        Ref<IGraphicContext> mGraphicContext{};
    };

} // namespace SE::Graphics