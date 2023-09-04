#pragma once

#include "Core/Memory.h"
#include "IGraphicBuffer.h"
#include "IGraphicContext.h"
#include "ISampler2D.h"
#include "ISamplerCubeMap.h"

#include <vector>
#include <vulkan/vulkan.h>

namespace SE::Graphics
{
    class IDescriptorSet
    {
      public:
        IDescriptorSet( ref_t<IGraphicContext> aGraphicContext, bool aIsUnbounded = false, uint32_t aDescriptorCount = 0 );
        ~IDescriptorSet() = default;

        virtual void *GetID()
        {
            return 0;
        };

        void         Write( ref_t<ISampler2D> aBuffer, uint32_t aBinding );
        void         Write( ref_t<ISamplerCubeMap> aBuffer, uint32_t aBinding );
        virtual void Write( ref_t<IGraphicBuffer> aBuffer, bool aDynamicOffset, uint32_t aOffset, uint32_t aSize,
                            uint32_t aBinding )                                            = 0;
        virtual void Write( vector_t<ref_t<ISampler2D>> aBuffer, uint32_t aBinding )      = 0;
        virtual void Write( vector_t<ref_t<ISamplerCubeMap>> aBuffer, uint32_t aBinding ) = 0;

      protected:
        ref_t<IGraphicContext> mGraphicContext{};
    };

} // namespace SE::Graphics