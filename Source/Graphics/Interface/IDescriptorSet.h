#pragma once

#include "Core/Memory.h"
#include "IGraphicBuffer.h"
#include "IGraphicContext.h"
#include "ISampler2D.h"
#include "ISamplerCubeMap.h"

#include "Core/Vector.h"
#include <vulkan/vulkan.h>

namespace SE::Graphics
{
    class IDescriptorSet
    {
      public:
        IDescriptorSet( Ref<IGraphicContext> aGraphicContext, bool aIsUnbounded = false, uint32_t aDescriptorCount = 0 );
        ~IDescriptorSet() = default;

        virtual void *GetID()
        {
            return 0;
        };

        void         Write( Ref<ISampler2D> aBuffer, uint32_t aBinding );
        void         Write( Ref<ISamplerCubeMap> aBuffer, uint32_t aBinding );
        virtual void Write( Ref<IGraphicBuffer> aBuffer, bool aDynamicOffset, uint32_t aOffset, uint32_t aSize,
                            uint32_t aBinding )                                         = 0;
        virtual void Write( vector_t<Ref<ISampler2D>> aBuffer, uint32_t aBinding )      = 0;
        virtual void Write( vector_t<Ref<ISamplerCubeMap>> aBuffer, uint32_t aBinding ) = 0;

      protected:
        Ref<IGraphicContext> mGraphicContext{};
    };

} // namespace SE::Graphics