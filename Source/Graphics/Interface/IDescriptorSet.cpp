#include "IDescriptorSet.h"

#include "Core/Memory.h"
#include <stdexcept>

namespace SE::Graphics
{

    IDescriptorSet::IDescriptorSet( ref_t<IGraphicContext> aGraphicContext, bool aIsUnbounded, uint32_t aDescriptorCount )
        : mGraphicContext{ aGraphicContext }
        // , mIsUnbounded{ aIsUnbounded }
        // , mDescriptorCount{ aDescriptorCount }

    {
    }

    void IDescriptorSet::Write( ref_t<ISampler2D> aBuffer, uint32_t aBinding ) { Write( vec_t{ aBuffer }, aBinding ); }
    void IDescriptorSet::Write( ref_t<ISamplerCubeMap> aBuffer, uint32_t aBinding ) { Write( vec_t{ aBuffer }, aBinding ); }
} // namespace SE::Graphics