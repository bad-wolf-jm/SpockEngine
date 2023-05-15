#include "IDescriptorSet.h"

#include "Core/Memory.h"
#include <stdexcept>

namespace SE::Graphics
{

    IDescriptorSet::IDescriptorSet( Ref<IGraphicContext> aGraphicContext, bool aIsUnbounded, uint32_t aDescriptorCount )
        : mGraphicContext{ aGraphicContext }
        , mIsUnbounded{ aIsUnbounded }
        , mDescriptorCount{ aDescriptorCount }

    {
    }

    void IDescriptorSet::Write( Ref<ISampler2D> aBuffer, uint32_t aBinding ) { Write( std::vector{ aBuffer }, aBinding ); }
    void IDescriptorSet::Write( Ref<ISamplerCubeMap> aBuffer, uint32_t aBinding ) { Write( std::vector{ aBuffer }, aBinding ); }

    void IDescriptorSet::AddBinding( uint32_t aBindingIndex, eDescriptorType aType, ShaderStageType aShaderStages )
    {
        auto lNewDescriptor = mDescriptors.emplace_back();

        lNewDescriptor.mBindingIndex = aBindingIndex;
        lNewDescriptor.mType         = aType;
        lNewDescriptor.mShaderStages = aShaderStages;
    }

} // namespace SE::Graphics