#include "IDescriptorSetLayout.h"

#include "Core/Memory.h"
#include <stdexcept>

namespace SE::Graphics
{

    IDescriptorSetLayout::IDescriptorSetLayout( Ref<IGraphicContext> aGraphicContext, bool aIsUnbounded, uint32_t aDescriptorCount )
        : mGraphicContext{ aGraphicContext }
        , mIsUnbounded{ aIsUnbounded }
        , mDescriptorCount{ aDescriptorCount }

    {
    }

    void IDescriptorSetLayout::AddBinding( uint32_t aBindingIndex, eDescriptorType aType, ShaderStageType aShaderStages )
    {
        auto &lNewDescriptor = mDescriptors.emplace_back();

        lNewDescriptor.mBindingIndex = aBindingIndex;
        lNewDescriptor.mType         = aType;
        lNewDescriptor.mShaderStages = aShaderStages;
    }
} // namespace SE::Graphics