#include "VkDescriptorSetLayout.h"
#include "VkDescriptorSet.h"

#include "Core/Memory.h"
#include <stdexcept>

namespace SE::Graphics
{
    VkDescriptorSetLayoutObject::VkDescriptorSetLayoutObject( Ref<IGraphicContext> aGraphicContext, bool aIsUnbounded,
                                                              uint32_t aDescriptorCount )
        : IDescriptorSetLayout{ aGraphicContext, aIsUnbounded, aDescriptorCount }
    {
    }

    void VkDescriptorSetLayoutObject::Build()
    {
        auto lDescriptorSet = mDescriptors;

        std::vector<VkDescriptorSetLayoutBinding> lBindings{};

        for( uint32_t j = 0; j < lDescriptorSet.size(); j++ )
        {
            VkDescriptorSetLayoutBinding &lNewBinding = lBindings.emplace_back();

            lNewBinding.binding         = lDescriptorSet[j].mBindingIndex;
            lNewBinding.descriptorCount = 1;

            switch( lDescriptorSet[j].mType )
            {
            case eDescriptorType::SAMPLER:
                lNewBinding.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
                break;
            case eDescriptorType::COMBINED_IMAGE_SAMPLER:
                lNewBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                break;
            case eDescriptorType::SAMPLED_IMAGE:
                lNewBinding.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
                break;
            case eDescriptorType::STORAGE_IMAGE:
                lNewBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                break;
            case eDescriptorType::UNIFORM_TEXEL_BUFFER:
                lNewBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
                break;
            case eDescriptorType::STORAGE_TEXEL_BUFFER:
                lNewBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
                break;
            case eDescriptorType::UNIFORM_BUFFER:
                lNewBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                break;
            case eDescriptorType::STORAGE_BUFFER:
                lNewBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                break;
            case eDescriptorType::UNIFORM_BUFFER_DYNAMIC:
                lNewBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
                break;
            case eDescriptorType::STORAGE_BUFFER_DYNAMIC:
                lNewBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
                break;
            case eDescriptorType::INPUT_ATTACHMENT:
                lNewBinding.descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
            default:
                break;
            }

            // lNewBinding.descriptorType     = (VkDescriptorType)lDescriptorSet[j].mType;
            lNewBinding.pImmutableSamplers = nullptr;

            if( lDescriptorSet[j].mShaderStages & eShaderStageTypeFlags::VERTEX )
                lNewBinding.stageFlags |= VK_SHADER_STAGE_VERTEX_BIT;

            if( lDescriptorSet[j].mShaderStages & eShaderStageTypeFlags::COMPUTE )
                lNewBinding.stageFlags |= VK_SHADER_STAGE_COMPUTE_BIT;

            if( lDescriptorSet[j].mShaderStages & eShaderStageTypeFlags::GEOMETRY )
                lNewBinding.stageFlags |= VK_SHADER_STAGE_GEOMETRY_BIT;

            if( lDescriptorSet[j].mShaderStages & eShaderStageTypeFlags::FRAGMENT )
                lNewBinding.stageFlags |= VK_SHADER_STAGE_FRAGMENT_BIT;

            if( lDescriptorSet[j].mShaderStages & eShaderStageTypeFlags::TESSELATION_CONTROL )
                lNewBinding.stageFlags |= VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;

            if( lDescriptorSet[j].mShaderStages & eShaderStageTypeFlags::TESSELATION_EVALUATION )
                lNewBinding.stageFlags |= VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
        }

        if( mIsUnbounded )
            lBindings[lBindings.size() - 1].descriptorCount = 1024;

        mLayout = New<sVkDescriptorSetLayoutObject>( Cast<VkGraphicContext>( mGraphicContext ), lBindings, mIsUnbounded );
    }

    Ref<IDescriptorSet> VkDescriptorSetLayoutObject::Allocate( uint32_t aDescriptorCount )
    {
        return SE::Core::New<VkDescriptorSetObject>( mGraphicContext, this, aDescriptorCount );
    }

} // namespace SE::Graphics