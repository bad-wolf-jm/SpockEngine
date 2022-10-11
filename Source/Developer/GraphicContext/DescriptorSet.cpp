#include "DescriptorSet.h"

#include "Core/Memory.h"
#include <stdexcept>

namespace LTSE::Graphics
{

    DescriptorSet::DescriptorSet( GraphicContext &aGraphicContext, Ref<DescriptorSetLayout> aLayout, uint32_t aDescriptorCount )
        : mGraphicContext{ aGraphicContext }
        , mLayout{ aLayout }
    {
        mDescriptorSetObject = mGraphicContext.AllocateDescriptors( aLayout->GetVkDescriptorSetLayoutObject(), aDescriptorCount );
    }

    void DescriptorSet::Write( Ref<Buffer> aBuffer, bool aDynamicOffset, uint32_t aOffset, uint32_t aSize, uint32_t aBinding )
    {
        Internal::sVkDescriptorSetObject::sBufferBindInfo lBufferBindInfo{};
        lBufferBindInfo.mBuffer       = aBuffer->mVkObject;
        lBufferBindInfo.mType          = aBuffer->mType;
        lBufferBindInfo.mDynamicOffset = false;
        lBufferBindInfo.mBinding       = aBinding;
        lBufferBindInfo.mOffset        = aOffset;
        lBufferBindInfo.mSize          = aSize;

        mDescriptorSetObject->Write( lBufferBindInfo );
    }

    void DescriptorSet::Write( std::vector<Ref<Texture2D>> aWriteOperations, uint32_t aBinding )
    {
        if( aWriteOperations.size() == 0 )
            return;

        Internal::sVkDescriptorSetObject::sImageBindInfo lImages{};

        for( auto &lBuffer : aWriteOperations )
        {
            lImages.mSampler.push_back( lBuffer->GetSampler() );
            lImages.mImageView.push_back( lBuffer->GetImageView() );
        }
        lImages.mBinding = aBinding;

        mDescriptorSetObject->Write( lImages );
    }

    void DescriptorSet::Write( Ref<Texture2D> aBuffer, uint32_t aBinding ) { Write( std::vector{ aBuffer }, aBinding ); }

    DescriptorBindingInfo::operator VkDescriptorSetLayoutBinding() const
    {
        VkDescriptorSetLayoutBinding lNewBinding = {};
        lNewBinding.binding                      = mBindingIndex;
        lNewBinding.descriptorCount              = 1;
        lNewBinding.descriptorType               = (VkDescriptorType)Type;
        lNewBinding.pImmutableSamplers           = nullptr;

        if( mShaderStages & Internal::eShaderStageTypeFlags::VERTEX )
            lNewBinding.stageFlags |= VK_SHADER_STAGE_VERTEX_BIT;

        if( mShaderStages & Internal::eShaderStageTypeFlags::COMPUTE )
            lNewBinding.stageFlags |= VK_SHADER_STAGE_COMPUTE_BIT;

        if( mShaderStages & Internal::eShaderStageTypeFlags::GEOMETRY )
            lNewBinding.stageFlags |= VK_SHADER_STAGE_GEOMETRY_BIT;

        if( mShaderStages & Internal::eShaderStageTypeFlags::FRAGMENT )
            lNewBinding.stageFlags |= VK_SHADER_STAGE_FRAGMENT_BIT;

        if( mShaderStages & Internal::eShaderStageTypeFlags::TESSELATION_CONTROL )
            lNewBinding.stageFlags |= VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;

        if( mShaderStages & Internal::eShaderStageTypeFlags::TESSELATION_EVALUATION )
            lNewBinding.stageFlags |= VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;

        return lNewBinding;
    }

    DescriptorSetLayout::DescriptorSetLayout( GraphicContext &aGraphicContext, DescriptorSetLayoutCreateInfo &aCreateInfo, bool aUnbounded )
        : mGraphicContext{ aGraphicContext }
        , Spec( aCreateInfo )
    {
        std::vector<VkDescriptorSetLayoutBinding> lBindings( aCreateInfo.Bindings.size() );

        for( uint32_t i = 0; i < aCreateInfo.Bindings.size(); i++ )
        {
            lBindings[i] = static_cast<VkDescriptorSetLayoutBinding>( aCreateInfo.Bindings[i] );
        }

        if( aUnbounded )
        {
            lBindings[lBindings.size() - 1].descriptorCount = 1024;
        }

        mDescriptorSetLayoutObject = New<Internal::sVkDescriptorSetLayoutObject>( mGraphicContext.mContext, lBindings, aUnbounded );
    }

} // namespace LTSE::Graphics