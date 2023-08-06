#include "VkDescriptorSet.h"

#include "Core/Memory.h"
#include <stdexcept>

#include "VkDescriptorSetLayout.h"

namespace SE::Graphics
{

    VkDescriptorSetObject::VkDescriptorSetObject( ref_t<IGraphicContext> aGraphicContext, IDescriptorSetLayout *aLayout,
                                                  uint32_t aDescriptorCount )
        : IDescriptorSet{ aGraphicContext, aLayout, aDescriptorCount }
    {
        mDescriptorSetObject = SE::Core::New<sVkDescriptorSetObject>(
            Cast<VkGraphicContext>( mGraphicContext ),
            Cast<VkGraphicContext>( mGraphicContext )
                ->AllocateDescriptorSet( Cast<VkDescriptorSetLayoutObject>( aLayout )->GetVkDescriptorSetLayoutObject()->mVkObject,
                                         aDescriptorCount ) );
    }

    void VkDescriptorSetObject::Write( ref_t<IGraphicBuffer> aBuffer, bool aDynamicOffset, uint32_t aOffset, uint32_t aSize,
                                       uint32_t aBinding )
    {
        sVkDescriptorSetObject::sBufferBindInfo lBufferBindInfo{};
        lBufferBindInfo.mBuffer        = Cast<VkGpuBuffer>( aBuffer )->mVkBuffer;
        lBufferBindInfo.mType          = Cast<VkGpuBuffer>( aBuffer )->mType;
        lBufferBindInfo.mDynamicOffset = false;
        lBufferBindInfo.mBinding       = aBinding;
        lBufferBindInfo.mOffset        = aOffset;
        lBufferBindInfo.mSize          = aSize;

        mDescriptorSetObject->Write( lBufferBindInfo );
    }

    void VkDescriptorSetObject::Write( vector_t<ref_t<ISampler2D>> aWriteOperations, uint32_t aBinding )
    {
        if( aWriteOperations.size() == 0 )
            return;

        sVkDescriptorSetObject::sImageBindInfo lImages{};

        for( auto &lBuffer : aWriteOperations )
        {
            lImages.mSampler.push_back( Cast<VkSampler2D>( lBuffer )->GetSampler() );
            lImages.mImageView.push_back( Cast<VkSampler2D>( lBuffer )->GetImageView() );
        }
        lImages.mBinding = aBinding;

        mDescriptorSetObject->Write( lImages );
    }
    // void VkDescriptorSetObject::Write( Ref<VkSampler2D> aBuffer, uint32_t aBinding ) { Write( vector_t{ aBuffer }, aBinding ); }

    void VkDescriptorSetObject::Write( vector_t<ref_t<ISamplerCubeMap>> aWriteOperations, uint32_t aBinding )
    {
        if( aWriteOperations.size() == 0 )
            return;

        sVkDescriptorSetObject::sImageBindInfo lImages{};

        for( auto &lBuffer : aWriteOperations )
        {
            lImages.mSampler.push_back( Cast<VkSamplerCubeMap>( lBuffer )->GetSampler() );
            lImages.mImageView.push_back( Cast<VkSamplerCubeMap>( lBuffer )->GetImageView() );
        }
        lImages.mBinding = aBinding;

        mDescriptorSetObject->Write( lImages );
    }
    // void VkDescriptorSetObject::Write( Ref<VkSamplerCubeMap> aBuffer, uint32_t aBinding )
    // {
    //     Write( vector_t{ aBuffer }, aBinding );
    // }

    // DescriptorBindingInfo::operator VkDescriptorSetLayoutBinding() const
    // {
    //     VkDescriptorSetLayoutBinding lNewBinding = {};
    //     lNewBinding.binding                      = mBindingIndex;
    //     lNewBinding.descriptorCount              = 1;
    //     lNewBinding.descriptorType               = (VkDescriptorType)Type;
    //     lNewBinding.pImmutableSamplers           = nullptr;

    //     if( mShaderStages & eShaderStageTypeFlags::VERTEX ) lNewBinding.stageFlags |= VK_SHADER_STAGE_VERTEX_BIT;

    //     if( mShaderStages & eShaderStageTypeFlags::COMPUTE ) lNewBinding.stageFlags |= VK_SHADER_STAGE_COMPUTE_BIT;

    //     if( mShaderStages & eShaderStageTypeFlags::GEOMETRY ) lNewBinding.stageFlags |= VK_SHADER_STAGE_GEOMETRY_BIT;

    //     if( mShaderStages & eShaderStageTypeFlags::FRAGMENT ) lNewBinding.stageFlags |= VK_SHADER_STAGE_FRAGMENT_BIT;

    //     if( mShaderStages & eShaderStageTypeFlags::TESSELATION_CONTROL )
    //         lNewBinding.stageFlags |= VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;

    //     if( mShaderStages & eShaderStageTypeFlags::TESSELATION_EVALUATION )
    //         lNewBinding.stageFlags |= VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;

    //     return lNewBinding;
    // }

    // DescriptorSetLayout::DescriptorSetLayout( Ref<VkGraphicContext> aGraphicContext, DescriptorSetLayoutCreateInfo &aCreateInfo,
    //                                           bool aUnbounded )
    //     : mGraphicContext{ aGraphicContext }
    //     , Spec( aCreateInfo )
    // {
    //     vector_t<VkDescriptorSetLayoutBinding> lBindings( aCreateInfo.Bindings.size() );

    //     for( uint32_t i = 0; i < aCreateInfo.Bindings.size(); i++ )
    //     {
    //         lBindings[i] = static_cast<VkDescriptorSetLayoutBinding>( aCreateInfo.Bindings[i] );
    //     }

    //     if( aUnbounded )
    //     {
    //         lBindings[lBindings.size() - 1].descriptorCount = 1024;
    //     }

    //     mDescriptorSetLayoutObject = New<sVkDescriptorSetLayoutObject>( mGraphicContext, lBindings, aUnbounded );
    // }

    // void VkDescriptorSetObject::Build()
    // {
    //     auto lDescriptorSet = mDescriptorLayout[i].mDescriptors;

    //     vector_t<VkDescriptorSetLayoutBinding> lBindings{};

    //     for( uint32_t j = 0; j < lDescriptorSet.size(); j++ )
    //     {
    //         VkDescriptorSetLayoutBinding &lNewBinding = lBindings.emplace_back();

    //         lNewBinding.binding            = lDescriptorSet[i].mBindingIndex;
    //         lNewBinding.descriptorCount    = 1;
    //         lNewBinding.descriptorType     = (VkDescriptorType)lDescriptorSet[i].mType;
    //         lNewBinding.pImmutableSamplers = nullptr;

    //         if( lDescriptorSet[j].mShaderStages & eShaderStageTypeFlags::VERTEX ) lNewBinding.stageFlags |=
    //         VK_SHADER_STAGE_VERTEX_BIT;

    //         if( lDescriptorSet[j].mShaderStages & eShaderStageTypeFlags::COMPUTE )
    //             lNewBinding.stageFlags |= VK_SHADER_STAGE_COMPUTE_BIT;

    //         if( lDescriptorSet[j].mShaderStages & eShaderStageTypeFlags::GEOMETRY )
    //             lNewBinding.stageFlags |= VK_SHADER_STAGE_GEOMETRY_BIT;

    //         if( lDescriptorSet[j].mShaderStages & eShaderStageTypeFlags::FRAGMENT )
    //             lNewBinding.stageFlags |= VK_SHADER_STAGE_FRAGMENT_BIT;

    //         if( lDescriptorSet[j].mShaderStages & eShaderStageTypeFlags::TESSELATION_CONTROL )
    //             lNewBinding.stageFlags |= VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;

    //         if( lDescriptorSet[j].mShaderStages & eShaderStageTypeFlags::TESSELATION_EVALUATION )
    //             lNewBinding.stageFlags |= VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
    //     }

    //     if( mIsUnbounded ) lBindings[lBindings.size() - 1].descriptorCount = 1024;

    //     mLayout        = New<sVkDescriptorSetLayoutObject>( Cast<VkGraphicContext>( mGraphicContext ), lBindings, mIsUnbounded );
    //     mDescriptorSet = New<sVkDescriptorSetObject>(
    //         Cast<VkGraphicContext>( mGraphicContext ),
    //         Cast<VkGraphicContext>( mGraphicContext )->AllocateDescriptorSet( mLayout->GetVkDescriptorSetLayoutObject()->mVkObject,
    //         mDescriptorCount ) );
    // }

} // namespace SE::Graphics