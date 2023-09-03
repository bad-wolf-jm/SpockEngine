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

    void VkDescriptorSetObject::Write( std::vector<ref_t<ISampler2D>> aWriteOperations, uint32_t aBinding )
    {
        if( aWriteOperations.size() == 0 )
            return;

        sVkDescriptorSetObject::sImageBindInfo lImages{};

        for( auto &lBuffer : aWriteOperations )
        {
            lImages.mSampler.push_back( Cast<VkSampler2D>( lBuffer )->GetSampler() );
            lImages.mImageView.push_back( Cast<VkSampler2D>( lBuffer )->GetImageView() );
            lImages.mIsDepth.push_back( Cast<VkSampler2D>( lBuffer )->GetTexture()->mSpec.mIsDepthTexture );
        }

        lImages.mBinding = aBinding;

        mDescriptorSetObject->Write( lImages );
    }
    // void VkDescriptorSetObject::Write( ref_t<VkSampler2D> aBuffer, uint32_t aBinding ) { Write( std::vector{ aBuffer }, aBinding ); }

    void VkDescriptorSetObject::Write( std::vector<ref_t<ISamplerCubeMap>> aWriteOperations, uint32_t aBinding )
    {
        if( aWriteOperations.size() == 0 )
            return;

        sVkDescriptorSetObject::sImageBindInfo lImages{};

        for( auto &lBuffer : aWriteOperations )
        {
            lImages.mSampler.push_back( Cast<VkSamplerCubeMap>( lBuffer )->GetSampler() );
            lImages.mImageView.push_back( Cast<VkSamplerCubeMap>( lBuffer )->GetImageView() );
            lImages.mIsDepth.push_back( Cast<VkSampler2D>( lBuffer )->GetTexture()->mSpec.mIsDepthTexture );
        }
        lImages.mBinding = aBinding;

        mDescriptorSetObject->Write( lImages );
    }
} // namespace SE::Graphics