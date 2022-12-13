#include "GraphicContext.h"

namespace SE::Graphics
{
    GraphicContext::GraphicContext( Ref<Internal::VkGraphicContext> aContext, Ref<IWindow> aWindow )
        : mViewportClient{ aWindow }
        , mContext{ aContext }
    {

        uint32_t                          lNumberOfDescriptorSets = 10000;
        std::vector<VkDescriptorPoolSize> lPoolSizes( 4 );
        lPoolSizes[0] = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 10000 };
        lPoolSizes[1] = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 10000 };
        lPoolSizes[2] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 10000 };
        lPoolSizes[3] = { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 10000 };

        mDescriptorPool = New<Internal::sVkDescriptorPoolObject>( mContext, lNumberOfDescriptorSets, lPoolSizes );
    }

    Ref<Internal::sVkDescriptorSetObject> GraphicContext::AllocateDescriptors( Ref<Internal::sVkDescriptorSetLayoutObject> aLayout,
                                                                               uint32_t aDescriptorCount )
    {
        return mDescriptorPool->Allocate( aLayout, aDescriptorCount );
    }
} // namespace SE::Graphics