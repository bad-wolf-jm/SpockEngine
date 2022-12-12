#include "GraphicContext.h"

namespace SE::Graphics
{

    GraphicContext::GraphicContext( uint32_t a_Width, uint32_t a_Height, uint32_t a_SampleCount, std::string a_Title )
    {
        mViewportClient = SE::Core::New<IWindow>( a_Width, a_Height, a_Title );
        mContext        = SE::Core::New<Internal::VkGraphicContext>( mViewportClient, a_SampleCount, true );

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

    Ref<Internal::sVkCommandBufferObject> GraphicContext::BeginSingleTimeCommands()
    {
        Ref<Internal::sVkCommandBufferObject> lCommandBuffer = SE::Core::New<Internal::sVkCommandBufferObject>( mContext );
        lCommandBuffer->Begin( VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT );
        return lCommandBuffer;
    }

    void GraphicContext::EndSingleTimeCommands( Ref<Internal::sVkCommandBufferObject> aCommandBuffer )
    {
        aCommandBuffer->End();
        aCommandBuffer->SubmitTo( mContext->GetGraphicsQueue() );
        mContext->WaitIdle( mContext->GetGraphicsQueue() );
    }

    void GraphicContext::WaitIdle() { mContext->WaitIdle(); }
} // namespace SE::Graphics