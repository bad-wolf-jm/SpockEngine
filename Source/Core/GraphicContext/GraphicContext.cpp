#include "GraphicContext.h"

namespace LTSE::Graphics
{

    GraphicContext::GraphicContext( uint32_t a_Width, uint32_t a_Height, uint32_t a_SampleCount, std::string a_Title )
    {
        mViewportClient = LTSE::Core::New<Window>( a_Width, a_Height, a_Title );
        mContext         = LTSE::Core::New<Internal::VkContext>( mViewportClient, true );

        uint32_t                          lNumberOfDescriptorSets = 1000;
        std::vector<VkDescriptorPoolSize> lPoolSizes( 4 );
        lPoolSizes[0] = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 };
        lPoolSizes[1] = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 };
        lPoolSizes[2] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 };
        lPoolSizes[3] = { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 };

        mDescriptorPool = New<Internal::sVkDescriptorPoolObject>( mContext, lNumberOfDescriptorSets, lPoolSizes );
    }

    Ref<Internal::sVkDescriptorSetObject> GraphicContext::AllocateDescriptors(
        Ref<Internal::sVkDescriptorSetLayoutObject> aLayout, uint32_t aDescriptorCount )
    {
        return mDescriptorPool->Allocate( aLayout, aDescriptorCount );
    }

    Ref<Internal::sVkCommandBufferObject> GraphicContext::BeginSingleTimeCommands()
    {
        Ref<Internal::sVkCommandBufferObject> l_CommandBuffer = LTSE::Core::New<Internal::sVkCommandBufferObject>( mContext );
        l_CommandBuffer->Begin( VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT );
        return l_CommandBuffer;
    }

    void GraphicContext::EndSingleTimeCommands( Ref<Internal::sVkCommandBufferObject> a_CommandBuffer )
    {
        a_CommandBuffer->End();
        a_CommandBuffer->SubmitTo( mContext->GetGraphicsQueue() );
        mContext->WaitIdle( mContext->GetGraphicsQueue() );
    }

    void GraphicContext::WaitIdle() { mContext->WaitIdle(); }
} // namespace LTSE::Graphics