#pragma once

#include "Core/Memory.h"
#include "Core/Platform/ViewportClient.h"
#include "Core/Vulkan/VkCommand.h"
#include "Core/Vulkan/VkContext.h"


namespace LTSE::Graphics
{

    using namespace LTSE::Core;

    class GraphicContext
    {
      public:
        Ref<Internal::VkContext> mContext         = nullptr;
        Ref<Window>              m_ViewportClient = nullptr;

      public:
        GraphicContext() = default;
        GraphicContext( uint32_t a_Width, uint32_t a_Height, uint32_t a_SampleCount, std::string a_Title );

        ~GraphicContext() = default;

        Ref<Window> GetViewportClient() { return m_ViewportClient; };

        Ref<Internal::sVkDescriptorSetObject> AllocateDescriptors(
            Ref<Internal::sVkDescriptorSetLayoutObject> aLayout, uint32_t aDescriptorCount = 0 );

        Ref<Internal::sVkCommandBufferObject> BeginSingleTimeCommands();

        void EndSingleTimeCommands( Ref<Internal::sVkCommandBufferObject> commandBuffer );

        void WaitIdle();

      private:
        Ref<Internal::sVkDescriptorPoolObject> mDescriptorPool;
    };
} // namespace LTSE::Graphics