#pragma once

#include "Core/GraphicContext/Window.h"
#include "Core/Memory.h"
#include "Core/Vulkan/VkCommand.h"
#include "Core/Vulkan/VkContext.h"

namespace SE::Graphics
{

    using namespace SE::Core;

    class GraphicContext
    {
      public:
        Ref<Internal::VkContext> mContext        = nullptr;
        Ref<Window>              mViewportClient = nullptr;

      public:
        GraphicContext() = default;
        GraphicContext( uint32_t aWidth, uint32_t aHeight, uint32_t aSampleCount, std::string aTitle );

        ~GraphicContext() = default;

        Ref<Window> GetViewportClient() { return mViewportClient; };

        Ref<Internal::sVkDescriptorSetObject> AllocateDescriptors( Ref<Internal::sVkDescriptorSetLayoutObject> aLayout,
                                                                   uint32_t                                    aDescriptorCount = 0 );

        Ref<Internal::sVkCommandBufferObject> BeginSingleTimeCommands();

        void EndSingleTimeCommands( Ref<Internal::sVkCommandBufferObject> commandBuffer );

        void WaitIdle();

      private:
        Ref<Internal::sVkDescriptorPoolObject> mDescriptorPool;
    };
} // namespace SE::Graphics