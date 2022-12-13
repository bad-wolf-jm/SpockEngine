#pragma once

#include "Core/Memory.h"
#include "Graphics/Interface/IWindow.h"
#include "Graphics/Vulkan/VkCommand.h"
#include "Graphics/Vulkan/VkGraphicContext.h"

namespace SE::Graphics
{

    using namespace SE::Core;

    class GraphicContext
    {
      public:
        Ref<Internal::VkGraphicContext> mContext        = nullptr;
        Ref<IWindow>                    mViewportClient = nullptr;

      public:
        GraphicContext() = default;
        GraphicContext( Ref<Internal::VkGraphicContext>, Ref<IWindow> );

        ~GraphicContext() = default;

        Ref<Internal::sVkDescriptorSetObject> AllocateDescriptors( Ref<Internal::sVkDescriptorSetLayoutObject> aLayout,
                                                                   uint32_t                                    aDescriptorCount = 0 );

      private:
        Ref<Internal::sVkDescriptorPoolObject> mDescriptorPool;
    };
} // namespace SE::Graphics