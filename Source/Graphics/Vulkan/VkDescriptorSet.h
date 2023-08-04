#pragma once

#include "Core/Memory.h"

#include "Graphics/Interface/IDescriptorSet.h"
#include "Graphics/Interface/IDescriptorSetLayout.h"

#include "Graphics/Vulkan/VkGraphicContext.h"
#include "Graphics/Vulkan/VkPipeline.h"
#include "Graphics/Vulkan/VkSampler2D.h"
#include "Graphics/Vulkan/VkSamplerCubeMap.h"

#include <vector>
#include <vulkan/vulkan.h>

namespace SE::Graphics
{
    template <typename _BufType>
    struct DescriptorBufferInfo
    {
        Ref<_BufType> Buffer        = nullptr;
        bool          DynamicOffset = false;
        uint32_t      Binding       = 0;
        uint32_t      Offset        = 0;
        uint32_t      Size          = 0;
    };

    class VkDescriptorSetObject : public IDescriptorSet
    {
      public:
        VkDescriptorSetObject( Ref<IGraphicContext> aGraphicContext, IDescriptorSetLayout *aLayout, uint32_t aDescriptorCount = 0 );
        ~VkDescriptorSetObject() = default;

        void *GetID()
        {
            return (void *)mDescriptorSetObject->mVkObject;
        }

        void Write( Ref<IGraphicBuffer> aBuffer, bool aDynamicOffset, uint32_t aOffset, uint32_t aSize, uint32_t aBinding );
        void Write( vector_t<Ref<ISampler2D>> aBuffer, uint32_t aBinding );
        void Write( vector_t<Ref<ISamplerCubeMap>> aBuffer, uint32_t aBinding );

        void Build();

        VkDescriptorSet GetVkDescriptorSet()
        {
            return mDescriptorSetObject->mVkObject;
        }

        Ref<sVkDescriptorSetObject> GetVkDescriptorSetObject()
        {
            return mDescriptorSetObject;
        }

      private:
        Ref<sVkDescriptorSetObject> mDescriptorSetObject = nullptr;
    };

} // namespace SE::Graphics