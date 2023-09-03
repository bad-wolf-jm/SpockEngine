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
    // template <typename _BufType>
    // struct DescriptorBufferInfo
    // {
    //     ref_t<_BufType> Buffer        = nullptr;
    //     bool          DynamicOffset = false;
    //     uint32_t      Binding       = 0;
    //     uint32_t      Offset        = 0;
    //     uint32_t      Size          = 0;
    // };

    class VkDescriptorSetObject : public IDescriptorSet
    {
      public:
        VkDescriptorSetObject( ref_t<IGraphicContext> aGraphicContext, IDescriptorSetLayout *aLayout, uint32_t aDescriptorCount = 0 );
        ~VkDescriptorSetObject() = default;

        void *GetID()
        {
            return (void *)mDescriptorSetObject->mVkObject;
        }

        void Write( ref_t<IGraphicBuffer> aBuffer, bool aDynamicOffset, uint32_t aOffset, uint32_t aSize, uint32_t aBinding );
        void Write( vec_t<ref_t<ISampler2D>> aBuffer, uint32_t aBinding );
        void Write( vec_t<ref_t<ISamplerCubeMap>> aBuffer, uint32_t aBinding );

        void Build();

        VkDescriptorSet GetVkDescriptorSet()
        {
            return mDescriptorSetObject->mVkObject;
        }

        ref_t<sVkDescriptorSetObject> GetVkDescriptorSetObject()
        {
            return mDescriptorSetObject;
        }

      private:
        ref_t<sVkDescriptorSetObject> mDescriptorSetObject = nullptr;
    };

} // namespace SE::Graphics