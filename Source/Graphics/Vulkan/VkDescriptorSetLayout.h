#pragma once

#include "Core/Memory.h"

#include "Graphics/Interface/IDescriptorSet.h"
#include "Graphics/Interface/IDescriptorSetLayout.h"
#include "Graphics/Interface/IGraphicContext.h"

#include "Graphics/Vulkan/VkGraphicContext.h"
#include "Graphics/Vulkan/VkPipeline.h"

#include <vector>
#include <vulkan/vulkan.h>

namespace SE::Graphics
{
    class VkDescriptorSetLayoutObject : public IDescriptorSetLayout
    {
      public:
        VkDescriptorSetLayoutObject( Ref<IGraphicContext> aGraphicContext, bool aIsUnbounded = false, uint32_t aDescriptorCount = 1 );
        ~VkDescriptorSetLayoutObject() = default;

        void Build();

        Ref<IDescriptorSet> Allocate( uint32_t aDescriptorCount = 1 );

        VkDescriptorSetLayout GetVkDescriptorSetLayout()
        {
            return mLayout->mVkObject;
        }
        Ref<sVkDescriptorSetLayoutObject> GetVkDescriptorSetLayoutObject()
        {
            return mLayout;
        }

      private:
        Ref<sVkDescriptorSetLayoutObject> mLayout = nullptr;
    };

} // namespace SE::Graphics