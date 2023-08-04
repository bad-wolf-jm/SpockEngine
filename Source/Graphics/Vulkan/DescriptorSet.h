#pragma once

#include "Core/Memory.h"

#include "Graphics/Interface/IDescriptorSetLayout.h"

#include "Graphics/Vulkan/VkGraphicContext.h"
#include "Graphics/Vulkan/VkPipeline.h"
#include "Graphics/Vulkan/VkSampler2D.h"
#include "Graphics/Vulkan/VkSamplerCubeMap.h"

#include <vector>
#include <vulkan/vulkan.h>

namespace SE::Graphics
{
    struct DescriptorBindingInfo
    {
        uint32_t        mBindingIndex = 0;
        eDescriptorType Type          = eDescriptorType::UNIFORM_BUFFER;
        ShaderStageType mShaderStages = {};
        bool            Unbounded     = false;

        operator VkDescriptorSetLayoutBinding() const;
    };

    struct DescriptorSetLayoutCreateInfo
    {
        vector_t<DescriptorBindingInfo> Bindings;
    };

    class DescriptorSetLayout
    {
      public:
        DescriptorSetLayout( Ref<VkGraphicContext> aGraphicContext, DescriptorSetLayoutCreateInfo &aCreateInfo,
                             bool aUnbounded = false );
        ~DescriptorSetLayout() = default;

        DescriptorSetLayoutCreateInfo Spec;

        Ref<sVkDescriptorSetLayoutObject> GetVkDescriptorSetLayoutObject()
        {
            return mDescriptorSetLayoutObject;
        }

        VkDescriptorSetLayout GetVkDescriptorSetLayout()
        {
            return mDescriptorSetLayoutObject->mVkObject;
        }

        operator VkDescriptorSetLayout() const
        {
            return mDescriptorSetLayoutObject->mVkObject;
        };

      private:
        Ref<VkGraphicContext>             mGraphicContext{};
        Ref<sVkDescriptorSetLayoutObject> mDescriptorSetLayoutObject = nullptr;
    };

    template <typename _BufType>
    struct DescriptorBufferInfo
    {
        Ref<_BufType> Buffer        = nullptr;
        bool          DynamicOffset = false;
        uint32_t      Binding       = 0;
        uint32_t      Offset        = 0;
        uint32_t      Size          = 0;
    };

    class DescriptorSet
    {
      public:
        DescriptorSet( Ref<VkGraphicContext> aGraphicContext, Ref<DescriptorSetLayout> aLayout, uint32_t aDescriptorCount = 0 );
        ~DescriptorSet() = default;

        void Write( Ref<VkGpuBuffer> aBuffer, bool aDynamicOffset, uint32_t aOffset, uint32_t aSize, uint32_t aBinding );

        void Write( Ref<VkSampler2D> aBuffer, uint32_t aBinding );
        void Write( vector_t<Ref<VkSampler2D>> aBuffer, uint32_t aBinding );

        void Write( Ref<VkSamplerCubeMap> aBuffer, uint32_t aBinding );
        void Write( vector_t<Ref<VkSamplerCubeMap>> aBuffer, uint32_t aBinding );

        VkDescriptorSet GetVkDescriptorSet()
        {
            return mDescriptorSetObject->mVkObject;
        }

        Ref<sVkDescriptorSetObject> GetVkDescriptorSetObject()
        {
            return mDescriptorSetObject;
        }

      private:
        Ref<VkGraphicContext>       mGraphicContext{};
        Ref<DescriptorSetLayout>    mLayout              = nullptr;
        Ref<sVkDescriptorSetObject> mDescriptorSetObject = nullptr;
    };

} // namespace SE::Graphics