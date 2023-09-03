#pragma once

#include "Core/Memory.h"

#include "Graphics/Interface/IDescriptorSetLayout.h"
#include "Graphics/Interface/IGraphicContext.h"
#include "Graphics/Interface/IGraphicsPipeline.h"

#include "Graphics/Vulkan/VkSampler2D.h"
#include "Graphics/Vulkan/VkSamplerCubeMap.h"
#include "Graphics/Vulkan/VkGraphicContext.h"
#include "Graphics/Vulkan/VkPipeline.h"

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
        vec_t<DescriptorBindingInfo> Bindings;
    };

    class DescriptorSetLayout
    {
      public:
        DescriptorSetLayout( ref_t<VkGraphicContext> aGraphicContext, DescriptorSetLayoutCreateInfo &aCreateInfo,
                             bool aUnbounded = false );
        ~DescriptorSetLayout() = default;

        DescriptorSetLayoutCreateInfo Spec;

        ref_t<sVkDescriptorSetLayoutObject> GetVkDescriptorSetLayoutObject() { return mDescriptorSetLayoutObject; }

        VkDescriptorSetLayout GetVkDescriptorSetLayout() { return mDescriptorSetLayoutObject->mVkObject; }

        operator VkDescriptorSetLayout() const { return mDescriptorSetLayoutObject->mVkObject; };

      private:
        ref_t<VkGraphicContext>             mGraphicContext{};
        ref_t<sVkDescriptorSetLayoutObject> mDescriptorSetLayoutObject = nullptr;
    };

    template <typename _BufType>
    struct DescriptorBufferInfo
    {
        ref_t<_BufType> Buffer        = nullptr;
        bool          DynamicOffset = false;
        uint32_t      Binding       = 0;
        uint32_t      Offset        = 0;
        uint32_t      Size          = 0;
    };

    class DescriptorSet
    {
      public:
        DescriptorSet( ref_t<IGraphicsPipeline> aGraphicsPipeline, uint32_t aDescriptorCount = 0 );
        DescriptorSet( ref_t<VkGraphicContext> aGraphicContext, ref_t<DescriptorSetLayout> aLayout, uint32_t aDescriptorCount = 0 );
        ~DescriptorSet() = default;

        void Write( ref_t<VkGpuBuffer> aBuffer, bool aDynamicOffset, uint32_t aOffset, uint32_t aSize, uint32_t aBinding );
        void Write( ref_t<VkSampler2D> aBuffer, uint32_t aBinding );
        void Write( vec_t<ref_t<VkSampler2D>> aBuffer, uint32_t aBinding );

        void Write( ref_t<VkSamplerCubeMap> aBuffer, uint32_t aBinding );
        void Write( vec_t<ref_t<VkSamplerCubeMap>> aBuffer, uint32_t aBinding );

        VkDescriptorSet GetVkDescriptorSet() { return mDescriptorSetObject->mVkObject; }

        ref_t<sVkDescriptorSetObject> GetVkDescriptorSetObject() { return mDescriptorSetObject; }

      private:
        ref_t<VkGraphicContext>       mGraphicContext{};
        ref_t<DescriptorSetLayout>    mLayout              = nullptr;
        ref_t<sVkDescriptorSetObject> mDescriptorSetObject = nullptr;
    };

} // namespace SE::Graphics