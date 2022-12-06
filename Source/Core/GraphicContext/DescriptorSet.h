#pragma once

#include "Core/GraphicContext//GraphicContext.h"
#include "Core/GraphicContext//Texture2D.h"
#include "Core/GraphicContext//TextureCubeMap.h"

#include "Graphics/VkSampler2D.h"

#include "Core/Memory.h"
#include "Core/Vulkan/VkPipeline.h"

#include <vector>
#include <vulkan/vulkan.h>

namespace SE::Graphics
{
    struct DescriptorBindingInfo
    {
        uint32_t                  mBindingIndex = 0;
        Internal::eDescriptorType Type          = Internal::eDescriptorType::UNIFORM_BUFFER;
        Internal::ShaderStageType mShaderStages = {};
        bool                      Unbounded     = false;

        operator VkDescriptorSetLayoutBinding() const;
    };

    struct DescriptorSetLayoutCreateInfo
    {
        std::vector<DescriptorBindingInfo> Bindings;
    };

    class DescriptorSetLayout
    {
      public:
        DescriptorSetLayout( GraphicContext &aGraphicContext, DescriptorSetLayoutCreateInfo &aCreateInfo, bool aUnbounded = false );
        ~DescriptorSetLayout() = default;

        DescriptorSetLayoutCreateInfo Spec;

        Ref<Internal::sVkDescriptorSetLayoutObject> GetVkDescriptorSetLayoutObject() { return mDescriptorSetLayoutObject; }

        VkDescriptorSetLayout GetVkDescriptorSetLayout() { return mDescriptorSetLayoutObject->mVkObject; }

        operator VkDescriptorSetLayout() const { return mDescriptorSetLayoutObject->mVkObject; };

      private:
        GraphicContext                              mGraphicContext{};
        Ref<Internal::sVkDescriptorSetLayoutObject> mDescriptorSetLayoutObject = nullptr;
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
        DescriptorSet( GraphicContext &aGraphicContext, Ref<DescriptorSetLayout> aLayout, uint32_t aDescriptorCount = 0 );
        ~DescriptorSet() = default;

        void Write( Ref<Buffer> aBuffer, bool aDynamicOffset, uint32_t aOffset, uint32_t aSize, uint32_t aBinding );

        void Write( Ref<Texture2D> aBuffer, uint32_t aBinding );
        void Write( std::vector<Ref<Texture2D>> aBuffer, uint32_t aBinding );

        void Write( Ref<VkSampler2D> aBuffer, uint32_t aBinding );
        void Write( std::vector<Ref<VkSampler2D>> aBuffer, uint32_t aBinding );

        VkDescriptorSet GetVkDescriptorSet() { return mDescriptorSetObject->mVkObject; }

        Ref<Internal::sVkDescriptorSetObject> GetVkDescriptorSetObject() { return mDescriptorSetObject; }

      private:
        GraphicContext                        mGraphicContext{};
        Ref<DescriptorSetLayout>              mLayout              = nullptr;
        Ref<Internal::sVkDescriptorSetObject> mDescriptorSetObject = nullptr;
    };

} // namespace SE::Graphics