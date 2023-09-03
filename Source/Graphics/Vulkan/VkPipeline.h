#pragma once

#include "Core/Memory.h"

#include "Graphics/Interface/IDescriptorSetLayout.h"
#include "Graphics/Interface/IGraphicBuffer.h"
#include "Graphics/Interface/IGraphicsPipeline.h"

#include "Graphics/Vulkan/VkGraphicContext.h"
#include "Graphics/Vulkan/VkRenderPass.h"

#include <memory>
#include <vulkan/vulkan.h>

namespace SE::Graphics
{
    using namespace SE::Core;

    struct sVkShaderModuleObject
    {
        VkShaderModule mVkObject = VK_NULL_HANDLE;

        sVkShaderModuleObject()                          = default;
        sVkShaderModuleObject( sVkShaderModuleObject & ) = default;
        sVkShaderModuleObject( ref_t<VkGraphicContext> mContext, std::vector<uint32_t> aByteCode );

        ~sVkShaderModuleObject();

      private:
        ref_t<VkGraphicContext> mContext = nullptr;
    };

    struct sVkDescriptorSetLayoutObject
    {
        VkDescriptorSetLayout mVkObject = VK_NULL_HANDLE;

        sVkDescriptorSetLayoutObject()                                 = default;
        sVkDescriptorSetLayoutObject( sVkDescriptorSetLayoutObject & ) = default;
        sVkDescriptorSetLayoutObject( ref_t<VkGraphicContext> mContext, std::vector<VkDescriptorSetLayoutBinding> aBindings,
                                      bool aUnbounded );

        ~sVkDescriptorSetLayoutObject();

      private:
        ref_t<VkGraphicContext> mContext = nullptr;
    };

    /** @class ShaderModule
     *
     */
    class ShaderModule
    {
      public:
        /** @brief Constructor
         *
         * @param l_Dev The Vulkan device for which the shader module is to be created
         * @param aFilePaths  A list of source files. The files in the list will be read in sequence
         *                     to produce a single source string which will be passed to shaderc for
         *                     compilation
         * @param aShaderType The type of the shader we are compiling.
         */
        ShaderModule( ref_t<VkGraphicContext> mContext, string_t aFilePaths, eShaderStageTypeFlags aShaderType );

        ShaderModule( ref_t<VkGraphicContext> mContext, std::vector<uint32_t> aShaderCode, eShaderStageTypeFlags aShaderType );

        ~ShaderModule() = default;

        /** @brief Retrieves the internal Vulkan shader stage creation structure
         *
         * Calculates the appropriate shader stage creation structure for the compiled
         * shader module.
         *
         * @returns The shader stage configuration.
         */
        VkPipelineShaderStageCreateInfo GetShaderStage();

        VkShaderModule GetVkShaderModule()
        {
            return mShaderModuleObject->mVkObject;
        }

        eShaderStageTypeFlags Type;

      private:
        ref_t<sVkShaderModuleObject> mShaderModuleObject = nullptr;
    };

    struct sVkDescriptorSetObject
    {
        struct sImageBindInfo
        {
            std::vector<VkSampler>   mSampler   = {};
            std::vector<VkImageView> mImageView = {};
            std::vector<bool>        mIsDepth   = {};
            uint32_t                 mBinding   = 0;
        };

        struct sBufferBindInfo
        {
            VkBuffer    mBuffer        = VK_NULL_HANDLE;
            eBufferType mType          = eBufferType::UNIFORM_BUFFER;
            bool        mDynamicOffset = false;
            uint32_t    mBinding       = 0;
            uint32_t    mOffset        = 0;
            uint32_t    mSize          = 0;
        };

        VkDescriptorSet mVkObject = VK_NULL_HANDLE;

        sVkDescriptorSetObject()                           = default;
        sVkDescriptorSetObject( sVkDescriptorSetObject & ) = default;
        sVkDescriptorSetObject( ref_t<VkGraphicContext> aContext, VkDescriptorSet aDescriporSet );

        void Write( sBufferBindInfo aBuffers );
        void Write( sImageBindInfo aImages );

        ~sVkDescriptorSetObject();

      private:
        ref_t<VkGraphicContext> mContext        = nullptr;
        VkDescriptorPool      mDescriptorPool = VK_NULL_HANDLE;
    };

    struct sVkDescriptorPoolObject
    {
        VkDescriptorPool mVkObject = VK_NULL_HANDLE;

        sVkDescriptorPoolObject()                            = default;
        sVkDescriptorPoolObject( sVkDescriptorPoolObject & ) = default;
        sVkDescriptorPoolObject( ref_t<VkGraphicContext> mContext, uint32_t aDescriptorSetCount,
                                 std::vector<VkDescriptorPoolSize> aPoolSizes );

        ref_t<sVkDescriptorSetObject> Allocate( ref_t<sVkDescriptorSetLayoutObject> aLayout, uint32_t aDescriptorCount = 0 );

        ~sVkDescriptorPoolObject();

      private:
        ref_t<VkGraphicContext> mContext = nullptr;
    };

    struct sVkPipelineLayoutObject
    {
        VkPipelineLayout mVkObject = VK_NULL_HANDLE;

        sVkPipelineLayoutObject()                            = default;
        sVkPipelineLayoutObject( sVkPipelineLayoutObject & ) = default;
        sVkPipelineLayoutObject( ref_t<VkGraphicContext> aContext, std::vector<ref_t<sVkDescriptorSetLayoutObject>> aDescriptorSetLayout,
                                 std::vector<sPushConstantRange> aPushConstantRanges );

        ~sVkPipelineLayoutObject();

      private:
        ref_t<VkGraphicContext> mContext = nullptr;
    };

    struct sShader
    {
        ref_t<ShaderModule> mShaderModule;
        string_t       mEntryPoint;
    };

    struct sVkPipelineObject
    {
        VkPipeline mVkObject = VK_NULL_HANDLE;

        sVkPipelineObject()                      = default;
        sVkPipelineObject( sVkPipelineObject & ) = default;
        sVkPipelineObject( ref_t<VkGraphicContext> aContext, uint8_t aSampleCount, std::vector<sBufferLayoutElement> aVertexBufferLayout,
                           std::vector<sBufferLayoutElement> aInstanceBufferLayout, ePrimitiveTopology aTopology,
                           eFaceCulling aCullMode, float aLineWidth, sDepthTesting aDepthTest, sBlending aBlending,
                           std::vector<sShader> aShaderStages, ref_t<sVkPipelineLayoutObject> aPipelineLayout,
                           ref_t<VkRenderPassObject> aRenderPass );

        ~sVkPipelineObject();

      private:
        ref_t<VkGraphicContext> mContext = nullptr;

        void Compile( std::vector<sBufferLayoutElement> &aVertexBufferLayout, uint32_t aBinding, uint32_t aStride,
                      VkVertexInputBindingDescription &o_Binding, std::vector<VkVertexInputAttributeDescription> &o_Attributes,
                      bool aInstanced );

        uint32_t CalculateOffsetsAndStride( std::vector<sBufferLayoutElement> &aVertexBufferLayout );
    };
} // namespace SE::Graphics