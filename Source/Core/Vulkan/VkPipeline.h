#pragma once

#include "Core/Memory.h"

#include "VkContext.h"
#include "VkRenderPass.h"

#include <memory>
#include <vulkan/vulkan.h>

namespace LTSE::Graphics::Internal
{
    using namespace LTSE::Core;

    struct sVkShaderModuleObject
    {
        VkShaderModule mVkObject = VK_NULL_HANDLE;

        sVkShaderModuleObject()                          = default;
        sVkShaderModuleObject( sVkShaderModuleObject & ) = default;
        sVkShaderModuleObject( Ref<VkContext> mContext, std::vector<uint32_t> aByteCode );

        ~sVkShaderModuleObject();

      private:
        Ref<VkContext> mContext = nullptr;
    };

    struct sVkDescriptorSetLayoutObject
    {
        VkDescriptorSetLayout mVkObject = VK_NULL_HANDLE;

        sVkDescriptorSetLayoutObject()                                 = default;
        sVkDescriptorSetLayoutObject( sVkDescriptorSetLayoutObject & ) = default;
        sVkDescriptorSetLayoutObject( Ref<VkContext> mContext, std::vector<VkDescriptorSetLayoutBinding> aBindings, bool aUnbounded );

        ~sVkDescriptorSetLayoutObject();

      private:
        Ref<VkContext> mContext = nullptr;
    };

    enum class eShaderStageTypeFlags : uint32_t
    {
        VERTEX                 = VK_SHADER_STAGE_VERTEX_BIT,
        GEOMETRY               = VK_SHADER_STAGE_GEOMETRY_BIT,
        FRAGMENT               = VK_SHADER_STAGE_FRAGMENT_BIT,
        TESSELATION_CONTROL    = VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT,
        TESSELATION_EVALUATION = VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT,
        COMPUTE                = VK_SHADER_STAGE_COMPUTE_BIT,
        DEFAULT                = 0xffffffff,
    };

    /** @brief */
    using ShaderStageType = EnumSet<eShaderStageTypeFlags, 0x000001ff>;

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
        ShaderModule( Ref<VkContext> mContext, std::string FilePaths, eShaderStageTypeFlags aShaderType );

        ~ShaderModule() = default;

        /** @brief Retrieves the internal Vulkan shader stage creation structure
         *
         * Calculates the appropriate shader stage creation structure for the compiled
         * shader module.
         *
         * @returns The shader stage configuration.
         */
        VkPipelineShaderStageCreateInfo GetShaderStage();

        VkShaderModule GetVkShaderModule() { return mShaderModuleObject->mVkObject; }

        eShaderStageTypeFlags Type;

      private:
        Ref<sVkShaderModuleObject> mShaderModuleObject = nullptr;
    };

    enum class eDescriptorType : uint32_t
    {
        SAMPLER                = VK_DESCRIPTOR_TYPE_SAMPLER,
        COMBINED_IMAGE_SAMPLER = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        SAMPLED_IMAGE          = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
        STORAGE_IMAGE          = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        UNIFORM_TEXEL_BUFFER   = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,
        STORAGE_TEXEL_BUFFER   = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER,
        UNIFORM_BUFFER         = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        STORAGE_BUFFER         = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        UNIFORM_BUFFER_DYNAMIC = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
        STORAGE_BUFFER_DYNAMIC = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC,
        INPUT_ATTACHMENT       = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT
    };

    struct sVkDescriptorSetObject
    {
        struct sImageBindInfo
        {
            std::vector<VkSampler>   mSampler   = {};
            std::vector<VkImageView> mImageView = {};
            uint32_t                 mBinding   = 0;
        };

        struct sBufferBindInfo
        {
            VkBuffer        mBuffer        = VK_NULL_HANDLE;
            eBufferBindType mType          = eBufferBindType::UNIFORM_BUFFER;
            bool            mDynamicOffset = false;
            uint32_t        mBinding       = 0;
            uint32_t        mOffset        = 0;
            uint32_t        mSize          = 0;
        };

        VkDescriptorSet mVkObject = VK_NULL_HANDLE;

        sVkDescriptorSetObject()                           = default;
        sVkDescriptorSetObject( sVkDescriptorSetObject & ) = default;
        sVkDescriptorSetObject( Ref<VkContext> aContext, VkDescriptorPool aDescriptorPool, VkDescriptorSet aDescriporSet );

        void Write( sBufferBindInfo aBuffers );
        void Write( sImageBindInfo aImages );

        ~sVkDescriptorSetObject();

      private:
        Ref<VkContext>   mContext        = nullptr;
        VkDescriptorPool mDescriptorPool = VK_NULL_HANDLE;
    };

    struct sVkDescriptorPoolObject
    {
        VkDescriptorPool mVkObject = VK_NULL_HANDLE;

        sVkDescriptorPoolObject()                            = default;
        sVkDescriptorPoolObject( sVkDescriptorPoolObject & ) = default;
        sVkDescriptorPoolObject( Ref<VkContext> mContext, uint32_t aDescriptorSetCount, std::vector<VkDescriptorPoolSize> aPoolSizes );

        Ref<sVkDescriptorSetObject> Allocate( Ref<sVkDescriptorSetLayoutObject> aLayout, uint32_t aDescriptorCount = 0 );

        ~sVkDescriptorPoolObject();

      private:
        Ref<VkContext> mContext = nullptr;
    };

    struct sPushConstantRange
    {
        ShaderStageType mShaderStages = { eShaderStageTypeFlags::VERTEX };
        uint32_t        mOffset       = 0;
        uint32_t        mSize         = 0;
    };

    struct sVkPipelineLayoutObject
    {
        VkPipelineLayout mVkObject = VK_NULL_HANDLE;

        sVkPipelineLayoutObject()                            = default;
        sVkPipelineLayoutObject( sVkPipelineLayoutObject & ) = default;
        sVkPipelineLayoutObject( Ref<VkContext> aContext, std::vector<Ref<sVkDescriptorSetLayoutObject>> aDescriptorSetLayout,
                                 std::vector<sPushConstantRange> aPushConstantRanges );

        ~sVkPipelineLayoutObject();

      private:
        Ref<VkContext> mContext = nullptr;
    };

    enum class ePrimitiveTopology : uint32_t
    {
        POINTS    = VK_PRIMITIVE_TOPOLOGY_POINT_LIST,
        TRIANGLES = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        LINES     = VK_PRIMITIVE_TOPOLOGY_LINE_LIST
    };

    enum class eFaceCulling : uint32_t
    {
        NONE           = VK_CULL_MODE_NONE,
        FRONT          = VK_CULL_MODE_FRONT_BIT,
        BACK           = VK_CULL_MODE_BACK_BIT,
        FRONT_AND_BACK = VK_CULL_MODE_FRONT_AND_BACK
    };

    struct sShader
    {
        Ref<ShaderModule> mShaderModule;
        std::string       mEntryPoint;
    };

    enum class eBufferDataType : uint32_t
    {
        UINT8  = VK_FORMAT_R8_UINT,
        UINT16 = VK_FORMAT_R16_UINT,
        UINT32 = VK_FORMAT_R32_UINT,
        INT8   = VK_FORMAT_R8_SINT,
        INT16  = VK_FORMAT_R16_SINT,
        INT32  = VK_FORMAT_R32_SINT,
        FLOAT  = VK_FORMAT_R32_SFLOAT,
        COLOR  = VK_FORMAT_R8G8B8A8_UNORM,
        VEC2   = VK_FORMAT_R32G32_SFLOAT,
        VEC3   = VK_FORMAT_R32G32B32_SFLOAT,
        VEC4   = VK_FORMAT_R32G32B32A32_SFLOAT,
        IVEC2  = VK_FORMAT_R32G32_SINT,
        IVEC3  = VK_FORMAT_R32G32B32_SINT,
        IVEC4  = VK_FORMAT_R32G32B32A32_SINT,
        UVEC2  = VK_FORMAT_R32G32_UINT,
        UVEC3  = VK_FORMAT_R32G32B32_UINT,
        UVEC4  = VK_FORMAT_R32G32B32A32_UINT
    };

    /** @brief */
    struct sBufferLayoutElement
    {
        std::string     mName;
        eBufferDataType mType;
        uint32_t        mBinding;
        uint32_t        mLocation;
        size_t          mSize;
        size_t          mOffset;

        sBufferLayoutElement() = default;
        sBufferLayoutElement( const std::string &aName, eBufferDataType aType, uint32_t aBinding, uint32_t aLocation );
    };

    /** @brief */
    struct sBufferLayout
    {
      public:
        sBufferLayout() {}
        sBufferLayout( std::initializer_list<sBufferLayoutElement> aElements );

        sBufferLayout &operator=( std::initializer_list<sBufferLayoutElement> aElements )
        {
            mElements = aElements;
            CalculateOffsetsAndStride();
            return *this;
        }

        uint32_t GetStride() const { return mStride; }

        const std::vector<sBufferLayoutElement> &GetElements() const { return mElements; }

        std::vector<sBufferLayoutElement>::iterator begin() { return mElements.begin(); }
        std::vector<sBufferLayoutElement>::iterator end() { return mElements.end(); }

        std::vector<sBufferLayoutElement>::const_iterator begin() const { return mElements.begin(); }
        std::vector<sBufferLayoutElement>::const_iterator end() const { return mElements.end(); }

        void Compile( uint32_t aBinding, VkVertexInputBindingDescription &o_Binding,
                      std::vector<VkVertexInputAttributeDescription> &o_Attributes, bool aInstanced );

      private:
        void CalculateOffsetsAndStride();

      private:
        std::vector<sBufferLayoutElement> mElements;
        uint32_t                          mStride = 0;
    };

    enum class eDepthCompareOperation : uint32_t
    {
        NEVER            = VK_COMPARE_OP_NEVER,
        LESS             = VK_COMPARE_OP_LESS,
        EQUAL            = VK_COMPARE_OP_EQUAL,
        LESS_OR_EQUAL    = VK_COMPARE_OP_LESS_OR_EQUAL,
        GREATER          = VK_COMPARE_OP_GREATER,
        NOT_EQUAL        = VK_COMPARE_OP_NOT_EQUAL,
        GREATER_OR_EQUAL = VK_COMPARE_OP_GREATER_OR_EQUAL,
        ALWAYS           = VK_COMPARE_OP_ALWAYS
    };

    enum class eBlendOperation : uint32_t
    {
        ADD              = VK_BLEND_OP_ADD,
        SUBTRACT         = VK_BLEND_OP_SUBTRACT,
        REVERSE_SUBTRACT = VK_BLEND_OP_REVERSE_SUBTRACT,
        MIN              = VK_BLEND_OP_MIN,
        MAX              = VK_BLEND_OP_MAX
    };
    enum class eBlendFactor : uint32_t
    {
        ZERO                     = VK_BLEND_FACTOR_ZERO,
        ONE                      = VK_BLEND_FACTOR_ONE,
        SRC_COLOR                = VK_BLEND_FACTOR_SRC_COLOR,
        ONE_MINUS_SRC_COLOR      = VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR,
        DST_COLOR                = VK_BLEND_FACTOR_DST_COLOR,
        ONE_MINUS_DST_COLOR      = VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR,
        SRC_ALPHA                = VK_BLEND_FACTOR_SRC_ALPHA,
        ONE_MINUS_SRC_ALPHA      = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
        DST_ALPHA                = VK_BLEND_FACTOR_DST_ALPHA,
        ONE_MINUS_DST_ALPHA      = VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA,
        CONSTANT_COLOR           = VK_BLEND_FACTOR_CONSTANT_COLOR,
        ONE_MINUS_CONSTANT_COLOR = VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR,
        CONSTANT_ALPHA           = VK_BLEND_FACTOR_CONSTANT_ALPHA,
        ONE_MINUS_CONSTANT_ALPHA = VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA,
        SRC_ALPHA_SATURATE       = VK_BLEND_FACTOR_SRC_ALPHA_SATURATE,
        SRC1_COLOR               = VK_BLEND_FACTOR_SRC1_COLOR,
        ONE_MINUS_SRC1_COLOR     = VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR,
        SRC1_ALPHA               = VK_BLEND_FACTOR_SRC1_ALPHA,
        ONE_MINUS_SRC1_ALPHA     = VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA
    };

    struct sBlending
    {
        bool mEnable = false;

        eBlendFactor    mSourceColorFactor   = eBlendFactor::ZERO;
        eBlendFactor    mDestColorFactor     = eBlendFactor::ZERO;
        eBlendOperation mColorBlendOperation = eBlendOperation::MAX;

        eBlendFactor    mSourceAlphaFactor   = eBlendFactor::ZERO;
        eBlendFactor    mDestAlphaFactor     = eBlendFactor::ZERO;
        eBlendOperation mAlphaBlendOperation = eBlendOperation::MAX;
    };

    struct sDepthTesting
    {
        bool                   mDepthWriteEnable = false;
        bool                   mDepthTestEnable  = false;
        eDepthCompareOperation mDepthComparison  = eDepthCompareOperation::ALWAYS;
    };

    struct sVkPipelineObject
    {
        VkPipeline mVkObject = VK_NULL_HANDLE;

        sVkPipelineObject()                      = default;
        sVkPipelineObject( sVkPipelineObject & ) = default;
        sVkPipelineObject( Ref<VkContext> aContext, uint8_t aSampleCount, sBufferLayout aVertexBufferLayout,
                           sBufferLayout aInstanceBufferLayout, ePrimitiveTopology aTopology, eFaceCulling aCullMode, float aLineWidth,
                           sDepthTesting aDepthTest, sBlending aBlending, std::vector<sShader> aShaderStages,
                           Ref<sVkPipelineLayoutObject> aPipelineLayout, Ref<sVkRenderPassObject> aRenderPass );

        ~sVkPipelineObject();

      private:
        Ref<VkContext> mContext = nullptr;
    };

} // namespace LTSE::Graphics::Internal