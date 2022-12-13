#include "VkPipeline.h"

#include <fstream>
#include <set>
#include <stdexcept>
#include <unordered_set>

#include "Core/Memory.h"
#include "Graphics/Vulkan/VkCoreMacros.h"

namespace SE::Graphics::Internal
{

    sVkShaderModuleObject::sVkShaderModuleObject( Ref<VkGraphicContext> aContext, std::vector<uint32_t> aByteCode )
        : mContext{ aContext }
    {
        mVkObject = mContext->CreateShaderModule( aByteCode );
    }

    sVkShaderModuleObject::~sVkShaderModuleObject() { mContext->DestroyShaderModule( mVkObject ); }

    static std::vector<char> ReadFile( const std::string &filename )
    {
        std::ifstream lFileObject( filename, std::ios::ate | std::ios::binary );

        if( !lFileObject.is_open() ) throw std::runtime_error( "failed to open file!" );

        size_t            lFileSize = (size_t)lFileObject.tellg();
        std::vector<char> lBuffer( lFileSize );

        lFileObject.seekg( 0 );
        lFileObject.read( lBuffer.data(), lFileSize );
        lFileObject.close();

        return lBuffer;
    }

    static std::vector<uint32_t> LoadShaderModuleBytecode( std::string aFilePaths )
    {
        auto lCode     = ReadFile( aFilePaths );
        auto lBytecode = std::vector<uint32_t>( lCode.size() / 4 );
        std::memcpy( lBytecode.data(), lCode.data(), lCode.size() );
        return lBytecode;
    }

    static bool IsSPIRV( std::string aFileName ) { return ( aFileName.substr( aFileName.find_last_of( "." ) + 1 ) == "spv" ); }

    static std::vector<uint32_t> CompileShaderSources( std::string FilePaths, eShaderStageTypeFlags aShaderType )
    {
        if( IsSPIRV( FilePaths ) ) return LoadShaderModuleBytecode( FilePaths );
        return std::vector<uint32_t>( 0 );
    }

    ShaderModule::ShaderModule( Ref<VkGraphicContext> mContext, std::string FilePaths, eShaderStageTypeFlags aShaderType )
        : Type{ aShaderType }
    {
        std::vector<uint32_t> lByteCode = CompileShaderSources( FilePaths, aShaderType );
        mShaderModuleObject             = New<sVkShaderModuleObject>( mContext, lByteCode );
    }

    VkPipelineShaderStageCreateInfo ShaderModule::GetShaderStage()
    {
        VkPipelineShaderStageCreateInfo shaderStages;
        shaderStages.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;

        switch( Type )
        {
        case eShaderStageTypeFlags::VERTEX:
            shaderStages.stage = VK_SHADER_STAGE_VERTEX_BIT;
            break;
        case eShaderStageTypeFlags::GEOMETRY:
            shaderStages.stage = VK_SHADER_STAGE_GEOMETRY_BIT;
            break;
        case eShaderStageTypeFlags::FRAGMENT:
            shaderStages.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
            break;
        case eShaderStageTypeFlags::TESSELATION_CONTROL:
            shaderStages.stage = VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;
            break;
        case eShaderStageTypeFlags::TESSELATION_EVALUATION:
            shaderStages.stage = VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
            break;
        case eShaderStageTypeFlags::COMPUTE:
            shaderStages.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            break;
        case eShaderStageTypeFlags::DEFAULT:
        default:
            throw std::runtime_error( "Unknown shader type" );
        }

        shaderStages.module              = mShaderModuleObject->mVkObject;
        shaderStages.pName               = "main";
        shaderStages.flags               = 0;
        shaderStages.pNext               = nullptr;
        shaderStages.pSpecializationInfo = nullptr;
        return shaderStages;
    }

    sVkDescriptorSetLayoutObject::sVkDescriptorSetLayoutObject(
        Ref<VkGraphicContext> aContext, std::vector<VkDescriptorSetLayoutBinding> aBindings, bool aUnbounded )
        : mContext{ aContext }
    {
        mVkObject = mContext->CreateDescriptorSetLayout( aBindings, aUnbounded );
    }

    sVkDescriptorSetLayoutObject::~sVkDescriptorSetLayoutObject() { mContext->DestroyDescriptorSetLayout( mVkObject ); }

    sVkDescriptorSetObject::sVkDescriptorSetObject(
        Ref<VkGraphicContext> aContext, VkDescriptorSet aDescriporSet )
        : mContext{ aContext }
        , mVkObject{ aDescriporSet }

    {
    }

    sVkDescriptorSetObject::~sVkDescriptorSetObject() { mContext->FreeDescriptorSet( &mVkObject ); }

    void sVkDescriptorSetObject::Write( sBufferBindInfo aBuffers )
    {
        VkWriteDescriptorSet   lWriteDSOps{};
        VkDescriptorBufferInfo lWriteBufferInfo{};

        lWriteBufferInfo.buffer = aBuffers.mBuffer;
        lWriteBufferInfo.offset = aBuffers.mOffset;
        lWriteBufferInfo.range  = aBuffers.mSize;

        lWriteDSOps.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        lWriteDSOps.pNext           = nullptr;
        lWriteDSOps.dstBinding      = aBuffers.mBinding;
        lWriteDSOps.dstSet          = mVkObject;
        lWriteDSOps.descriptorCount = 1;

        if( aBuffers.mType == eBufferBindType::STORAGE_BUFFER )
        {
            lWriteDSOps.descriptorType = aBuffers.mDynamicOffset ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC
                                                                 : static_cast<VkDescriptorType>( eDescriptorType::STORAGE_BUFFER );
        }
        else if( aBuffers.mType == eBufferBindType::UNIFORM_BUFFER )
        {
            lWriteDSOps.descriptorType = aBuffers.mDynamicOffset ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC
                                                                 : static_cast<VkDescriptorType>( eDescriptorType::UNIFORM_BUFFER );
        }
        else
        {
            throw std::runtime_error( "Bad buffer specification!" );
        }
        lWriteDSOps.pBufferInfo = &( lWriteBufferInfo );

        mContext->UpdateDescriptorSets( lWriteDSOps );
    }

    void sVkDescriptorSetObject::Write( sImageBindInfo aImages )
    {
        VkWriteDescriptorSet               lWriteDSOps;
        std::vector<VkDescriptorImageInfo> lWriteBufferInfo;

        for( uint32_t j = 0; j < aImages.mSampler.size(); j++ )
        {
            VkDescriptorImageInfo lImageInfo{};
            lImageInfo.sampler     = aImages.mSampler[j];
            lImageInfo.imageView   = aImages.mImageView[j];
            lImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            lWriteBufferInfo.push_back( lImageInfo );
        }

        lWriteDSOps.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        lWriteDSOps.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        lWriteDSOps.pNext           = nullptr;
        lWriteDSOps.dstBinding      = aImages.mBinding;
        lWriteDSOps.dstSet          = mVkObject;
        lWriteDSOps.dstArrayElement = 0;
        lWriteDSOps.descriptorCount = lWriteBufferInfo.size();
        lWriteDSOps.pImageInfo      = lWriteBufferInfo.data();

        mContext->UpdateDescriptorSets( lWriteDSOps );
    }

    sVkDescriptorPoolObject::sVkDescriptorPoolObject(
        Ref<VkGraphicContext> aContext, uint32_t aDescriptorSetCount, std::vector<VkDescriptorPoolSize> aPoolSizes )
        : mContext{ aContext }
    {
    }

    sVkDescriptorPoolObject::~sVkDescriptorPoolObject() { mContext->DestroyDescriptorPool( mVkObject ); }

    Ref<sVkDescriptorSetObject> sVkDescriptorPoolObject::Allocate(
        Ref<sVkDescriptorSetLayoutObject> aLayout, uint32_t aDescriptorCount )
    {
        return SE::Core::New<sVkDescriptorSetObject>(
            mContext, mContext->AllocateDescriptorSet( aLayout->mVkObject, aDescriptorCount ) );
    }

    sVkPipelineLayoutObject::sVkPipelineLayoutObject( Ref<VkGraphicContext> aContext,
        std::vector<Ref<sVkDescriptorSetLayoutObject>> aDescriptorSetLayout, std::vector<sPushConstantRange> aPushConstantRanges )
        : mContext{ aContext }
    {

        std::vector<VkDescriptorSetLayout> lDescriptorSetLayouts( aDescriptorSetLayout.size() );
        for( uint32_t i = 0; i < aDescriptorSetLayout.size(); i++ ) lDescriptorSetLayouts[i] = aDescriptorSetLayout[i]->mVkObject;

        std::vector<VkPushConstantRange> lPushConstantRanges( aPushConstantRanges.size() );
        for( uint32_t i = 0; i < aPushConstantRanges.size(); i++ )
        {
            VkPushConstantRange lPushConstant;
            lPushConstant.offset     = aPushConstantRanges[i].mOffset;
            lPushConstant.size       = aPushConstantRanges[i].mSize;
            lPushConstant.stageFlags = (VkShaderStageFlags)aPushConstantRanges[i].mShaderStages;
            lPushConstantRanges[i]   = lPushConstant;
        }

        mVkObject = mContext->CreatePipelineLayout( lDescriptorSetLayouts, lPushConstantRanges );
    }

    sVkPipelineLayoutObject::~sVkPipelineLayoutObject() { mContext->DestroyPipelineLayout( mVkObject ); }

    static uint32_t BufferDataTypeSize( eBufferDataType aType )
    {
        // clang-format off
        switch( aType )
        {
        case eBufferDataType::UINT8:  return 1 * sizeof( uint8_t  );
        case eBufferDataType::UINT16: return 1 * sizeof( uint16_t );
        case eBufferDataType::UINT32: return 1 * sizeof( uint32_t );
        case eBufferDataType::INT8:   return 1 * sizeof( int8_t   );
        case eBufferDataType::INT16:  return 1 * sizeof( int16_t  );
        case eBufferDataType::INT32:  return 1 * sizeof( int32_t  );
        case eBufferDataType::FLOAT:  return 1 * sizeof( float    );
        case eBufferDataType::COLOR:  return 4 * sizeof( uint8_t  );
        case eBufferDataType::VEC2:   return 2 * sizeof( float    );
        case eBufferDataType::VEC3:   return 3 * sizeof( float    );
        case eBufferDataType::VEC4:   return 4 * sizeof( float    );
        case eBufferDataType::IVEC2:  return 2 * sizeof( int32_t  );
        case eBufferDataType::IVEC3:  return 3 * sizeof( int32_t  );
        case eBufferDataType::IVEC4:  return 4 * sizeof( int32_t  );
        case eBufferDataType::UVEC2:  return 2 * sizeof( uint32_t );
        case eBufferDataType::UVEC3:  return 3 * sizeof( uint32_t );
        case eBufferDataType::UVEC4:  return 4 * sizeof( uint32_t );
        }
        // clang-format on
        return 0;
    };

    sBufferLayoutElement::sBufferLayoutElement(
        const std::string &aName, eBufferDataType aType, uint32_t aBinding, uint32_t aLocation )
        : mName( aName )
        , mType( aType )
        , mBinding( aBinding )
        , mLocation( aLocation )
        , mSize( BufferDataTypeSize( aType ) )
        , mOffset( 0 )
    {
    }

    sBufferLayout::sBufferLayout( std::initializer_list<sBufferLayoutElement> aElements )
        : mElements( aElements )
    {
        CalculateOffsetsAndStride();
    }

    void sBufferLayout::CalculateOffsetsAndStride()
    {
        mStride = 0;

        size_t offset = 0;
        for( auto &lElement : mElements )
        {
            lElement.mOffset = offset;
            offset += lElement.mSize;
            mStride += lElement.mSize;
        }
    }

    void sBufferLayout::Compile( uint32_t aBinding, VkVertexInputBindingDescription &aBindingDesc,
        std::vector<VkVertexInputAttributeDescription> &aAttributes, bool aInstanced )
    {
        aBindingDesc.binding   = aBinding;
        aBindingDesc.stride    = mStride;
        aBindingDesc.inputRate = aInstanced ? VK_VERTEX_INPUT_RATE_INSTANCE : VK_VERTEX_INPUT_RATE_VERTEX;

        aAttributes.resize( mElements.size() );
        for( uint32_t i = 0; i < mElements.size(); i++ )
        {
            VkVertexInputAttributeDescription positionAttribute{};
            positionAttribute.binding  = aBinding;
            positionAttribute.location = mElements[i].mLocation;
            positionAttribute.format   = (VkFormat)mElements[i].mType;
            positionAttribute.offset   = mElements[i].mOffset;
            aAttributes[i]             = positionAttribute;
        }
    }

    sVkPipelineObject::sVkPipelineObject( Ref<VkGraphicContext> aContext, uint8_t aSampleCount, sBufferLayout aVertexBufferLayout,
        sBufferLayout aInstanceBufferLayout, ePrimitiveTopology aTopology, eFaceCulling aCullMode, float aLineWidth,
        sDepthTesting aDepthTest, sBlending aBlending, std::vector<sShader> aShaderStages,
        Ref<sVkPipelineLayoutObject> aPipelineLayout, Ref<sVkAbstractRenderPassObject> aRenderPass )
        : mContext{ aContext }
    {

        VkGraphicsPipelineCreateInfo aCreateInfo{};
        aCreateInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        aCreateInfo.pNext = nullptr;

        VkDynamicState lStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_LINE_WIDTH };
        VkPipelineDynamicStateCreateInfo lDynamicState{};
        lDynamicState.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        lDynamicState.dynamicStateCount = 2;
        lDynamicState.pDynamicStates    = lStates;
        aCreateInfo.pDynamicState       = &lDynamicState;

        VkPipelineVertexInputStateCreateInfo lVertexInputInfo{};
        lVertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

        VkVertexInputBindingDescription                lBindings;
        std::vector<VkVertexInputAttributeDescription> lAttributes;
        aVertexBufferLayout.Compile( 0, lBindings, lAttributes, false );

        VkVertexInputBindingDescription                lInstanceBindings;
        std::vector<VkVertexInputAttributeDescription> instance_attributes;
        aInstanceBufferLayout.Compile( 1, lInstanceBindings, instance_attributes, true );

        if( lAttributes.size() != 0 )
        {
            if( instance_attributes.size() != 0 )
            {
                VkVertexInputBindingDescription lAllBindings[2] = { lBindings, lInstanceBindings };
                lAttributes.insert( lAttributes.end(), instance_attributes.begin(), instance_attributes.end() );

                lVertexInputInfo.pVertexAttributeDescriptions    = lAttributes.data();
                lVertexInputInfo.vertexAttributeDescriptionCount = lAttributes.size();
                lVertexInputInfo.pVertexBindingDescriptions      = lAllBindings;
                lVertexInputInfo.vertexBindingDescriptionCount   = 2;
                aCreateInfo.pVertexInputState                    = &lVertexInputInfo;
            }
            else
            {
                VkVertexInputBindingDescription lAllBindings[1]  = { lBindings };
                lVertexInputInfo.pVertexAttributeDescriptions    = lAttributes.data();
                lVertexInputInfo.vertexAttributeDescriptionCount = lAttributes.size();
                lVertexInputInfo.pVertexBindingDescriptions      = lAllBindings;
                lVertexInputInfo.vertexBindingDescriptionCount   = 1;
                aCreateInfo.pVertexInputState                    = &lVertexInputInfo;
            }
        }
        else
        {
            lVertexInputInfo.pVertexAttributeDescriptions    = nullptr;
            lVertexInputInfo.vertexAttributeDescriptionCount = 0;
            lVertexInputInfo.pVertexBindingDescriptions      = nullptr;
            lVertexInputInfo.vertexBindingDescriptionCount   = 0;
            aCreateInfo.pVertexInputState                    = &lVertexInputInfo;
        }

        VkPipelineColorBlendAttachmentState lColorBlendAttachment{};
        lColorBlendAttachment.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        lColorBlendAttachment.blendEnable         = aBlending.mEnable ? VK_TRUE : VK_FALSE;
        lColorBlendAttachment.srcColorBlendFactor = static_cast<VkBlendFactor>( aBlending.mSourceColorFactor );
        lColorBlendAttachment.dstColorBlendFactor = static_cast<VkBlendFactor>( aBlending.mDestColorFactor );
        lColorBlendAttachment.colorBlendOp        = static_cast<VkBlendOp>( aBlending.mColorBlendOperation );
        lColorBlendAttachment.srcAlphaBlendFactor = static_cast<VkBlendFactor>( aBlending.mSourceAlphaFactor );
        lColorBlendAttachment.dstAlphaBlendFactor = static_cast<VkBlendFactor>( aBlending.mDestAlphaFactor );
        lColorBlendAttachment.alphaBlendOp        = static_cast<VkBlendOp>( aBlending.mAlphaBlendOperation );

        std::vector<VkPipelineColorBlendAttachmentState> lBlendAttachments(aRenderPass->GetColorAttachmentCount(), lColorBlendAttachment);

        VkPipelineColorBlendStateCreateInfo lColorBlendingInfo{};
        lColorBlendingInfo.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        lColorBlendingInfo.logicOpEnable   = VK_FALSE;
        lColorBlendingInfo.logicOp         = VK_LOGIC_OP_COPY;
        lColorBlendingInfo.attachmentCount = lBlendAttachments.size();
        lColorBlendingInfo.pAttachments    = lBlendAttachments.data();
        lColorBlendingInfo.pNext           = nullptr;
        aCreateInfo.pColorBlendState       = &lColorBlendingInfo;

        VkPipelineMultisampleStateCreateInfo lMultisamplingInfo{};
        lMultisamplingInfo.sType                 = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        lMultisamplingInfo.rasterizationSamples  = (VkSampleCountFlagBits)aSampleCount;
        lMultisamplingInfo.sampleShadingEnable   = VK_FALSE;
        lMultisamplingInfo.minSampleShading      = 1.0f;
        lMultisamplingInfo.pSampleMask           = nullptr;
        lMultisamplingInfo.alphaToCoverageEnable = VK_FALSE;
        lMultisamplingInfo.alphaToOneEnable      = VK_FALSE;
        lMultisamplingInfo.pNext                 = nullptr;
        aCreateInfo.pMultisampleState            = &lMultisamplingInfo;

        VkViewport                        lViewportInfo{};
        VkRect2D                          lScissorInfo{};
        VkPipelineViewportStateCreateInfo lViewportState{};
        lViewportState.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        lViewportState.pNext         = nullptr;
        lViewportState.viewportCount = 1;
        lViewportState.pViewports    = &lViewportInfo;
        lViewportState.scissorCount  = 1;
        lViewportState.pScissors     = &lScissorInfo;
        aCreateInfo.pViewportState   = &lViewportState;

        VkPipelineInputAssemblyStateCreateInfo lInputAssemblyInfo{};
        lInputAssemblyInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        lInputAssemblyInfo.topology               = (VkPrimitiveTopology)aTopology;
        lInputAssemblyInfo.primitiveRestartEnable = VK_FALSE;
        lInputAssemblyInfo.pNext                  = nullptr;
        aCreateInfo.pInputAssemblyState           = &lInputAssemblyInfo;

        VkPipelineRasterizationStateCreateInfo lRasterizationConfig{};
        lRasterizationConfig.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        lRasterizationConfig.depthClampEnable        = VK_FALSE;
        lRasterizationConfig.rasterizerDiscardEnable = VK_FALSE;
        lRasterizationConfig.pNext                   = nullptr;

        switch( aTopology )
        {
        case ePrimitiveTopology::TRIANGLES:
            lRasterizationConfig.polygonMode = VK_POLYGON_MODE_FILL;
            break;
        case ePrimitiveTopology::LINES:
            lRasterizationConfig.polygonMode = VK_POLYGON_MODE_LINE;
            break;
        case ePrimitiveTopology::POINTS:
        default:
            lRasterizationConfig.polygonMode = VK_POLYGON_MODE_POINT;
            break;
        }

        lRasterizationConfig.lineWidth               = aLineWidth;
        lRasterizationConfig.cullMode                = (VkCullModeFlags)aCullMode;
        lRasterizationConfig.frontFace               = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        lRasterizationConfig.depthBiasEnable         = VK_FALSE;
        lRasterizationConfig.depthBiasConstantFactor = 0.0f;
        lRasterizationConfig.depthBiasClamp          = 0.0f;
        lRasterizationConfig.depthBiasSlopeFactor    = 0.0f;
        aCreateInfo.pRasterizationState              = &lRasterizationConfig;

        VkPipelineDepthStencilStateCreateInfo lDepthStencilInfo{};
        lDepthStencilInfo.sType                 = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        lDepthStencilInfo.depthTestEnable       = aDepthTest.mDepthTestEnable ? VK_TRUE : VK_FALSE;
        lDepthStencilInfo.depthWriteEnable      = aDepthTest.mDepthWriteEnable ? VK_TRUE : VK_FALSE;
        lDepthStencilInfo.depthCompareOp        = (VkCompareOp)aDepthTest.mDepthComparison;
        lDepthStencilInfo.depthBoundsTestEnable = VK_FALSE;
        lDepthStencilInfo.minDepthBounds        = 0.0f;
        lDepthStencilInfo.maxDepthBounds        = 1.0f;
        lDepthStencilInfo.stencilTestEnable     = VK_FALSE;
        lDepthStencilInfo.pNext                 = nullptr;
        aCreateInfo.pDepthStencilState          = &lDepthStencilInfo;

        std::vector<VkPipelineShaderStageCreateInfo> lShaderStages( aShaderStages.size() );
        for( uint32_t i = 0; i < aShaderStages.size(); i++ )
        {
            lShaderStages[i].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            lShaderStages[i].stage  = (VkShaderStageFlagBits)aShaderStages[i].mShaderModule->Type;
            lShaderStages[i].module = aShaderStages[i].mShaderModule->GetVkShaderModule();
            lShaderStages[i].pName  = aShaderStages[i].mEntryPoint.c_str();
            lShaderStages[i].pNext  = nullptr;
        }
        aCreateInfo.stageCount = lShaderStages.size();
        aCreateInfo.pStages    = lShaderStages.data();

        aCreateInfo.layout             = aPipelineLayout->mVkObject;
        aCreateInfo.renderPass         = aRenderPass->mVkObject;
        aCreateInfo.subpass            = 0;
        aCreateInfo.basePipelineHandle = VK_NULL_HANDLE;

        mVkObject = mContext->CreatePipeline( aCreateInfo );
    }

    sVkPipelineObject::~sVkPipelineObject() { mContext->DestroyPipeline( mVkObject ); }

} // namespace SE::Graphics::Internal