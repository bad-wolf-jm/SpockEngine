#pragma once

#include <filesystem>

#include "Core/Math/Types.h"

#include "Graphics/Vulkan/VkAbstractRenderPass.h"
#include "Graphics/Vulkan/VkGraphicContext.h"

#include "Graphics/Vulkan/DescriptorSet.h"
#include "Graphics/Vulkan/GraphicsPipeline.h"

namespace SE::Core
{

    using namespace math;
    using namespace SE::Graphics;
    namespace fs = std::filesystem;

    struct SceneRenderPipelineCreateInfo
    {
        bool                             Opaque               = false;
        bool                             IsTwoSided           = false;
        bool                             DepthTest            = true;
        bool                             DepthWrite           = true;
        float                            LineWidth            = 1.0f;
        ePrimitiveTopology               Topology             = ePrimitiveTopology::TRIANGLES;
        fs::path                         VertexShader         = "";
        fs::path                         FragmentShader       = "";
        sBufferLayout                    InputBufferLayout    = {};
        sBufferLayout                    InstanceBufferLayout = {};
        Ref<sVkAbstractRenderPassObject> RenderPass           = nullptr;
    };

    template <typename _VertexType>
    class SceneRenderPipeline
    {
      public:
        SceneRenderPipelineCreateInfo Spec;
        Ref<GraphicsPipeline>         Pipeline = nullptr;

        SceneRenderPipeline()  = default;
        ~SceneRenderPipeline() = default;

        SceneRenderPipeline( Ref<VkGraphicContext> a_GraphicContext )
            : mGraphicContext{ a_GraphicContext }
        {
        }

        void Initialize( SceneRenderPipelineCreateInfo &aCreateInfo )
        {
            Spec = aCreateInfo;

            std::string       lVertexShaderFiles = GetResourcePath( aCreateInfo.VertexShader ).string();
            Ref<ShaderModule> lVertexShaderModule =
                New<ShaderModule>( mGraphicContext, lVertexShaderFiles, eShaderStageTypeFlags::VERTEX );

            Ref<ShaderModule> lFragmentShaderModule = nullptr;
            if( !aCreateInfo.FragmentShader.empty() )
            {
                std::string lFragmentShaderFiles = GetResourcePath( aCreateInfo.FragmentShader ).string();
                lFragmentShaderModule = New<ShaderModule>( mGraphicContext, lFragmentShaderFiles, eShaderStageTypeFlags::FRAGMENT );
            }
            GraphicsPipelineCreateInfo lPipelineCreateInfo{};
            lPipelineCreateInfo.mShaderStages = { { lVertexShaderModule, "main" } };
            if( lFragmentShaderModule )
                lPipelineCreateInfo.mShaderStages.push_back( sShader{ lFragmentShaderModule, "main" } );
            lPipelineCreateInfo.InputBufferLayout    = _VertexType::GetDefaultLayout();
            lPipelineCreateInfo.InstanceBufferLayout = aCreateInfo.InstanceBufferLayout;
            lPipelineCreateInfo.Topology             = Spec.Topology;
            lPipelineCreateInfo.Opaque               = Spec.Opaque;

            if( Spec.IsTwoSided )
                lPipelineCreateInfo.Culling = eFaceCulling::NONE;
            else
                lPipelineCreateInfo.Culling = eFaceCulling::BACK;

            lPipelineCreateInfo.SampleCount      = Spec.RenderPass->mSampleCount;
            lPipelineCreateInfo.LineWidth        = Spec.LineWidth;
            lPipelineCreateInfo.RenderPass       = Spec.RenderPass;
            lPipelineCreateInfo.DepthWriteEnable = Spec.DepthWrite;
            lPipelineCreateInfo.DepthTestEnable  = Spec.DepthTest;
            lPipelineCreateInfo.DepthComparison  = eDepthCompareOperation::LESS_OR_EQUAL;
            lPipelineCreateInfo.PushConstants    = GetPushConstantLayout();
            lPipelineCreateInfo.SetLayouts       = GetDescriptorSetLayout();

            Pipeline = New<GraphicsPipeline>( mGraphicContext, lPipelineCreateInfo );
        }

        virtual std::vector<Ref<DescriptorSetLayout>> GetDescriptorSetLayout() = 0;
        virtual std::vector<sPushConstantRange>       GetPushConstantLayout()  = 0;

      protected:
        Ref<VkGraphicContext> mGraphicContext;
    };

} // namespace SE::Core
