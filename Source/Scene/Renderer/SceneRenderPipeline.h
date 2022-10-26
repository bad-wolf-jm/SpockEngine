#pragma once

#include <filesystem>

#include "Core/Math/Types.h"

#include "Core/GraphicContext//GraphicContext.h"
#include "Core/Vulkan/VkRenderPass.h"


#include "Core/GraphicContext//DescriptorSet.h"
#include "Core/GraphicContext//GraphicsPipeline.h"
#include "Core/GraphicContext//RenderContext.h"

namespace LTSE::Core
{

    using namespace math;
    using namespace LTSE::Graphics;
    namespace fs = std::filesystem;

    struct SceneRenderPipelineCreateInfo
    {
        bool                                               Opaque               = false;
        bool                                               IsTwoSided           = false;
        float                                              LineWidth            = 1.0f;
        ePrimitiveTopology                                 Topology             = ePrimitiveTopology::TRIANGLES;
        fs::path                                           VertexShader         = "";
        fs::path                                           FragmentShader       = "";
        sBufferLayout                                      InputBufferLayout    = {};
        sBufferLayout                                      InstanceBufferLayout = {};
        Ref<LTSE::Graphics::Internal::sVkRenderPassObject> RenderPass           = nullptr;
    };

    template <typename _VertexType>
    class SceneRenderPipeline
    {
      public:
        SceneRenderPipelineCreateInfo Spec;
        Ref<GraphicsPipeline>         Pipeline = nullptr;

        SceneRenderPipeline()  = default;
        ~SceneRenderPipeline() = default;

        SceneRenderPipeline( GraphicContext &a_GraphicContext )
            : mGraphicContext{ a_GraphicContext }
        {
        }

        void Initialize( SceneRenderPipelineCreateInfo &a_CreateInfo )
        {
            Spec = a_CreateInfo;

            std::string                 l_VertexShaderFiles = GetResourcePath( a_CreateInfo.VertexShader ).string();
            Ref<Internal::ShaderModule> l_VertexShaderModule =
                New<Internal::ShaderModule>( mGraphicContext.mContext, l_VertexShaderFiles, Internal::eShaderStageTypeFlags::VERTEX );

            std::string                 l_FragmentShaderFiles  = GetResourcePath( a_CreateInfo.FragmentShader ).string();
            Ref<Internal::ShaderModule> l_FragmentShaderModule = New<Internal::ShaderModule>(
                mGraphicContext.mContext, l_FragmentShaderFiles, Internal::eShaderStageTypeFlags::FRAGMENT );

            GraphicsPipelineCreateInfo l_PipelineCreateInfo{};
            l_PipelineCreateInfo.mShaderStages        = { { l_VertexShaderModule, "main" }, { l_FragmentShaderModule, "main" } };
            l_PipelineCreateInfo.InputBufferLayout    = _VertexType::GetDefaultLayout();
            l_PipelineCreateInfo.InstanceBufferLayout = a_CreateInfo.InstanceBufferLayout;
            l_PipelineCreateInfo.Topology             = Spec.Topology;
            l_PipelineCreateInfo.Opaque               = Spec.Opaque;
            if( Spec.IsTwoSided )
            {
                l_PipelineCreateInfo.Culling = eFaceCulling::NONE;
            }
            else
            {
                l_PipelineCreateInfo.Culling = eFaceCulling::BACK;
            }

            l_PipelineCreateInfo.SampleCount      = Spec.RenderPass->mSampleCount;
            l_PipelineCreateInfo.LineWidth        = Spec.LineWidth;
            l_PipelineCreateInfo.RenderPass       = Spec.RenderPass;
            l_PipelineCreateInfo.DepthWriteEnable = true;
            l_PipelineCreateInfo.DepthTestEnable  = true;
            l_PipelineCreateInfo.DepthComparison  = eDepthCompareOperation::LESS_OR_EQUAL;
            l_PipelineCreateInfo.PushConstants    = GetPushConstantLayout();
            l_PipelineCreateInfo.SetLayouts       = GetDescriptorSetLayout();

            Pipeline = New<GraphicsPipeline>( mGraphicContext, l_PipelineCreateInfo );
        }

        virtual std::vector<Ref<DescriptorSetLayout>> GetDescriptorSetLayout() = 0;
        virtual std::vector<sPushConstantRange>       GetPushConstantLayout()  = 0;

      protected:
        GraphicContext mGraphicContext;
    };

} // namespace LTSE::Core
