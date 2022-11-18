#pragma once

#include <filesystem>

#include "Core/Math/Types.h"

#include "Core/GraphicContext//GraphicContext.h"
#include "Core/Vulkan/VkAbstractRenderPass.h"

#include "Core/GraphicContext//DescriptorSet.h"
#include "Core/GraphicContext//GraphicsPipeline.h"

namespace LTSE::Core
{

    using namespace math;
    using namespace LTSE::Graphics;
    using namespace LTSE::Graphics::Internal;
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

        SceneRenderPipeline( GraphicContext &a_GraphicContext )
            : mGraphicContext{ a_GraphicContext }
        {
        }

        void Initialize( SceneRenderPipelineCreateInfo &aCreateInfo )
        {
            Spec = aCreateInfo;

            std::string                 lVertexShaderFiles = GetResourcePath( aCreateInfo.VertexShader ).string();
            Ref<Internal::ShaderModule> lVertexShaderModule =
                New<Internal::ShaderModule>( mGraphicContext.mContext, lVertexShaderFiles, Internal::eShaderStageTypeFlags::VERTEX );

            std::string                 l_FragmentShaderFiles  = GetResourcePath( aCreateInfo.FragmentShader ).string();
            Ref<Internal::ShaderModule> l_FragmentShaderModule = New<Internal::ShaderModule>(
                mGraphicContext.mContext, l_FragmentShaderFiles, Internal::eShaderStageTypeFlags::FRAGMENT );

            GraphicsPipelineCreateInfo lPipelineCreateInfo{};
            lPipelineCreateInfo.mShaderStages        = { { lVertexShaderModule, "main" }, { l_FragmentShaderModule, "main" } };
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
            lPipelineCreateInfo.DepthWriteEnable = Spec.DepthTest;
            lPipelineCreateInfo.DepthTestEnable  = Spec.DepthWrite;
            lPipelineCreateInfo.DepthComparison  = eDepthCompareOperation::LESS_OR_EQUAL;
            lPipelineCreateInfo.PushConstants    = GetPushConstantLayout();
            lPipelineCreateInfo.SetLayouts       = GetDescriptorSetLayout();

            Pipeline = New<GraphicsPipeline>( mGraphicContext, lPipelineCreateInfo );
        }

        virtual std::vector<Ref<DescriptorSetLayout>> GetDescriptorSetLayout() = 0;
        virtual std::vector<sPushConstantRange>       GetPushConstantLayout()  = 0;

      protected:
        GraphicContext mGraphicContext;
    };

} // namespace LTSE::Core
