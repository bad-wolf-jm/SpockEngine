#include "MeshRenderer.h"

#include <chrono>

#include "Scene/Primitives/Primitives.h"
#include "Scene/VertexData.h"

#include "Core/Logging.h"
#include "Core/Resource.h"

namespace SE::Core
{

    using namespace math;

    Ref<IDescriptorSetLayout> MeshRenderer::GetCameraSetLayout( Ref<IGraphicContext> aGraphicContext )
    {
        auto lNewLayout = CreateDescriptorSetLayout( aGraphicContext );
        lNewLayout.AddBinding( 0, eDescriptorType::UNIFORM_BUFFER, { eShaderStageTypeFlags::FRAGMENT } );
        lNewLayout.AddBinding( 1, eDescriptorType::UNIFORM_BUFFER, { eShaderStageTypeFlags::FRAGMENT } );
        lNewLayout.Build();

        return lNewLayout;
    }

    Ref<IDescriptorSetLayout> MeshRenderer::GetTextureSetLayout( Ref<IGraphicContext> aGraphicContext )
    {
        auto lNewLayout = CreateDescriptorSetLayout( aGraphicContext );
        lNewLayout.AddBinding( 0, eDescriptorType::STORAGE_BUFFER, { eShaderStageTypeFlags::FRAGMENT } );
        lNewLayout.AddBinding( 1, eDescriptorType::COMBINED_IMAGE_SAMPLER, { eShaderStageTypeFlags::FRAGMENT } );
        lNewLayout.Build();

        return lNewLayout;
    }

    Ref<IDescriptorSetLayout> MeshRenderer::GetNodeSetLayout( Ref<IGraphicContext> aGraphicContext )
    {
        auto lNewLayout = CreateDescriptorSetLayout( aGraphicContext );
        lNewLayout.AddBinding( 0, eDescriptorType::UNIFORM_BUFFER, { eShaderStageTypeFlags::VERTEX } );
        lNewLayout.Build();

        return lNewLayout;
    }

    std::vector<Ref<IDescriptorSetLayout>> MeshRenderer::GetDescriptorSetLayout()
    {
        return { CameraSetLayout, TextureSetLayout, NodeSetLayout };
    }

    std::vector<sPushConstantRange> MeshRenderer::GetPushConstantLayout()
    {
        return { { { eShaderStageTypeFlags::FRAGMENT }, 0, sizeof( MaterialPushConstants ) } };
    };

    MeshRenderer::MeshRenderer( Ref<IGraphicContext> aGraphicContext, MeshRendererCreateInfo const &aCreateInfo )
        : SceneRenderPipeline<VertexData>( aGraphicContext )
        , Spec{ aCreateInfo }
    {

        SceneRenderPipelineCreateInfo lCreateInfo{};
        lCreateInfo.Opaque         = Spec.Opaque;
        lCreateInfo.LineWidth      = Spec.LineWidth;
        lCreateInfo.VertexShader   = Spec.VertexShader;
        lCreateInfo.FragmentShader = Spec.FragmentShader;
        lCreateInfo.RenderPass     = Spec.RenderPass;

        CameraSetLayout  = GetCameraSetLayout( mGraphicContext );
        TextureSetLayout = GetTextureSetLayout( mGraphicContext );
        NodeSetLayout    = GetNodeSetLayout( mGraphicContext );

        Initialize( lCreateInfo );
    }

} // namespace SE::Core