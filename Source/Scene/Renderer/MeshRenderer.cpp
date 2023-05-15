#include "MeshRenderer.h"

#include <chrono>

// #include "Graphics/Vulkan/VkPipeline.h"
#include "Scene/Primitives/Primitives.h"
#include "Scene/VertexData.h"

#include "Core/Logging.h"
#include "Core/Resource.h"

namespace SE::Core
{

    using namespace math;

    Ref<DescriptorSetLayout> MeshRenderer::GetCameraSetLayout( Ref<IGraphicContext> aGraphicContext )
    {
        DescriptorSetLayoutCreateInfo lCameraBindLayout{};
        lCameraBindLayout.Bindings = {
            DescriptorBindingInfo{
                0, eDescriptorType::UNIFORM_BUFFER, { eShaderStageTypeFlags::VERTEX, eShaderStageTypeFlags::FRAGMENT } },
            DescriptorBindingInfo{ 1, eDescriptorType::UNIFORM_BUFFER, { eShaderStageTypeFlags::FRAGMENT } } };

        return New<DescriptorSetLayout>( aGraphicContext, lCameraBindLayout );
    }

    Ref<DescriptorSetLayout> MeshRenderer::GetTextureSetLayout( Ref<IGraphicContext> aGraphicContext )
    {
        DescriptorSetLayoutCreateInfo lTextureBindLayout{};
        lTextureBindLayout.Bindings = {
            DescriptorBindingInfo{ 0, eDescriptorType::STORAGE_BUFFER, { eShaderStageTypeFlags::FRAGMENT } },
            DescriptorBindingInfo{ 1, eDescriptorType::COMBINED_IMAGE_SAMPLER, { eShaderStageTypeFlags::FRAGMENT } } };

        return New<DescriptorSetLayout>( aGraphicContext, lTextureBindLayout, true );
    }

    Ref<DescriptorSetLayout> MeshRenderer::GetNodeSetLayout( Ref<IGraphicContext> aGraphicContext )
    {
        DescriptorSetLayoutCreateInfo lNodeBindLayout{};
        lNodeBindLayout.Bindings = { DescriptorBindingInfo{ 0, eDescriptorType::UNIFORM_BUFFER, { eShaderStageTypeFlags::VERTEX } } };
        return New<DescriptorSetLayout>( aGraphicContext, lNodeBindLayout );
    }

    std::vector<Ref<DescriptorSetLayout>> MeshRenderer::GetDescriptorSetLayout()
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