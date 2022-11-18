#include "MeshRenderer.h"

#include <chrono>

#include "Core/Vulkan/VkPipeline.h"
#include "Scene/Primitives/Primitives.h"
#include "Scene/VertexData.h"

#include "Core/Logging.h"
#include "Core/Resource.h"

namespace SE::Core
{

    using namespace math;

    Ref<DescriptorSetLayout> MeshRenderer::GetCameraSetLayout( GraphicContext &aGraphicContext )
    {
        DescriptorSetLayoutCreateInfo l_CameraBindLayout{};
        l_CameraBindLayout.Bindings = { DescriptorBindingInfo{ 0, Internal::eDescriptorType::UNIFORM_BUFFER,
                                            { Internal::eShaderStageTypeFlags::VERTEX, Internal::eShaderStageTypeFlags::FRAGMENT } },
            DescriptorBindingInfo{ 1, Internal::eDescriptorType::UNIFORM_BUFFER, { Internal::eShaderStageTypeFlags::FRAGMENT } } };

        return New<DescriptorSetLayout>( aGraphicContext, l_CameraBindLayout );
    }

    Ref<DescriptorSetLayout> MeshRenderer::GetTextureSetLayout( GraphicContext &aGraphicContext )
    {
        DescriptorSetLayoutCreateInfo lTextureBindLayout{};
        lTextureBindLayout.Bindings = {
            DescriptorBindingInfo{ 0, Internal::eDescriptorType::STORAGE_BUFFER, { Internal::eShaderStageTypeFlags::FRAGMENT } },
            DescriptorBindingInfo{
                1, Internal::eDescriptorType::COMBINED_IMAGE_SAMPLER, { Internal::eShaderStageTypeFlags::FRAGMENT } } };

        return New<DescriptorSetLayout>( aGraphicContext, lTextureBindLayout, true );
    }

    Ref<DescriptorSetLayout> MeshRenderer::GetNodeSetLayout( GraphicContext &aGraphicContext )
    {
        DescriptorSetLayoutCreateInfo l_NodeBindLayout{};
        l_NodeBindLayout.Bindings = {
            DescriptorBindingInfo{ 0, Internal::eDescriptorType::UNIFORM_BUFFER, { Internal::eShaderStageTypeFlags::VERTEX } } };
        return New<DescriptorSetLayout>( aGraphicContext, l_NodeBindLayout );
    }

    std::vector<Ref<DescriptorSetLayout>> MeshRenderer::GetDescriptorSetLayout()
    {
        return { CameraSetLayout, TextureSetLayout, NodeSetLayout };
    }

    std::vector<sPushConstantRange> MeshRenderer::GetPushConstantLayout()
    {
        return { { { Internal::eShaderStageTypeFlags::FRAGMENT }, 0, sizeof( MaterialPushConstants ) } };
    };

    MeshRenderer::MeshRenderer( GraphicContext &aGraphicContext, MeshRendererCreateInfo const &aCreateInfo )
        : SceneRenderPipeline<VertexData>( aGraphicContext )
        , Spec{ aCreateInfo }
    {

        SceneRenderPipelineCreateInfo lCreateInfo{};
        lCreateInfo.Opaque         = aCreateInfo.Opaque;
        lCreateInfo.LineWidth      = aCreateInfo.LineWidth;
        lCreateInfo.VertexShader   = aCreateInfo.VertexShader;
        lCreateInfo.FragmentShader = aCreateInfo.FragmentShader;
        lCreateInfo.RenderPass     = aCreateInfo.RenderPass;

        CameraSetLayout  = GetCameraSetLayout( mGraphicContext );
        TextureSetLayout = GetTextureSetLayout( mGraphicContext );
        NodeSetLayout    = GetNodeSetLayout( mGraphicContext );

        Initialize( lCreateInfo );
    }

} // namespace SE::Core