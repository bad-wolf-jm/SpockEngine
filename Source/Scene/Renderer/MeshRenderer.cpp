#include "MeshRenderer.h"

#include <chrono>

#include "Core/Vulkan/VkPipeline.h"
#include "Scene/Primitives/Primitives.h"
#include "Scene/VertexData.h"

#include "Core/Logging.h"
#include "Core/Resource.h"

namespace LTSE::Core
{

    using namespace math;

    Ref<DescriptorSetLayout> MeshRenderer::GetCameraSetLayout( GraphicContext &a_GraphicContext )
    {
        DescriptorSetLayoutCreateInfo l_CameraBindLayout{};
        l_CameraBindLayout.Bindings = { DescriptorBindingInfo{ 0, Internal::eDescriptorType::UNIFORM_BUFFER,
                                            { Internal::eShaderStageTypeFlags::VERTEX, Internal::eShaderStageTypeFlags::FRAGMENT } },
            DescriptorBindingInfo{ 1, Internal::eDescriptorType::UNIFORM_BUFFER, { Internal::eShaderStageTypeFlags::FRAGMENT } } };

        return New<DescriptorSetLayout>( a_GraphicContext, l_CameraBindLayout );
    }

    Ref<DescriptorSetLayout> MeshRenderer::GetTextureSetLayout( GraphicContext &a_GraphicContext )
    {
        DescriptorSetLayoutCreateInfo lTextureBindLayout{};
        lTextureBindLayout.Bindings = {
            DescriptorBindingInfo{ 0, Internal::eDescriptorType::STORAGE_BUFFER, { Internal::eShaderStageTypeFlags::FRAGMENT } },
            DescriptorBindingInfo{
                1, Internal::eDescriptorType::COMBINED_IMAGE_SAMPLER, { Internal::eShaderStageTypeFlags::FRAGMENT } } };

        return New<DescriptorSetLayout>( a_GraphicContext, lTextureBindLayout, true );
    }

    Ref<DescriptorSetLayout> MeshRenderer::GetNodeSetLayout( GraphicContext &a_GraphicContext )
    {
        DescriptorSetLayoutCreateInfo l_NodeBindLayout{};
        l_NodeBindLayout.Bindings = {
            DescriptorBindingInfo{ 0, Internal::eDescriptorType::UNIFORM_BUFFER, { Internal::eShaderStageTypeFlags::VERTEX } } };
        return New<DescriptorSetLayout>( a_GraphicContext, l_NodeBindLayout );
    }

    std::vector<Ref<DescriptorSetLayout>> MeshRenderer::GetDescriptorSetLayout()
    {
        return { CameraSetLayout, TextureSetLayout, NodeSetLayout };
    }

    std::vector<sPushConstantRange> MeshRenderer::GetPushConstantLayout()
    {
        return { { { Internal::eShaderStageTypeFlags::FRAGMENT }, 0, sizeof( MaterialPushConstants ) } };
    };

    MeshRenderer::MeshRenderer( GraphicContext &a_GraphicContext, MeshRendererCreateInfo const &a_CreateInfo )
        : SceneRenderPipeline<VertexData>( a_GraphicContext )
        , Spec{ a_CreateInfo }
    {

        SceneRenderPipelineCreateInfo l_CreateInfo{};
        l_CreateInfo.Opaque         = a_CreateInfo.Opaque;
        l_CreateInfo.LineWidth      = a_CreateInfo.LineWidth;
        l_CreateInfo.VertexShader   = a_CreateInfo.VertexShader;
        l_CreateInfo.FragmentShader = a_CreateInfo.FragmentShader;
        l_CreateInfo.RenderPass     = a_CreateInfo.RenderPass;

        CameraSetLayout  = GetCameraSetLayout( mGraphicContext );
        TextureSetLayout = GetTextureSetLayout( mGraphicContext );
        NodeSetLayout    = GetNodeSetLayout( mGraphicContext );

        Initialize( l_CreateInfo );
    }

} // namespace LTSE::Core