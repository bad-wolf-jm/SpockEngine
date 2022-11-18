#include "DeferredLightingRenderer.h"

#include <chrono>

#include "Core/Vulkan/VkPipeline.h"
#include "Scene/Primitives/Primitives.h"
#include "Scene/VertexData.h"

#include "Core/Logging.h"
#include "Core/Resource.h"

namespace LTSE::Core
{

    using namespace math;
    using namespace Internal;

    Ref<DescriptorSetLayout> DeferredLightingRenderer::GetCameraSetLayout( GraphicContext &aGraphicContext )
    {
        DescriptorSetLayoutCreateInfo l_CameraBindLayout{};
        l_CameraBindLayout.Bindings = {
            DescriptorBindingInfo{ 0, eDescriptorType::UNIFORM_BUFFER, { eShaderStageTypeFlags::FRAGMENT } },
            DescriptorBindingInfo{ 1, eDescriptorType::UNIFORM_BUFFER, { eShaderStageTypeFlags::FRAGMENT } } };

        return New<DescriptorSetLayout>( aGraphicContext, l_CameraBindLayout );
    }

    Ref<DescriptorSetLayout> DeferredLightingRenderer::GetTextureSetLayout( GraphicContext &aGraphicContext )
    {
        DescriptorSetLayoutCreateInfo lTextureBindLayout{};
        lTextureBindLayout.Bindings = {
            DescriptorBindingInfo{ 0, eDescriptorType::COMBINED_IMAGE_SAMPLER, { eShaderStageTypeFlags::FRAGMENT } },
            DescriptorBindingInfo{ 1, eDescriptorType::COMBINED_IMAGE_SAMPLER, { eShaderStageTypeFlags::FRAGMENT } },
            DescriptorBindingInfo{ 2, eDescriptorType::COMBINED_IMAGE_SAMPLER, { eShaderStageTypeFlags::FRAGMENT } },
            DescriptorBindingInfo{ 3, eDescriptorType::COMBINED_IMAGE_SAMPLER, { eShaderStageTypeFlags::FRAGMENT } } };

        return New<DescriptorSetLayout>( aGraphicContext, lTextureBindLayout, false );
    }

    std::vector<Ref<DescriptorSetLayout>> DeferredLightingRenderer::GetDescriptorSetLayout()
    {
        return { CameraSetLayout, TextureSetLayout };
    }

    std::vector<sPushConstantRange> DeferredLightingRenderer::GetPushConstantLayout()
    {
        return { { { eShaderStageTypeFlags::FRAGMENT }, 0, sizeof( MaterialPushConstants ) } };
    };

    DeferredLightingRenderer::DeferredLightingRenderer( GraphicContext                           &aGraphicContext,
                                                        DeferredLightingRendererCreateInfo const &aCreateInfo )
        : SceneRenderPipeline<EmptyVertexData>( aGraphicContext )
        , Spec{ aCreateInfo }
    {
        SceneRenderPipelineCreateInfo lCreateInfo{};
        lCreateInfo.VertexShader   = "Shaders/Deferred/DeferredLightingMSAA.vert.spv";
        lCreateInfo.FragmentShader = "Shaders/Deferred/DeferredLightingMSAA.frag.spv";
        lCreateInfo.RenderPass     = aCreateInfo.RenderPass;
        lCreateInfo.DepthTest      = false;
        lCreateInfo.DepthWrite     = false;

        CameraSetLayout  = GetCameraSetLayout( mGraphicContext );
        TextureSetLayout = GetTextureSetLayout( mGraphicContext );

        Initialize( lCreateInfo );
    }

} // namespace LTSE::Core