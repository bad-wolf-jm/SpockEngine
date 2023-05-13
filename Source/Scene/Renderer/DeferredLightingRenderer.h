#pragma once

#include <filesystem>

#include "Core/Math/Types.h"
#include "Core/Memory.h"

#include "Graphics/Vulkan/DescriptorSet.h"
#include "Graphics/Vulkan/VkGraphicsPipeline.h"
#include "Graphics/Vulkan/VkGraphicContext.h"

#include "Scene/VertexData.h"

#include "Graphics/Vulkan/VkAbstractRenderPass.h"
#include "SceneRenderPipeline.h"

namespace SE::Core
{

    using namespace math;
    using namespace SE::Graphics;
    namespace fs = std::filesystem;

    struct DeferredLightingRendererCreateInfo
    {
        bool Opaque = false;

        Ref<sVkAbstractRenderPassObject> RenderPass = nullptr;
    };

    class DeferredLightingRenderer : public SceneRenderPipeline<EmptyVertexData>
    {

      public:
        struct MaterialPushConstants
        {
            uint32_t mNumSamples;
        };

        DeferredLightingRendererCreateInfo Spec = {};

        Ref<DescriptorSetLayout> CameraSetLayout  = nullptr;
        Ref<DescriptorSetLayout> TextureSetLayout = nullptr;

        Ref<DescriptorSetLayout> DirectionalShadowSetLayout = nullptr;
        Ref<DescriptorSetLayout> SpotlightShadowSetLayout = nullptr;
        Ref<DescriptorSetLayout> PointLightShadowSetLayout = nullptr;

      public:
        DeferredLightingRenderer() = default;
        DeferredLightingRenderer( Ref<VkGraphicContext> aGraphicContext, DeferredLightingRendererCreateInfo const &aCreateInfo );

        static Ref<DescriptorSetLayout> GetCameraSetLayout( Ref<VkGraphicContext> aGraphicContext );
        static Ref<DescriptorSetLayout> GetTextureSetLayout( Ref<VkGraphicContext> aGraphicContext );
        static Ref<DescriptorSetLayout> GetDirectionalShadowSetLayout( Ref<VkGraphicContext> aGraphicContext );
        static Ref<DescriptorSetLayout> GetSpotlightShadowSetLayout( Ref<VkGraphicContext> aGraphicContext );
        static Ref<DescriptorSetLayout> GetPointLightShadowSetLayout( Ref<VkGraphicContext> aGraphicContext );

        std::vector<Ref<DescriptorSetLayout>> GetDescriptorSetLayout();
        std::vector<sPushConstantRange>       GetPushConstantLayout();

        ~DeferredLightingRenderer() = default;
    };

} // namespace SE::Core