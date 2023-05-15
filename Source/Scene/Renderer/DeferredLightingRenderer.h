#pragma once

#include <filesystem>

#include "Core/Math/Types.h"
#include "Core/Memory.h"

// #include "Graphics/Vulkan/DescriptorSet.h"
// #include "Graphics/Vulkan/VkGraphicsPipeline.h"
// #include "Graphics/Vulkan/IGraphicContext.h"
// #include "Graphics/Vulkan/VkRenderPass.h"
#include "Graphics/API.h"

#include "Scene/VertexData.h"

#include "SceneRenderPipeline.h"

namespace SE::Core
{

    using namespace math;
    using namespace SE::Graphics;
    namespace fs = std::filesystem;

    struct DeferredLightingRendererCreateInfo
    {
        bool Opaque = false;

        Ref<VkRenderPass> RenderPass = nullptr;
    };

    class DeferredLightingRenderer // : public SceneRenderPipeline<EmptyVertexData>
    {

      public:
        struct MaterialPushConstants
        {
            uint32_t mNumSamples;
        };

        // DeferredLightingRendererCreateInfo Spec = {};
        // Ref<DescriptorSetLayout> CameraSetLayout  = nullptr;
        // Ref<DescriptorSetLayout> TextureSetLayout = nullptr;
        // Ref<DescriptorSetLayout> DirectionalShadowSetLayout = nullptr;
        // Ref<DescriptorSetLayout> SpotlightShadowSetLayout = nullptr;
        // Ref<DescriptorSetLayout> PointLightShadowSetLayout = nullptr;

      public:
        DeferredLightingRenderer() = default;
        DeferredLightingRenderer( Ref<IGraphicContext> aGraphicContext, Ref<IRenderContext> aRenderContext );

        static Ref<IDescriptorSetLayout> GetCameraSetLayout( Ref<IGraphicContext> aGraphicContext );
        static Ref<IDescriptorSetLayout> GetTextureSetLayout( Ref<IGraphicContext> aGraphicContext );
        static Ref<IDescriptorSetLayout> GetDirectionalShadowSetLayout( Ref<IGraphicContext> aGraphicContext );
        static Ref<IDescriptorSetLayout> GetSpotlightShadowSetLayout( Ref<IGraphicContext> aGraphicContext );
        static Ref<IDescriptorSetLayout> GetPointLightShadowSetLayout( Ref<IGraphicContext> aGraphicContext );

        // std::vector<Ref<DescriptorSetLayout>> GetDescriptorSetLayout();
        // std::vector<sPushConstantRange>       GetPushConstantLayout();

        ~DeferredLightingRenderer() = default;

      private:
        Ref<IGraphicContext>   mGraphicContext    = nullptr;
        Ref<IGraphicBuffer>    mCameraBuffer = nullptr;
        Ref<IGraphicsPipeline> mPipeline     = nullptr;
    };

} // namespace SE::Core