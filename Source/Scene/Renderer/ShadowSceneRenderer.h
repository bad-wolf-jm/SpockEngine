#pragma once
#include "Core/Memory.h"

#include "Graphics/Vulkan/ARenderContext.h"
#include "Graphics/Vulkan/DescriptorSet.h"
#include "Graphics/Vulkan/GraphicsPipeline.h"
#include "Graphics/Vulkan/VkRenderTarget.h"

#include "Scene/Components.h"
#include "Scene/Scene.h"

#include "Renderer/ASceneRenderer.h"
#include "Renderer/SceneRenderData.h"

#include "CoordinateGridRenderer.h"
#include "MeshRenderer.h"
#include "ParticleSystemRenderer.h"

namespace SE::Core
{
    using namespace math;
    using namespace SE::Graphics;
    namespace fs = std::filesystem;

    struct ShadowMeshRendererCreateInfo
    {
        Ref<sVkAbstractRenderPassObject> RenderPass = nullptr;
    };

    class ShadowMeshRenderer : public SceneRenderPipeline<VertexData>
    {

      public:
        ShadowMeshRendererCreateInfo Spec = {};

        Ref<DescriptorSetLayout> CameraSetLayout = nullptr;
        Ref<DescriptorSetLayout> NodeSetLayout   = nullptr;

      public:
        ShadowMeshRenderer() = default;
        ShadowMeshRenderer( Ref<VkGraphicContext> aGraphicContext, ShadowMeshRendererCreateInfo const &aCreateInfo );

        static Ref<DescriptorSetLayout> GetCameraSetLayout( Ref<VkGraphicContext> aGraphicContext );
        static Ref<DescriptorSetLayout> GetNodeSetLayout( Ref<VkGraphicContext> aGraphicContext );

        std::vector<Ref<DescriptorSetLayout>> GetDescriptorSetLayout();
        std::vector<sPushConstantRange>       GetPushConstantLayout();

        ~ShadowMeshRenderer() = default;
    };

    class ShadowSceneRenderer : public ASceneRenderer
    {
      public:
        ShadowMatrices View;

      public:
        ShadowSceneRenderer() = default;
        ShadowSceneRenderer( Ref<VkGraphicContext> aGraphicContext );

        ~ShadowSceneRenderer() = default;

        Ref<VkTexture2D> GetOutputImage();

        void Update( Ref<Scene> aWorld );
        void Render();

        void ResizeOutput( uint32_t aOutputWidth, uint32_t aOutputHeight );

        static Ref<DescriptorSet> GetDirectionalShadowMapsLayout();

        std::vector<Ref<Graphics::VkSampler2D>> &GetDirectionalShadowMapSamplers() { return mDirectionalShadowMapSamplers; };
        std::vector<Ref<Graphics::VkSampler2D>> &GetSpotlightShadowMapSamplers() { return mSpotlightShadowMapSamplers; };

      protected:
        std::vector<ARenderContext>             mDirectionalShadowMapRenderContext    = {};
        std::vector<Ref<Graphics::VkSampler2D>> mDirectionalShadowMapSamplers         = {};
        std::vector<Ref<VkGpuBuffer>>           mDirectionalShadowCameraUniformBuffer = {};
        std::vector<Ref<DescriptorSet>>         mDirectionalShadowSceneDescriptors    = {};
        ShadowMeshRenderer                      mRenderPipeline{};

        std::vector<ARenderContext>             mSpotlightShadowMapRenderContext    = {};
        std::vector<Ref<Graphics::VkSampler2D>> mSpotlightShadowMapSamplers         = {};
        std::vector<Ref<VkGpuBuffer>>           mSpotlightShadowCameraUniformBuffer = {};
        std::vector<Ref<DescriptorSet>>         mSpotlightShadowSceneDescriptors    = {};
        ShadowMeshRenderer                      mSpotlightRenderPipeline{};

        std::vector<Ref<VkRenderTarget>> mPointLightShadowMaps = {};
        std::vector<Ref<VkRenderTarget>> mSpotlightShadowMaps  = {};

        Ref<VkRenderTarget> mGeometryRenderTarget = nullptr;
        ARenderContext      mGeometryContext{};

        Ref<VkGpuBuffer> mCameraUniformBuffer    = nullptr;
        Ref<VkGpuBuffer> mShaderParametersBuffer = nullptr;

        Ref<DescriptorSetLayout> mCameraSetLayout  = nullptr;
        Ref<DescriptorSetLayout> mNodeSetLayout    = nullptr;
        Ref<DescriptorSetLayout> mTextureSetLayout = nullptr;

        Ref<DescriptorSet> mSceneDescriptors = nullptr;
        Ref<DescriptorSet> mNodeDescriptors  = nullptr;

        Ref<DescriptorSetLayout> mShadowMapDescriptorLayout = nullptr;
        Ref<DescriptorSet>       mShadowMapDescriptorSet    = nullptr;

        // ShadowMeshRenderer mRenderPipeline{};
    };

} // namespace SE::Core