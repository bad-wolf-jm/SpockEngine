#pragma once
#include "Core/Memory.h"

// #include "Graphics/Vulkan/VkRenderContext.h"
// #include "Graphics/Vulkan/DescriptorSet.h"
// #include "Graphics/Vulkan/VkGraphicsPipeline.h"
// #include "Graphics/Vulkan/VkRenderTarget.h"
#include "Graphics/API.h"

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
        Ref<VkRenderPass> RenderPass = nullptr;
    };

    class ShadowMeshRenderer //: public SceneRenderPipeline<VertexData>
    {

      public:
        ShadowMeshRendererCreateInfo Spec = {};

        Ref<IDescriptorSetLayout> CameraSetLayout = nullptr;
        Ref<IDescriptorSetLayout> NodeSetLayout   = nullptr;

      public:
        ShadowMeshRenderer() = default;
        ShadowMeshRenderer( Ref<IGraphicContext> aGraphicContext, ShadowMeshRendererCreateInfo const &aCreateInfo );

        // static Ref<IDescriptorSetLayout> GetCameraSetLayout( Ref<IGraphicContext> aGraphicContext );
        // static Ref<IDescriptorSetLayout> GetNodeSetLayout( Ref<IGraphicContext> aGraphicContext );
        // std::vector<Ref<IDescriptorSetLayout>> GetDescriptorSetLayout();
        // std::vector<sPushConstantRange>        GetPushConstantLayout();

        ~ShadowMeshRenderer() = default;
    };

    class OmniShadowMeshRenderer// : public SceneRenderPipeline<VertexData>
    {

      public:
        ShadowMeshRendererCreateInfo Spec = {};

        Ref<IDescriptorSetLayout> CameraSetLayout = nullptr;
        Ref<IDescriptorSetLayout> NodeSetLayout   = nullptr;

      public:
        OmniShadowMeshRenderer() = default;
        OmniShadowMeshRenderer( Ref<IGraphicContext> aGraphicContext, ShadowMeshRendererCreateInfo const &aCreateInfo );

        static Ref<IDescriptorSetLayout> GetCameraSetLayout( Ref<IGraphicContext> aGraphicContext );
        static Ref<IDescriptorSetLayout> GetNodeSetLayout( Ref<IGraphicContext> aGraphicContext );

        std::vector<Ref<IDescriptorSetLayout>> GetDescriptorSetLayout();
        std::vector<sPushConstantRange>        GetPushConstantLayout();

        ~OmniShadowMeshRenderer() = default;
    };

    class ShadowSceneRenderer : public ASceneRenderer
    {
      public:
        ShadowMatrices     View;
        OmniShadowMatrices mOmniView;

      public:
        ShadowSceneRenderer() = default;
        ShadowSceneRenderer( Ref<IGraphicContext> aGraphicContext );

        ~ShadowSceneRenderer() = default;

        Ref<VkTexture2D> GetOutputImage();

        void Update( Ref<Scene> aWorld );
        void Render();

        void                ResizeOutput( uint32_t aOutputWidth, uint32_t aOutputHeight );
        Ref<VkRenderTarget> NewRenderTarget( uint32_t aOutputWidth, uint32_t aOutputHeight );

        static Ref<DescriptorSet> GetDirectionalShadowMapsLayout();

        std::vector<Ref<Graphics::VkSampler2D>>      &GetDirectionalShadowMapSamplers() { return mDirectionalShadowMapSamplers; };
        std::vector<Ref<Graphics::VkSampler2D>>      &GetSpotlightShadowMapSamplers() { return mSpotlightShadowMapSamplers; };
        std::vector<Ref<Graphics::VkSamplerCubeMap>> &GetPointLightShadowMapSamplers() { return mPointLightShadowMapSamplers; };

      protected:
        std::vector<VkRenderContext>            mDirectionalShadowMapRenderContext    = {};
        std::vector<Ref<Graphics::VkSampler2D>> mDirectionalShadowMapSamplers         = {};
        std::vector<Ref<VkGpuBuffer>>           mDirectionalShadowCameraUniformBuffer = {};
        std::vector<Ref<DescriptorSet>>         mDirectionalShadowSceneDescriptors    = {};
        ShadowMeshRenderer                      mRenderPipeline{};

        std::vector<VkRenderContext>            mSpotlightShadowMapRenderContext    = {};
        std::vector<Ref<Graphics::VkSampler2D>> mSpotlightShadowMapSamplers         = {};
        std::vector<Ref<VkGpuBuffer>>           mSpotlightShadowCameraUniformBuffer = {};
        std::vector<Ref<DescriptorSet>>         mSpotlightShadowSceneDescriptors    = {};

        std::vector<std::array<VkRenderContext, 6>>    mPointLightsShadowMapRenderContext    = {};
        std::vector<Ref<Graphics::VkSamplerCubeMap>>   mPointLightShadowMapSamplers          = {};
        std::vector<std::array<Ref<VkGpuBuffer>, 6>>   mPointLightsShadowCameraUniformBuffer = {};
        std::vector<std::array<Ref<DescriptorSet>, 6>> mPointLightsShadowSceneDescriptors    = {};
        OmniShadowMeshRenderer                         mOmniRenderPipeline{};

        Ref<VkRenderTarget> mGeometryRenderTarget = nullptr;
        VkRenderContext     mGeometryContext{};

        Ref<VkGpuBuffer> mCameraUniformBuffer    = nullptr;
        Ref<VkGpuBuffer> mShaderParametersBuffer = nullptr;

        Ref<IDescriptorSetLayout> mCameraSetLayout  = nullptr;
        Ref<IDescriptorSetLayout> mNodeSetLayout    = nullptr;
        Ref<IDescriptorSetLayout> mTextureSetLayout = nullptr;

        Ref<DescriptorSet> mSceneDescriptors = nullptr;
        Ref<DescriptorSet> mNodeDescriptors  = nullptr;

        Ref<IDescriptorSetLayout> mShadowMapDescriptorLayout = nullptr;
        Ref<DescriptorSet>        mShadowMapDescriptorSet    = nullptr;

        // ShadowMeshRenderer mRenderPipeline{};
    };

} // namespace SE::Core