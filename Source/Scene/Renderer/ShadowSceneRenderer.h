#pragma once
#include "Core/Memory.h"

// #include "Graphics/Vulkan/Ref<IRenderContext>.h"
// #include "Graphics/Vulkan/IDescriptorSet.h"
// #include "Graphics/Vulkan/VkGraphicsPipeline.h"
// #include "Graphics/Vulkan/IRenderTarget.h"
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
        Ref<IRenderContext> RenderPass = nullptr;
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

    class OmniShadowMeshRenderer // : public SceneRenderPipeline<VertexData>
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

        Ref<ITexture> GetOutputImage();

        void Update( Ref<Scene> aWorld );
        void Render();

        void               ResizeOutput( uint32_t aOutputWidth, uint32_t aOutputHeight );
        Ref<IRenderTarget> NewRenderTarget( uint32_t aOutputWidth, uint32_t aOutputHeight );

        static Ref<IDescriptorSet> GetDirectionalShadowMapsLayout();

        std::vector<Ref<ISampler2D>>      &GetDirectionalShadowMapSamplers() { return mDirectionalShadowMapSamplers; };
        std::vector<Ref<ISampler2D>>      &GetSpotlightShadowMapSamplers() { return mSpotlightShadowMapSamplers; };
        std::vector<Ref<ISamplerCubeMap>> &GetPointLightShadowMapSamplers() { return mPointLightShadowMapSamplers; };

      protected:
        std::vector<Ref<IRenderContext>> mDirectionalShadowMapRenderContext    = {};
        std::vector<Ref<ISampler2D>>     mDirectionalShadowMapSamplers         = {};
        std::vector<Ref<IGraphicBuffer>> mDirectionalShadowCameraUniformBuffer = {};
        std::vector<Ref<IDescriptorSet>> mDirectionalShadowSceneDescriptors    = {};
        ShadowMeshRenderer               mRenderPipeline{};

        std::vector<Ref<IRenderContext>> mSpotlightShadowMapRenderContext    = {};
        std::vector<Ref<ISampler2D>>     mSpotlightShadowMapSamplers         = {};
        std::vector<Ref<IGraphicBuffer>> mSpotlightShadowCameraUniformBuffer = {};
        std::vector<Ref<IDescriptorSet>> mSpotlightShadowSceneDescriptors    = {};

        std::vector<std::array<Ref<IRenderContext>, 6>> mPointLightsShadowMapRenderContext    = {};
        std::vector<Ref<ISamplerCubeMap>>               mPointLightShadowMapSamplers          = {};
        std::vector<std::array<Ref<IGraphicBuffer>, 6>> mPointLightsShadowCameraUniformBuffer = {};
        std::vector<std::array<Ref<IDescriptorSet>, 6>> mPointLightsShadowSceneDescriptors    = {};
        OmniShadowMeshRenderer                          mOmniRenderPipeline{};

        Ref<IRenderTarget>  mGeometryRenderTarget = nullptr;
        Ref<IRenderContext> mGeometryContext{};

        Ref<IGraphicBuffer> mCameraUniformBuffer    = nullptr;
        Ref<IGraphicBuffer> mShaderParametersBuffer = nullptr;

        Ref<IDescriptorSetLayout> mCameraSetLayout  = nullptr;
        Ref<IDescriptorSetLayout> mNodeSetLayout    = nullptr;
        Ref<IDescriptorSetLayout> mTextureSetLayout = nullptr;

        Ref<IDescriptorSet> mSceneDescriptors = nullptr;
        Ref<IDescriptorSet> mNodeDescriptors  = nullptr;

        Ref<IDescriptorSetLayout> mShadowMapDescriptorLayout = nullptr;
        Ref<IDescriptorSet>       mShadowMapDescriptorSet    = nullptr;

        // ShadowMeshRenderer mRenderPipeline{};
    };

} // namespace SE::Core