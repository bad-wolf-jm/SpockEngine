#pragma once
#include "Core/Memory.h"

#include "Graphics/API.h"

#include "Scene/Components.h"
#include "Scene/Scene.h"

#include "ASceneRenderer.h"
// #include "SceneRenderData.h"

// #include "CoordinateGridRenderer.h"

#include "Common/LightInputData.hpp"

// #include "MeshRenderer.h"
// #include "ParticleSystemRenderer.h"

namespace SE::Core
{
    using namespace math;
    using namespace SE::Graphics;
    namespace fs = std::filesystem;

    struct ShadowMatrices
    {
        math::mat4 mMVP;

        ShadowMatrices()  = default;
        ~ShadowMatrices() = default;

        ShadowMatrices( const ShadowMatrices & ) = default;
    };

    struct OmniShadowMatrices
    {
        math::mat4 mMVP;
        math::vec4 mLightPos;

        OmniShadowMatrices()  = default;
        ~OmniShadowMatrices() = default;

        OmniShadowMatrices( const OmniShadowMatrices & ) = default;
    };
    struct ShadowMeshRendererCreateInfo
    {
        Ref<IRenderContext> RenderPass = nullptr;
    };

    class ShadowMeshRenderer
    {

      public:
        ShadowMeshRendererCreateInfo Spec = {};

        Ref<IDescriptorSetLayout> CameraSetLayout = nullptr;
        Ref<IDescriptorSetLayout> NodeSetLayout   = nullptr;

      public:
        ShadowMeshRenderer() = default;
        ShadowMeshRenderer( Ref<IGraphicContext> aGraphicContext, ShadowMeshRendererCreateInfo const &aCreateInfo );

        static Ref<IDescriptorSetLayout> GetCameraSetLayout( Ref<IGraphicContext> aGraphicContext );
        Ref<IGraphicsPipeline>           Pipeline()
        {
            return mPipeline;
        }

        ~ShadowMeshRenderer() = default;

      private:
        Ref<IGraphicContext>   mGraphicContext    = nullptr;
        Ref<IGraphicBuffer>    mCameraBuffer      = nullptr;
        Ref<IDescriptorSet>    mCameraDescriptors = nullptr;
        Ref<IGraphicsPipeline> mPipeline          = nullptr;
    };

    class OmniShadowMeshRenderer
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

        Ref<IGraphicsPipeline> Pipeline()
        {
            return mPipeline;
        }

        ~OmniShadowMeshRenderer() = default;

      private:
        Ref<IGraphicContext>   mGraphicContext    = nullptr;
        Ref<IGraphicBuffer>    mCameraBuffer      = nullptr;
        Ref<IDescriptorSet>    mCameraDescriptors = nullptr;
        Ref<IGraphicsPipeline> mPipeline          = nullptr;
    };

    class ShadowSceneRenderer : public BaseSceneRenderer
    {
      public:
        ShadowMatrices     View;
        OmniShadowMatrices mOmniView;

      public:
        ShadowSceneRenderer() = default;
        ShadowSceneRenderer( Ref<IGraphicContext> aGraphicContext );

        ~ShadowSceneRenderer() = default;

        Ref<ITexture2D> GetOutputImage();

        void Update( Ref<Scene> aWorld );
        void Render();

        void               ResizeOutput( uint32_t aOutputWidth, uint32_t aOutputHeight );
        Ref<IRenderTarget> NewRenderTarget( uint32_t aOutputWidth, uint32_t aOutputHeight );

        static Ref<IDescriptorSet> GetDirectionalShadowMapsLayout();

        Ref<ISampler2D> &GetDirectionalShadowMapSampler()
        {
            return mDirectionalShadowMapSampler;
        }

        std::vector<Ref<ISamplerCubeMap>> &GetPointLightShadowMapSamplers()
        {
            return mPointLightShadowMapSamplers;
        }

        void SetLights( sDirectionalLight const &aDirectionalLights );

        void SetLights( std::vector<sPunctualLight> const &aPointLights );

      protected:
        Ref<IRenderContext>     mDirectionalShadowMapRenderContext    = nullptr;
        Ref<ISampler2D>         mDirectionalShadowMapSampler          = nullptr;
        Ref<IGraphicBuffer>     mDirectionalShadowCameraUniformBuffer = nullptr;
        Ref<IDescriptorSet>     mDirectionalShadowSceneDescriptor     = nullptr;
        Ref<ShadowMeshRenderer> mRenderPipeline                       = nullptr;

        std::vector<std::array<Ref<IRenderContext>, 6>> mPointLightsShadowMapRenderContext    = {};
        std::vector<Ref<ISamplerCubeMap>>               mPointLightShadowMapSamplers          = {};
        std::vector<std::array<Ref<IGraphicBuffer>, 6>> mPointLightsShadowCameraUniformBuffer = {};
        std::vector<std::array<Ref<IDescriptorSet>, 6>> mPointLightsShadowSceneDescriptors    = {};
        Ref<OmniShadowMeshRenderer>                     mOmniRenderPipeline                   = nullptr;

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

        sDirectionalLight           mDirectionalLight;
        std::vector<sPunctualLight> mPointLights;
    };

} // namespace SE::Core