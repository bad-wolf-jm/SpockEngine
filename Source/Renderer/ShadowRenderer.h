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
        mat4 mMVP;

        ShadowMatrices()  = default;
        ~ShadowMatrices() = default;

        ShadowMatrices( const ShadowMatrices & ) = default;
    };

    struct OmniShadowMatrices
    {
        mat4 mMVP;
        vec4 mLightPos;

        OmniShadowMatrices()  = default;
        ~OmniShadowMatrices() = default;

        OmniShadowMatrices( const OmniShadowMatrices & ) = default;
    };
    struct ShadowMeshRendererCreateInfo
    {
        ref_t<IRenderContext> RenderPass = nullptr;
    };

    class ShadowMeshRenderer
    {

      public:
        ShadowMeshRendererCreateInfo Spec = {};

      public:
        ShadowMeshRenderer() = default;
        ShadowMeshRenderer( ref_t<IGraphicContext> aGraphicContext, ShadowMeshRendererCreateInfo const &aCreateInfo );

        ref_t<IGraphicsPipeline> Pipeline()
        {
            return mPipeline;
        }

        ref_t<IDescriptorSet> View()
        {
            return mCameraDescriptors;
        }

        void SetView( ShadowMatrices const &aView );

        ~ShadowMeshRenderer() = default;

      private:
        ref_t<IGraphicContext>      mGraphicContext    = nullptr;
        ref_t<IGraphicBuffer>       mCameraBuffer      = nullptr;
        ref_t<IDescriptorSetLayout> mCameraSetLayout   = nullptr;
        ref_t<IDescriptorSet>       mCameraDescriptors = nullptr;
        ref_t<IGraphicsPipeline>    mPipeline          = nullptr;
    };

    class OmniShadowMeshRenderer
    {

      public:
        ShadowMeshRendererCreateInfo Spec = {};

      public:
        OmniShadowMeshRenderer() = default;
        OmniShadowMeshRenderer( ref_t<IGraphicContext> aGraphicContext, ShadowMeshRendererCreateInfo const &aCreateInfo );

        ref_t<IGraphicsPipeline> Pipeline()
        {
            return mPipeline;
        }

        ~OmniShadowMeshRenderer() = default;

        ref_t<IDescriptorSet> AllocateDescriptors();

      private:
        ref_t<IGraphicContext>      mGraphicContext    = nullptr;
        ref_t<IGraphicBuffer>       mCameraBuffer      = nullptr;
        ref_t<IDescriptorSet>       mCameraDescriptors = nullptr;
        ref_t<IDescriptorSetLayout> mCameraSetLayout   = nullptr;
        ref_t<IGraphicsPipeline>    mPipeline          = nullptr;
    };

    class ShadowSceneRenderer : public BaseSceneRenderer
    {
      public:
        ShadowMatrices     View;
        OmniShadowMatrices mOmniView;

      public:
        ShadowSceneRenderer() = default;
        ShadowSceneRenderer( ref_t<IGraphicContext> aGraphicContext );

        ~ShadowSceneRenderer() = default;

        ref_t<ITexture2D> GetOutputImage();

        void Update( ref_t<Scene> aWorld );
        void Render();

        void               ResizeOutput( uint32_t aOutputWidth, uint32_t aOutputHeight );
        ref_t<IRenderTarget> NewRenderTarget( uint32_t aOutputWidth, uint32_t aOutputHeight );

        static ref_t<IDescriptorSet> GetDirectionalShadowMapsLayout();

        ref_t<ISampler2D> &GetDirectionalShadowMapSampler()
        {
            return mDirectionalShadowMapSampler;
        }

        vec_t<ref_t<ISamplerCubeMap>> &GetPointLightShadowMapSamplers()
        {
            return mPointLightShadowMapSamplers;
        }

        void SetLights( sDirectionalLight const &aDirectionalLights );

        void SetLights( vec_t<sPunctualLight> const &aPointLights );

      private:
        void RenderPunctualShadowMap( vec3 aLightPosition, std::array<ref_t<IRenderContext>, 6> aContext,
                                      std::array<ref_t<IGraphicBuffer>, 6> const &aUniforms,
                                      std::array<ref_t<IDescriptorSet>, 6> const &aDescriptors );
        void RenderCubeFace( mat4 viewMatrix, mat4 lProjection, ref_t<IRenderContext> lContext, ref_t<IDescriptorSet> aDescriptors );

      protected:
        ref_t<IRenderContext>     mDirectionalShadowMapRenderContext    = nullptr;
        ref_t<ISampler2D>         mDirectionalShadowMapSampler          = nullptr;
        ref_t<IGraphicBuffer>     mDirectionalShadowCameraUniformBuffer = nullptr;
        ref_t<IDescriptorSet>     mDirectionalShadowSceneDescriptor     = nullptr;
        ref_t<ShadowMeshRenderer> mRenderPipeline                       = nullptr;

        vec_t<std::array<ref_t<IRenderContext>, 6>> mPointLightsShadowMapRenderContext    = {};
        vec_t<ref_t<ISamplerCubeMap>>               mPointLightShadowMapSamplers          = {};
        vec_t<std::array<ref_t<IGraphicBuffer>, 6>> mPointLightsShadowCameraUniformBuffer = {};
        vec_t<std::array<ref_t<IDescriptorSet>, 6>> mPointLightsShadowSceneDescriptors    = {};
        ref_t<OmniShadowMeshRenderer>                     mOmniRenderPipeline                   = nullptr;

        ref_t<IRenderTarget>  mGeometryRenderTarget = nullptr;
        ref_t<IRenderContext> mGeometryContext{};

        ref_t<IGraphicBuffer> mCameraUniformBuffer    = nullptr;
        ref_t<IGraphicBuffer> mShaderParametersBuffer = nullptr;

        ref_t<IDescriptorSetLayout> mCameraSetLayout  = nullptr;
        ref_t<IDescriptorSetLayout> mNodeSetLayout    = nullptr;
        ref_t<IDescriptorSetLayout> mTextureSetLayout = nullptr;

        ref_t<IDescriptorSet> mSceneDescriptors = nullptr;
        ref_t<IDescriptorSet> mNodeDescriptors  = nullptr;

        ref_t<IDescriptorSetLayout> mShadowMapDescriptorLayout = nullptr;
        ref_t<IDescriptorSet>       mShadowMapDescriptorSet    = nullptr;

        sDirectionalLight           mDirectionalLight;
        vec_t<sPunctualLight> mPointLights;
    };

} // namespace SE::Core