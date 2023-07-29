#pragma once
#include "Core/Memory.h"

#include "Graphics/API.h"

#include "Scene/Components.h"
#include "Scene/Scene.h"

#include "ASceneRenderer.h"
#include "SceneRenderData.h"

// #include "CoordinateGridRenderer.h"

#include "Renderer2/Common/LightInputData.hpp"

// #include "MeshRenderer.h"
// #include "ParticleSystemRenderer.h"

namespace SE::Core
{
    using namespace math;
    using namespace SE::Graphics;
    namespace fs = std::filesystem;

    struct NewShadowMeshRendererCreateInfo
    {
        Ref<IRenderContext> RenderPass = nullptr;
    };

    class NewShadowMeshRenderer
    {

      public:
        NewShadowMeshRendererCreateInfo Spec = {};

        Ref<IDescriptorSetLayout> CameraSetLayout = nullptr;
        Ref<IDescriptorSetLayout> NodeSetLayout   = nullptr;

      public:
        NewShadowMeshRenderer() = default;
        NewShadowMeshRenderer( Ref<IGraphicContext> aGraphicContext, NewShadowMeshRendererCreateInfo const &aCreateInfo );

        static Ref<IDescriptorSetLayout> GetCameraSetLayout( Ref<IGraphicContext> aGraphicContext );
        Ref<IGraphicsPipeline>           Pipeline()
        {
            return mPipeline;
        }

        ~NewShadowMeshRenderer() = default;

      private:
        Ref<IGraphicContext>   mGraphicContext    = nullptr;
        Ref<IGraphicBuffer>    mCameraBuffer      = nullptr;
        Ref<IDescriptorSet>    mCameraDescriptors = nullptr;
        Ref<IGraphicsPipeline> mPipeline          = nullptr;
    };

    class NewOmniShadowMeshRenderer
    {

      public:
        NewShadowMeshRendererCreateInfo Spec = {};

        Ref<IDescriptorSetLayout> CameraSetLayout = nullptr;
        Ref<IDescriptorSetLayout> NodeSetLayout   = nullptr;

      public:
        NewOmniShadowMeshRenderer() = default;
        NewOmniShadowMeshRenderer( Ref<IGraphicContext> aGraphicContext, NewShadowMeshRendererCreateInfo const &aCreateInfo );

        static Ref<IDescriptorSetLayout> GetCameraSetLayout( Ref<IGraphicContext> aGraphicContext );
        static Ref<IDescriptorSetLayout> GetNodeSetLayout( Ref<IGraphicContext> aGraphicContext );

        Ref<IGraphicsPipeline> Pipeline()
        {
            return mPipeline;
        }

        ~NewOmniShadowMeshRenderer() = default;

      private:
        Ref<IGraphicContext>   mGraphicContext    = nullptr;
        Ref<IGraphicBuffer>    mCameraBuffer      = nullptr;
        Ref<IDescriptorSet>    mCameraDescriptors = nullptr;
        Ref<IGraphicsPipeline> mPipeline          = nullptr;
    };

    class NewShadowSceneRenderer : public BaseSceneRenderer
    {
      public:
        NewShadowMatrices     View;
        NewOmniShadowMatrices mOmniView;

      public:
        NewShadowSceneRenderer() = default;
        NewShadowSceneRenderer( Ref<IGraphicContext> aGraphicContext );

        ~NewShadowSceneRenderer() = default;

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
        Ref<IRenderContext>        mDirectionalShadowMapRenderContext    = nullptr;
        Ref<ISampler2D>            mDirectionalShadowMapSampler          = nullptr;
        Ref<IGraphicBuffer>        mDirectionalShadowCameraUniformBuffer = nullptr;
        Ref<IDescriptorSet>        mDirectionalShadowSceneDescriptor     = nullptr;
        Ref<NewShadowMeshRenderer> mRenderPipeline                       = nullptr;

        std::vector<std::array<Ref<IRenderContext>, 6>> mPointLightsShadowMapRenderContext    = {};
        std::vector<Ref<ISamplerCubeMap>>               mPointLightShadowMapSamplers          = {};
        std::vector<std::array<Ref<IGraphicBuffer>, 6>> mPointLightsShadowCameraUniformBuffer = {};
        std::vector<std::array<Ref<IDescriptorSet>, 6>> mPointLightsShadowSceneDescriptors    = {};
        Ref<NewOmniShadowMeshRenderer>                  mOmniRenderPipeline                   = nullptr;

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