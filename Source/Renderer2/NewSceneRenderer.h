#pragma once
#include "Core/Memory.h"

#include "Graphics/API.h"

#include "Scene/Components.h"
// #include "Scene/Scene.h"

#include "ASceneRenderer.h"
#include "Renderer/SceneRenderData.h"

#include "Scene/Renderer/EffectProcessor.h"
#include "Scene/Renderer/ShadowSceneRenderer.h"
// #include "CoordinateGridRenderer.h"
// #include "MeshRenderer.h"
// #include "ParticleSystemRenderer.h"

namespace SE::Core
{
    using namespace math;

    enum class eShadingModel
    {
        STANDARD,
        SUBSURFACE,
        CLOTH,
        UNLIT
    }


    struct sShaderMaterial
    {
        using Tex2D = Ref<ISampler2D>;
        
        std::string mName = "";

        eMaterialType mType       = eMaterialType::Opaque;
        float         mLineWidth  = 1.0f;
        bool          mIsTwoSided = true;
        bool          mHasUV1     = true;

        vec4  mBaseColorFactor    = { 1.0f, 1.0f, 1.0f, 1.0f };
        int   mBaseColorUVChannel = 0;
        Tex2D mBaseColorTexture   = nullptr;

        float mMetallicFactor     = 0.0f;
        float mRoughnessFactor    = 1.0f;
        int   mMetalnessUVChannel = 0;
        Tex2D mMetaRoughTexture   = nullptr;

        float mOcclusionStrength  = 0.0f;
        int   mOcclusionUVChannel = 0;
        Tex2D mOcclusionTexture   = nullptr;

        vec4  mEmissiveFactor    = { 0.0f, 0.0f, 0.0f, 0.0f };
        Tex2D mEmissiveTexture   = nullptr;
        int   mEmissiveUVChannel = 0;

        int   mNormalUVChannel = 0;
        Tex2D mNormalTexture   = nullptr;

        sShaderMaterial()                          = default;
        sShaderMaterial( const sShaderMaterial & ) = default;
    };

    class NewSceneRenderer : public BaseSceneRenderer
    {
      public:
        WorldMatrices  mView;
        CameraSettings mSettings;
        bool           mRenderCoordinateGrid = true;
        bool           mRenderGizmos         = false;
        bool           mGrayscaleRendering   = false;
        bool           mUseFXAA              = false;

      public:
        NewSceneRenderer() = default;
        NewSceneRenderer( Ref<IGraphicContext> aGraphicContext, eColorFormat aOutputFormat, uint32_t aOutputSampleCount );

        ~NewSceneRenderer() = default;

        Ref<ITexture2D> GetOutputImage();

        void Update( Ref<Scene> aWorld );
        void Render();

        void ResizeOutput( uint32_t aOutputWidth, uint32_t aOutputHeight );

      protected:
        MeshRendererCreateInfo     GetRenderPipelineCreateInfo( sMaterialShaderComponent &aPipelineSpecification );
        MeshRendererCreateInfo     GetRenderPipelineCreateInfo( sMeshRenderData &aPipelineSpecification );
        ParticleRendererCreateInfo GetRenderPipelineCreateInfo( sParticleShaderComponent &aPipelineSpecification );
        ParticleRendererCreateInfo GetRenderPipelineCreateInfo( sParticleRenderData &aPipelineSpecification );

        Ref<MeshRenderer>           GetRenderPipeline( sMaterialShaderComponent &aPipelineSpecification );
        Ref<MeshRenderer>           GetRenderPipeline( sMeshRenderData &aPipelineSpecification );
        Ref<MeshRenderer>           GetRenderPipeline( MeshRendererCreateInfo const &aPipelineSpecification );
        Ref<ParticleSystemRenderer> GetRenderPipeline( sParticleShaderComponent &aPipelineSpecification );
        Ref<ParticleSystemRenderer> GetRenderPipeline( sParticleRenderData &aPipelineSpecification );
        Ref<ParticleSystemRenderer> GetRenderPipeline( ParticleRendererCreateInfo &aPipelineSpecification );

      protected:
        Ref<IRenderTarget>  mGeometryRenderTarget = nullptr;
        Ref<IRenderContext> mGeometryContext{};

        Ref<CoordinateGridRenderer> mCoordinateGridRenderer = nullptr;
        Ref<ShadowSceneRenderer>    mShadowSceneRenderer    = nullptr;

        Ref<EffectProcessor> mCopyRenderer     = nullptr;
        Ref<EffectProcessor> mFxaaRenderer     = nullptr;
        Ref<ISampler2D>      mFxaaSampler      = nullptr;
        Ref<IRenderTarget>   mFxaaRenderTarget = nullptr;
        Ref<IRenderContext>  mFxaaContext{};

        Ref<IGraphicBuffer> mCameraUniformBuffer    = nullptr;
        Ref<IGraphicBuffer> mShaderParametersBuffer = nullptr;

        Ref<IDescriptorSetLayout> mCameraSetLayout  = nullptr;
        Ref<IDescriptorSetLayout> mNodeSetLayout    = nullptr;
        Ref<IDescriptorSetLayout> mTextureSetLayout = nullptr;

        Ref<IDescriptorSet> mSceneDescriptors = nullptr;
        Ref<IDescriptorSet> mNodeDescriptors  = nullptr;

        Ref<IDescriptorSetLayout> mLightingDirectionalShadowLayout   = nullptr;
        Ref<IDescriptorSetLayout> mLightingSpotlightShadowLayout     = nullptr;
        Ref<IDescriptorSetLayout> mLightingPointLightShadowLayout    = nullptr;
        Ref<IDescriptorSet>       mLightingPassDirectionalShadowMaps = nullptr;
        Ref<IDescriptorSet>       mLightingPassSpotlightShadowMaps   = nullptr;
        Ref<IDescriptorSet>       mLightingPassPointLightShadowMaps  = nullptr;

        std::unordered_map<MeshRendererCreateInfo, Ref<MeshRenderer>, MeshRendererCreateInfoHash> mMeshRenderers = {};
        std::unordered_map<ParticleRendererCreateInfo, Ref<ParticleSystemRenderer>, ParticleSystemRendererCreateInfoHash>
            mParticleRenderers = {};

        std::unordered_map<Entity, Ref<IDescriptorSet>> mMaterials = {};

        void CreateRenderTarget( uint32_t aOutputWidth, uint32_t aOutputHeight );
        void CreateMSAARenderTarget( uint32_t aOutputWidth, uint32_t aOutputHeight );
        void CreateFXAARenderTarget( uint32_t aOutputWidth, uint32_t aOutputHeight );
    };

} // namespace SE::Core