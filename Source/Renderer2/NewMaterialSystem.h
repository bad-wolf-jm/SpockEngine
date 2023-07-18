#pragma once

#include <unordered_map>

#include "Core/Entity/Collection.h"
#include "Core/Math/Types.h"
#include "Core/Memory.h"

#include "Graphics/API.h"

namespace SE::Core
{
    using namespace math;
    using namespace SE::Graphics;

    using Material = Entity;

    enum class eShadingModel : uint8_t
    {
        STANDARD = 0,
        SUBSURFACE,
        CLOTH,
        UNLIT
    };

    enum class eBlendMode : uint8_t
    {
        Opaque,
        Mask,
        Blend
    };

    struct sMaterialInfo
    {
        eBlendMode    mType         = eBlendMode::Opaque;
        eShadingModel mShadingModel = eShadingModel::UNLIT;
        float         mLineWidth    = 1.0f;
        bool          mIsTwoSided   = true;
        bool          mHasUV1       = true;

        sMaterialInfo()                        = default;
        sMaterialInfo( const sMaterialInfo & ) = default;
    };

    template <typename _Ty>
    struct sTextureComponent
    {
        using Tex2D = Ref<ISampler2D>;

        _Ty mFactor{};

        int   mUVChannel = 0;
        Tex2D mTexture   = nullptr;

        sTextureComponent()                            = default;
        sTextureComponent( const sTextureComponent & ) = default;
    };

    struct sBaseColorTexture : public sTextureComponent<vec4>
    {
        sBaseColorTexture()                            = default;
        sBaseColorTexture( const sBaseColorTexture & ) = default;
    };

    struct sEmissiveTexture : public sTextureComponent<vec3>
    {
        sEmissiveTexture()                           = default;
        sEmissiveTexture( const sEmissiveTexture & ) = default;
    };

    struct sMetalRoughTexture : public sTextureComponent<vec4>
    {
        float mMetallicFactor  = 0.0f;
        float mRoughnessFactor = 1.0f;

        sMetalRoughTexture()                             = default;
        sMetalRoughTexture( const sMetalRoughTexture & ) = default;
    };

    struct sNormalsTexture : public sTextureComponent<vec3>
    {
        sNormalsTexture()                          = default;
        sNormalsTexture( const sNormalsTexture & ) = default;
    };

    struct sOcclusionTexture : public sTextureComponent<float>
    {
        sOcclusionTexture()                            = default;
        sOcclusionTexture( const sOcclusionTexture & ) = default;
    };

    struct sVertexShader
    {
        std::string mCode;

        sVertexShader()                        = default;
        sVertexShader( const sVertexShader & ) = default;
    };

    struct sFragmentShader
    {
        std::string mCode;

        sFragmentShader()                          = default;
        sFragmentShader( const sFragmentShader & ) = default;
    };

    // struct sNewShaderMaterial
    // {

    //     std::string   mName         = "";
    //     eMaterialType mType         = eMaterialType::Opaque;
    //     eShadingModel mShadingModel = eShadingModel::UNLIT;
    //     float         mLineWidth    = 1.0f;
    //     bool          mIsTwoSided   = true;
    //     bool          mHasUV1       = true;

    //     vec4  mBaseColorFactor    = { 1.0f, 1.0f, 1.0f, 1.0f };
    //     int   mBaseColorUVChannel = 0;
    //     Tex2D mBaseColorTexture   = nullptr;

    //     float mMetallicFactor     = 0.0f;
    //     float mRoughnessFactor    = 1.0f;
    //     int   mMetalnessUVChannel = 0;
    //     Tex2D mMetalRoughTexture  = nullptr;

    //     float mOcclusionStrength  = 0.0f;
    //     int   mOcclusionUVChannel = 0;
    //     Tex2D mOcclusionTexture   = nullptr;

    //     vec3  mEmissiveFactor    = { 0.0f, 0.0f, 0.0f, 0.0f };
    //     int   mEmissiveUVChannel = 0;
    //     Tex2D mEmissiveTexture   = nullptr;

    //     int   mNormalUVChannel = 0;
    //     Tex2D mNormalTexture   = nullptr;

    //     fs::path mVertexShader   = "";
    //     fs::path mFragmentShader = "";

    //     sNewShaderMaterial()                             = default;
    //     sNewShaderMaterial( const sNewShaderMaterial & ) = default;

    //     size_t Hash();
    // };

    class NewMaterialSystem
    {
        //   public:
        // WorldMatrices  mView;
        // CameraSettings mSettings;
        // bool           mRenderCoordinateGrid = true;
        // bool           mRenderGizmos         = false;
        // bool           mGrayscaleRendering   = false;
        // bool           mUseFXAA              = false;

      public:
        NewMaterialSystem() = default;
        NewMaterialSystem( Ref<IGraphicContext> aGraphicContext );

        ~NewMaterialSystem() = default;
        Material CreateMaterial( std::string aName );
        Material BeginMaterial( std::string aName );
        void     EndMaterial( Material aMaterial );
        size_t   GetMaterialHash( Material aMaterial );

        // Ref<ITexture2D> GetOutputImage();

        // void Update( Ref<Scene> aWorld );
        // void Render();

        // void ResizeOutput( uint32_t aOutputWidth, uint32_t aOutputHeight );

        //   protected:
        // MeshRendererCreateInfo     GetRenderPipelineCreateInfo( sMaterialShaderComponent &aPipelineSpecification );
        // MeshRendererCreateInfo     GetRenderPipelineCreateInfo( sMeshRenderData &aPipelineSpecification );
        // ParticleRendererCreateInfo GetRenderPipelineCreateInfo( sParticleShaderComponent &aPipelineSpecification );
        // ParticleRendererCreateInfo GetRenderPipelineCreateInfo( sParticleRenderData &aPipelineSpecification );

        // Ref<MeshRenderer>           GetRenderPipeline( sMaterialShaderComponent &aPipelineSpecification );
        // Ref<MeshRenderer>           GetRenderPipeline( sMeshRenderData &aPipelineSpecification );
        // Ref<MeshRenderer>           GetRenderPipeline( MeshRendererCreateInfo const &aPipelineSpecification );
        // Ref<ParticleSystemRenderer> GetRenderPipeline( sParticleShaderComponent &aPipelineSpecification );
        // Ref<ParticleSystemRenderer> GetRenderPipeline( sParticleRenderData &aPipelineSpecification );
        // Ref<ParticleSystemRenderer> GetRenderPipeline( ParticleRendererCreateInfo &aPipelineSpecification );
      private:
        Ref<IGraphicContext> mGraphicContext;

        EntityCollection mMaterialRegistry;

        std::unordered_map<size_t, Ref<IGraphicsPipeline>> mGraphicsPipelines;

        //   protected:
        //     Ref<IRenderTarget>  mGeometryRenderTarget = nullptr;
        //     Ref<IRenderContext> mGeometryContext{};

        // Ref<CoordinateGridRenderer> mCoordinateGridRenderer = nullptr;
        // Ref<ShadowSceneRenderer>    mShadowSceneRenderer    = nullptr;

        // Ref<EffectProcessor> mCopyRenderer     = nullptr;
        // Ref<EffectProcessor> mFxaaRenderer     = nullptr;
        // Ref<ISampler2D>     mFxaaSampler      = nullptr;
        // Ref<IRenderTarget>  mFxaaRenderTarget = nullptr;
        // Ref<IRenderContext> mFxaaContext      = nullptr;

        // Ref<IGraphicBuffer> mCameraUniformBuffer    = nullptr;
        // Ref<IGraphicBuffer> mShaderParametersBuffer = nullptr;

        // Ref<IDescriptorSetLayout> mCameraSetLayout  = nullptr;
        // Ref<IDescriptorSetLayout> mNodeSetLayout    = nullptr;
        // Ref<IDescriptorSetLayout> mTextureSetLayout = nullptr;

        // Ref<IDescriptorSet> mSceneDescriptors = nullptr;
        // Ref<IDescriptorSet> mNodeDescriptors  = nullptr;

        // Ref<IDescriptorSetLayout> mLightingDirectionalShadowLayout   = nullptr;
        // Ref<IDescriptorSetLayout> mLightingSpotlightShadowLayout     = nullptr;
        // Ref<IDescriptorSetLayout> mLightingPointLightShadowLayout    = nullptr;
        // Ref<IDescriptorSet>       mLightingPassDirectionalShadowMaps = nullptr;
        // Ref<IDescriptorSet>       mLightingPassSpotlightShadowMaps   = nullptr;
        // Ref<IDescriptorSet>       mLightingPassPointLightShadowMaps  = nullptr;

        // std::unordered_map<MeshRendererCreateInfo, Ref<MeshRenderer>, MeshRendererCreateInfoHash> mMeshRenderers = {};
        // std::unordered_map<ParticleRendererCreateInfo, Ref<ParticleSystemRenderer>, ParticleSystemRendererCreateInfoHash>
        //     mParticleRenderers = {};

        // std::unordered_map<Entity, Ref<IDescriptorSet>> mMaterials = {};

        // void CreateRenderTarget( uint32_t aOutputWidth, uint32_t aOutputHeight );
        // void CreateMSAARenderTarget( uint32_t aOutputWidth, uint32_t aOutputHeight );
        // void CreateFXAARenderTarget( uint32_t aOutputWidth, uint32_t aOutputHeight );
    };

} // namespace SE::Core