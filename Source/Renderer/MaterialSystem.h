#pragma once

#include <unordered_map>

#include "Core/Entity/Collection.h"
#include "Core/Math/Types.h"
#include "Core/Memory.h"

#include "Graphics/API.h"

#include "Common/LightInputData.hpp"
#include "Common/ShaderMaterial.hpp"

#include "Core/Memory.h"

#include "Graphics/Vulkan/DescriptorSet.h"
#include "Graphics/Vulkan/VkGraphicsPipeline.h"
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
        eBlendMode    mType            = eBlendMode::Opaque;
        eShadingModel mShadingModel    = eShadingModel::UNLIT;
        float         mLineWidth       = 1.0f;
        bool          mIsTwoSided      = false;
        bool          mRequiresNormals = true;
        bool          mRequiresUV0     = false;
        bool          mRequiresUV1     = false;

        sMaterialInfo()                        = default;
        sMaterialInfo( const sMaterialInfo & ) = default;
    };

    template <typename _Ty>
    struct sTextureComponent
    {
        using Tex2D = ref_t<ISampler2D>;

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
        string_t mCode;

        sVertexShader()                        = default;
        sVertexShader( const sVertexShader & ) = default;
    };

    struct sFragmentShader
    {
        string_t mCode;

        sFragmentShader()                          = default;
        sFragmentShader( const sFragmentShader & ) = default;
    };

    class MaterialSystem
    {
      public:
        MaterialSystem() = default;
        MaterialSystem( ref_t<IGraphicContext> aGraphicContext );

        ~MaterialSystem() = default;

        Material CreateMaterial( string_t const &aName );
        Material CreateMaterial( fs::path const &aMaterialPath );
        Material BeginMaterial( string_t const &aName );
        void     EndMaterial( Material const &aMaterial );
        size_t   GetMaterialHash( Material const &aMaterial );

        int32_t AppendTextureData( ref_t<ISampler2D> aTexture );
        void    UpdateMaterialData();

        vector_t<Material> GetMaterialData();

      private:
        ref_t<IGraphicContext> mGraphicContext;

        EntityCollection mMaterialRegistry;

        vector_t<ref_t<ISampler2D>> mTextureData;
        vector_t<sShaderMaterial>   mMaterialData;
        sDirectionalLight           mDirectionalLight;
        vector_t<sPunctualLight>    mPointLights;

        std::map<size_t, ref_t<IShaderProgram>> mVertexShaders;
        std::map<size_t, ref_t<IShaderProgram>> mFragmentShaders;

        ref_t<IGraphicBuffer>       mViewParameters                 = nullptr;
        ref_t<IDescriptorSet>       mViewParametersDescriptor       = nullptr;
        ref_t<IDescriptorSetLayout> mViewParametersDescriptorLayout = nullptr;

        ref_t<IGraphicBuffer>       mCameraParameters                 = nullptr;
        ref_t<IDescriptorSet>       mCameraParametersDescriptor       = nullptr;
        ref_t<IDescriptorSetLayout> mCameraParametersDescriptorLayout = nullptr;

        ref_t<IGraphicBuffer>       mShaderMaterials                 = nullptr;
        ref_t<IDescriptorSet>       mShaderMaterialsDescriptor       = nullptr;
        ref_t<IDescriptorSetLayout> mShaderMaterialsDescriptorLayout = nullptr;

        ref_t<IGraphicBuffer>       mShaderDirectionalLights           = nullptr;
        ref_t<IDescriptorSet>       mDirectionalLightsDescriptor       = nullptr;
        ref_t<IDescriptorSetLayout> mDirectionalLightsDescriptorLayout = nullptr;

        ref_t<IGraphicBuffer>       mShaderPunctualLights           = nullptr;
        ref_t<IDescriptorSet>       mPunctualLightsDescriptor       = nullptr;
        ref_t<IDescriptorSetLayout> mPunctualLightsDescriptorLayout = nullptr;

        Cuda::GPUMemory             mMaterialCudaTextures{};
        ref_t<IDescriptorSet>       mMaterialTexturesDescriptor       = nullptr;
        ref_t<IDescriptorSetLayout> mMaterialTexturesDescriptorLayout = nullptr;

        ref_t<ISampler2D>           mShaderDirectionalLightShadowMap           = nullptr;
        ref_t<IDescriptorSet>       mDirectionalLightShadowMapDescriptor       = nullptr;
        ref_t<IDescriptorSetLayout> mDirectionalLightShadowMapDescriptorLayout = nullptr;

        vector_t<ref_t<ISamplerCubeMap>> mPunctualLightShadowMaps                = {};
        ref_t<IDescriptorSet>            mPunctualLightShadowMapDescriptor       = nullptr;
        ref_t<IDescriptorSetLayout>      mPunctualLightShadowMapDescriptorLayout = nullptr;

        std::unordered_map<Material, int32_t> mMaterialIndexLookup;

      public:
        ref_t<IShaderProgram>    CreateVertexShader( Material const &aMaterial );
        ref_t<IShaderProgram>    CreateFragmentShader( Material const &aMaterial );
        ref_t<IGraphicsPipeline> CreateGraphicsPipeline( Material const &aMaterial, ref_t<IRenderContext> aRenderPass );

        ref_t<IGraphicsPipeline> GetGraphicsPipeline( Material const &aMaterial );

        void SetLights( sDirectionalLight const &aDirectionalLights );
        void SetLights( vector_t<sPunctualLight> const &aPointLights );

        void SetShadowMap( ref_t<ISampler2D> aDirectionalShadowMap );
        void SetShadowMap( vector_t<ref_t<ISamplerCubeMap>> aPunctualLightShadowMaps );

        void ConfigureRenderContext( ref_t<IRenderContext> aRenderPass );
        void SetViewParameters( mat4 aProjection, mat4 aView, vec3 aCameraPosition );
        void SetCameraParameters( float aGamma, float aExposure, vec3 aCameraPosition );
        void SelectMaterialInstance( ref_t<IRenderContext> aRenderPass, Material aMaterialID );

      private:
        template <typename _Ty>
        void DefineConstantIfComponentIsPresent( ref_t<IShaderProgram> aShaderProgram, Material aMaterial, const char *aName )
        {
            if( aMaterial.Has<_Ty>() )
                aShaderProgram->AddCode( fmt::format( "#define {}", aName ) );
        }

        void DefineConstant( ref_t<IShaderProgram> aShaderProgram, const char *aName )
        {
            aShaderProgram->AddCode( fmt::format( "#define {}", aName ) );
        }

        void     AddDefinitions( ref_t<IShaderProgram> aShaderProgram, Material aMaterial );
        string_t CreateShaderName( Material aMaterial, const char *aPrefix );

        void    AppendMaterialData( Material aMaterial, sMaterialInfo const &aInfo );
        int32_t GetMaterialIndex( Material aMaterial );
    };

} // namespace SE::Core