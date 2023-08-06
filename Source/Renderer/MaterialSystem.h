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

    class MaterialSystem
    {
      public:
        MaterialSystem() = default;
        MaterialSystem( Ref<IGraphicContext> aGraphicContext );

        ~MaterialSystem() = default;

        Material CreateMaterial( std::string const &aName );
        Material CreateMaterial( fs::path const &aMaterialPath );
        Material BeginMaterial( std::string const &aName );
        void     EndMaterial( Material const &aMaterial );
        size_t   GetMaterialHash( Material const &aMaterial );

        int32_t AppendTextureData( Ref<ISampler2D> aTexture );
        void    UpdateMaterialData();

        std::vector<Material> GetMaterialData();

      private:
        Ref<IGraphicContext> mGraphicContext;

        EntityCollection mMaterialRegistry;

        std::vector<Ref<ISampler2D>> mTextureData;
        std::vector<sShaderMaterial> mMaterialData;
        sDirectionalLight            mDirectionalLight;
        std::vector<sPunctualLight>  mPointLights;

        std::map<size_t, Ref<IShaderProgram>> mVertexShaders;
        std::map<size_t, Ref<IShaderProgram>> mFragmentShaders;

        Ref<IGraphicBuffer>       mViewParameters                 = nullptr;
        Ref<IDescriptorSet>       mViewParametersDescriptor       = nullptr;
        Ref<IDescriptorSetLayout> mViewParametersDescriptorLayout = nullptr;

        Ref<IGraphicBuffer>       mCameraParameters                 = nullptr;
        Ref<IDescriptorSet>       mCameraParametersDescriptor       = nullptr;
        Ref<IDescriptorSetLayout> mCameraParametersDescriptorLayout = nullptr;

        Ref<IGraphicBuffer>       mShaderMaterials                 = nullptr;
        Ref<IDescriptorSet>       mShaderMaterialsDescriptor       = nullptr;
        Ref<IDescriptorSetLayout> mShaderMaterialsDescriptorLayout = nullptr;

        Ref<IGraphicBuffer>       mShaderDirectionalLights           = nullptr;
        Ref<IDescriptorSet>       mDirectionalLightsDescriptor       = nullptr;
        Ref<IDescriptorSetLayout> mDirectionalLightsDescriptorLayout = nullptr;

        Ref<IGraphicBuffer>       mShaderPunctualLights           = nullptr;
        Ref<IDescriptorSet>       mPunctualLightsDescriptor       = nullptr;
        Ref<IDescriptorSetLayout> mPunctualLightsDescriptorLayout = nullptr;

        Cuda::GPUMemory           mMaterialCudaTextures{};
        Ref<IDescriptorSet>       mMaterialTexturesDescriptor       = nullptr;
        Ref<IDescriptorSetLayout> mMaterialTexturesDescriptorLayout = nullptr;

        Ref<ISampler2D>           mShaderDirectionalLightShadowMap           = nullptr;
        Ref<IDescriptorSet>       mDirectionalLightShadowMapDescriptor       = nullptr;
        Ref<IDescriptorSetLayout> mDirectionalLightShadowMapDescriptorLayout = nullptr;

        std::vector<Ref<ISamplerCubeMap>> mPunctualLightShadowMaps                = {};
        Ref<IDescriptorSet>               mPunctualLightShadowMapDescriptor       = nullptr;
        Ref<IDescriptorSetLayout>         mPunctualLightShadowMapDescriptorLayout = nullptr;

        std::unordered_map<Material, int32_t> mMaterialIndexLookup;

      public:
        Ref<IShaderProgram>    CreateVertexShader( Material const &aMaterial );
        Ref<IShaderProgram>    CreateFragmentShader( Material const &aMaterial );
        Ref<IGraphicsPipeline> CreateGraphicsPipeline( Material const &aMaterial, Ref<IRenderContext> aRenderPass );

        Ref<IGraphicsPipeline> GetGraphicsPipeline( Material const &aMaterial );

        void SetLights( sDirectionalLight const &aDirectionalLights );
        void SetLights( std::vector<sPunctualLight> const &aPointLights );

        void SetShadowMap( Ref<ISampler2D> aDirectionalShadowMap );
        void SetShadowMap( std::vector<Ref<ISamplerCubeMap>> aPunctualLightShadowMaps );

        void ConfigureRenderContext( Ref<IRenderContext> aRenderPass );
        void SetViewParameters( mat4 aProjection, mat4 aView, vec3 aCameraPosition );
        void SetCameraParameters( float aGamma, float aExposure, vec3 aCameraPosition );
        void SelectMaterialInstance( Ref<IRenderContext> aRenderPass, Material aMaterialID );

      private:
        template <typename _Ty>
        void DefineConstantIfComponentIsPresent( Ref<IShaderProgram> aShaderProgram, Material aMaterial, const char *aName )
        {
            if( aMaterial.Has<_Ty>() )
                aShaderProgram->AddCode( fmt::format( "#define {}", aName ) );
        }

        void DefineConstant( Ref<IShaderProgram> aShaderProgram, const char *aName )
        {
            aShaderProgram->AddCode( fmt::format( "#define {}", aName ) );
        }

        void        AddDefinitions( Ref<IShaderProgram> aShaderProgram, Material aMaterial );
        std::string CreateShaderName( Material aMaterial, const char *aPrefix );

        void    AppendMaterialData( Material aMaterial, sMaterialInfo const &aInfo );
        int32_t GetMaterialIndex( Material aMaterial );
    };

} // namespace SE::Core