#include "MaterialSystem.h"

#include "Core/Logging.h"
#include "Core/Profiling/BlockTimer.h"
#include "Core/Resource.h"

#include "Scene/MaterialSystem/MaterialSystem.h"
#include "Scene/Serialize/AssetFile.h"

#include "glslang/Include/glslang_c_interface.h"
#include "glslang/Include/glslang_c_shader_types.h"
#include "glslang/Public/resource_limits_c.h"
#include <fstream>

namespace SE::Core
{
    struct ViewParameters
    {
        mat4 mProjection;
        mat4 mView;
        vec3 mCameraPosition;

        ViewParameters()  = default;
        ~ViewParameters() = default;

        ViewParameters( const ViewParameters & ) = default;
    };

    struct CameraParameters
    {
        float mExposure = 4.5f;
        float mGamma    = 2.2f;
        ALIGN( 16 ) vec3 mPosition{};

        CameraParameters()  = default;
        ~CameraParameters() = default;

        CameraParameters( const CameraParameters & ) = default;
    };

    struct sMaterialNotReady
    {
    };

    struct sMaterialNeedsUpdate
    {
        bool x = true;
    };

    NewMaterialSystem::NewMaterialSystem( Ref<IGraphicContext> aGraphicContext )
        : mGraphicContext{ aGraphicContext }
    {
        mViewParameters =
            CreateBuffer( mGraphicContext, eBufferType::UNIFORM_BUFFER, true, true, true, true, sizeof( ViewParameters ) );
        mViewParametersDescriptorLayout = CreateDescriptorSetLayout( mGraphicContext );
        mViewParametersDescriptorLayout->AddBinding( 0, eDescriptorType::UNIFORM_BUFFER, { eShaderStageTypeFlags::VERTEX } );
        mViewParametersDescriptorLayout->Build();

        mCameraParameters =
            CreateBuffer( mGraphicContext, eBufferType::UNIFORM_BUFFER, true, true, true, true, sizeof( CameraParameters ) );
        mCameraParametersDescriptorLayout = CreateDescriptorSetLayout( mGraphicContext );
        mCameraParametersDescriptorLayout->AddBinding( 0, eDescriptorType::UNIFORM_BUFFER, { eShaderStageTypeFlags::FRAGMENT } );
        mCameraParametersDescriptorLayout->Build();

        // Descriptors layout for material data
        mShaderMaterials =
            CreateBuffer( mGraphicContext, eBufferType::STORAGE_BUFFER, true, true, true, true, sizeof( sShaderMaterial ) );
        mShaderMaterialsDescriptorLayout = CreateDescriptorSetLayout( mGraphicContext, true );
        mShaderMaterialsDescriptorLayout->AddBinding( 0, eDescriptorType::STORAGE_BUFFER, { eShaderStageTypeFlags::FRAGMENT } );
        mShaderMaterialsDescriptorLayout->Build();

        // Descriptors layout for texture array
        mMaterialTexturesDescriptorLayout = CreateDescriptorSetLayout( mGraphicContext, true );
        mMaterialTexturesDescriptorLayout->AddBinding( 0, eDescriptorType::COMBINED_IMAGE_SAMPLER,
                                                       { eShaderStageTypeFlags::FRAGMENT } );
        mMaterialTexturesDescriptorLayout->Build();

        // Descriptors layout for directional lights
        mShaderDirectionalLights = mShaderPunctualLights =
            CreateBuffer( mGraphicContext, eBufferType::UNIFORM_BUFFER, true, true, true, true, sizeof( sDirectionalLight ) );
        mDirectionalLightsDescriptorLayout = CreateDescriptorSetLayout( mGraphicContext );
        mDirectionalLightsDescriptorLayout->AddBinding( 0, eDescriptorType::UNIFORM_BUFFER, { eShaderStageTypeFlags::FRAGMENT } );
        mDirectionalLightsDescriptorLayout->Build();

        // Descriptors layout for pun ctual lights

        mPunctualLightsDescriptorLayout = CreateDescriptorSetLayout( mGraphicContext, true );
        mPunctualLightsDescriptorLayout->AddBinding( 0, eDescriptorType::STORAGE_BUFFER, { eShaderStageTypeFlags::FRAGMENT } );
        mPunctualLightsDescriptorLayout->Build();

        mShaderPunctualLights =
            CreateBuffer( mGraphicContext, eBufferType::STORAGE_BUFFER, true, true, true, true, sizeof( sPunctualLight ) );
        mPunctualLightsDescriptor = mPunctualLightsDescriptorLayout->Allocate( 1 );
        mPunctualLightsDescriptor->Write( mShaderPunctualLights, false, 0, 1, 0 );

        // Shadow maps for directional and punctual lights
        mDirectionalLightShadowMapDescriptorLayout = CreateDescriptorSetLayout( aGraphicContext );
        mDirectionalLightShadowMapDescriptorLayout->AddBinding( 0, eDescriptorType::COMBINED_IMAGE_SAMPLER,
                                                                { eShaderStageTypeFlags::FRAGMENT } );
        mDirectionalLightShadowMapDescriptorLayout->Build();
        mDirectionalLightShadowMapDescriptor = mDirectionalLightShadowMapDescriptorLayout->Allocate();

        mPunctualLightShadowMapDescriptorLayout = CreateDescriptorSetLayout( aGraphicContext, true );
        mPunctualLightShadowMapDescriptorLayout->AddBinding( 0, eDescriptorType::COMBINED_IMAGE_SAMPLER,
                                                             { eShaderStageTypeFlags::FRAGMENT } );
        mPunctualLightShadowMapDescriptorLayout->Build();
    }

    std::vector<Material> NewMaterialSystem::GetMaterialData()
    {
        std::vector<Material> lMaterials;
        mMaterialRegistry.ForEach<sMaterialInfo>( [&]( auto aMaterial, auto const &aInfo ) { lMaterials.push_back( aMaterial ); } );

        return lMaterials;
    }

    void NewMaterialSystem::SetLights( sDirectionalLight const &aDirectionalLights )
    {
        mDirectionalLight = aDirectionalLights;
        mShaderDirectionalLights->Write( mDirectionalLight );
    }

    void NewMaterialSystem::SetLights( std::vector<sPunctualLight> const &aPointLights )
    {
        mPointLights = aPointLights;

        if( mShaderPunctualLights->SizeAs<sPunctualLight>() != mPointLights.size() )
        {
            auto lBufferSize      = std::max( mPointLights.size(), static_cast<size_t>( 1 ) ) * sizeof( sPunctualLight );
            mShaderPunctualLights = CreateBuffer( mGraphicContext, eBufferType::STORAGE_BUFFER, true, false, true, true, lBufferSize );
            mPunctualLightsDescriptor = mPunctualLightsDescriptorLayout->Allocate( 1 );
            mPunctualLightsDescriptor->Write( mShaderPunctualLights, false, 0, lBufferSize, 0 );
        }

        mShaderPunctualLights->Upload( mPointLights );
    }

    Material NewMaterialSystem::CreateMaterial( std::string const &aName )
    {
        Material lNewMaterial = mMaterialRegistry.CreateEntity( aName );
        lNewMaterial.Add<sMaterialInfo>();

        return lNewMaterial;
    }

    Material NewMaterialSystem::BeginMaterial( std::string const &aName )
    {
        auto lNewMaterial = CreateMaterial( aName );
        lNewMaterial.Add<sMaterialNotReady>();

        return lNewMaterial;
    }

    void NewMaterialSystem::EndMaterial( Material const &aMaterial )
    {
        aMaterial.Remove<sMaterialNotReady>();
        aMaterial.Add<sMaterialNeedsUpdate>();
    }

    size_t NewMaterialSystem::GetMaterialHash( Material const &aMaterial )
    {
        uint8_t lBitOffset = 0;
        size_t  lHashValue = 0;

        // clang-format off
        lHashValue |= aMaterial.Has<sBaseColorTexture> () << lBitOffset++;
        lHashValue |= aMaterial.Has<sEmissiveTexture>()   << lBitOffset++;
        lHashValue |= aMaterial.Has<sMetalRoughTexture>() << lBitOffset++;
        lHashValue |= aMaterial.Has<sNormalsTexture>()    << lBitOffset++;
        lHashValue |= aMaterial.Has<sOcclusionTexture>()  << lBitOffset++;
        lHashValue |= aMaterial.Has<sVertexShader>()      << lBitOffset++;
        lHashValue |= aMaterial.Has<sFragmentShader>()    << lBitOffset++;

        auto const& lMaterialInfo = aMaterial.Get<sMaterialInfo>();
        lHashValue |= lMaterialInfo.mRequiresUV0           << lBitOffset++;
        lHashValue |= lMaterialInfo.mRequiresUV1           << lBitOffset++;
        lHashValue |= lMaterialInfo.mRequiresNormals       << lBitOffset++;
        lHashValue |= lMaterialInfo.mIsTwoSided            << lBitOffset++; lBitOffset++;
        lHashValue |= (uint8_t)lMaterialInfo.mShadingModel << lBitOffset++; lBitOffset++;
        lHashValue |= (uint8_t)lMaterialInfo.mType         << lBitOffset;
        // clang-format on

        return lHashValue;
    }

    void NewMaterialSystem::AddDefinitions( Ref<IShaderProgram> aShaderProgram, Material aMaterial )
    {
        DefineConstant( aShaderProgram, "__GLSL__" );
        DefineConstant( aShaderProgram, "VULKAN_SEMANTICS" );

        auto const &lMaterialInfo = aMaterial.Get<sMaterialInfo>();

        switch( lMaterialInfo.mShadingModel )
        {
        case eShadingModel::SUBSURFACE:
            DefineConstant( aShaderProgram, "SHADING_MODEL_SUBSURFACE" );
            break;
        case eShadingModel::CLOTH:
            DefineConstant( aShaderProgram, "SHADING_MODEL_CLOTH" );
            break;
        case eShadingModel::STANDARD:
        default:
            DefineConstant( aShaderProgram, "SHADING_MODEL_STANDARD" );
            break;
        }

        if( lMaterialInfo.mRequiresUV0 )
            DefineConstant( aShaderProgram, "MATERIAL_HAS_UV0" );

        if( lMaterialInfo.mRequiresUV1 )
            DefineConstant( aShaderProgram, "MATERIAL_HAS_UV1" );

        if( lMaterialInfo.mRequiresNormals )
            DefineConstant( aShaderProgram, "MATERIAL_HAS_NORMALS" );

        // clang-format off
        DefineConstantIfComponentIsPresent<sBaseColorTexture>  ( aShaderProgram, aMaterial, "MATERIAL_HAS_BASE_COLOR_TEXTURE"  );
        DefineConstantIfComponentIsPresent<sEmissiveTexture>   ( aShaderProgram, aMaterial, "MATERIAL_HAS_EMISSIVE_TEXTURE"    );
        DefineConstantIfComponentIsPresent<sMetalRoughTexture> ( aShaderProgram, aMaterial, "MATERIAL_HAS_METAL_ROUGH_TEXTURE" );
        DefineConstantIfComponentIsPresent<sNormalsTexture>    ( aShaderProgram, aMaterial, "MATERIAL_HAS_NORMALS_TEXTURE"     );
        DefineConstantIfComponentIsPresent<sOcclusionTexture>  ( aShaderProgram, aMaterial, "MATERIAL_HAS_OCCLUSION_TEXTURE"   );
        // clang-format on
    }

    std::string NewMaterialSystem::CreateShaderName( Material aMaterial, const char *aPrefix )
    {
        std::string lMateriaName = aMaterial.TryGet<sTag>( sTag{} ).mValue;
        if( !lMateriaName.empty() )
            return fmt::format( "{}_{}_{}", aPrefix, lMateriaName, GetMaterialHash( aMaterial ) );
        else
            return fmt::format( "{}_UNNAMED_{}", aPrefix, lMateriaName, GetMaterialHash( aMaterial ) );
    }

    Ref<IShaderProgram> NewMaterialSystem::CreateVertexShader( Material const &aMaterial )
    {
        if( mVertexShaders.find( GetMaterialHash( aMaterial ) ) != mVertexShaders.end() )
            return mVertexShaders[GetMaterialHash( aMaterial )];

        std::string lShaderName = CreateShaderName( aMaterial, "vertex_shader" );
        auto        lShader     = CreateShaderProgram( mGraphicContext, eShaderStageTypeFlags::VERTEX, 450, lShaderName );
        mVertexShaders[GetMaterialHash( aMaterial )] = lShader;

        AddDefinitions( lShader, aMaterial );

        lShader->AddFile( "D:\\Work\\Git\\SpockEngine\\Shaders\\Source\\Common\\Definitions.hpp" );
        lShader->AddFile( "D:\\Work\\Git\\SpockEngine\\Shaders\\Source\\Common\\Varying.hpp" );

        if( aMaterial.Has<sVertexShader>() )
            lShader->AddCode( "//" );
        else
            lShader->AddCode( "//" );

        lShader->AddFile( "D:\\Work\\Git\\SpockEngine\\Shaders\\Source\\MainVertexShader.hpp" );

        return lShader;
    }

    Ref<IShaderProgram> NewMaterialSystem::CreateFragmentShader( Material const &aMaterial )
    {

        if( mFragmentShaders.find( GetMaterialHash( aMaterial ) ) != mFragmentShaders.end() )
            return mFragmentShaders[GetMaterialHash( aMaterial )];

        std::string lShaderName = CreateShaderName( aMaterial, "fragment_shader" );
        auto        lShader     = CreateShaderProgram( mGraphicContext, eShaderStageTypeFlags::FRAGMENT, 450, lShaderName );
        mFragmentShaders[GetMaterialHash( aMaterial )] = lShader;

        lShader->AddCode( "#extension GL_EXT_nonuniform_qualifier : enable" );

        AddDefinitions( lShader, aMaterial );
        lShader->AddFile( "D:\\Work\\Git\\SpockEngine\\Shaders\\Source\\Common\\Definitions.hpp" );
        lShader->AddFile( "D:\\Work\\Git\\SpockEngine\\Shaders\\Source\\Common\\Varying.hpp" );
        lShader->AddFile( "D:\\Work\\Git\\SpockEngine\\Shaders\\Source\\Common\\LightInputData.hpp" );
        lShader->AddFile( "D:\\Work\\Git\\SpockEngine\\Shaders\\Source\\Common\\ShaderMaterial.hpp" );
        lShader->AddFile( "D:\\Work\\Git\\SpockEngine\\Shaders\\Source\\FragmentShaderUniformInputs.hpp" );
        lShader->AddFile( "D:\\Work\\Git\\SpockEngine\\Shaders\\Source\\Common\\HelperFunctions.hpp" );
        lShader->AddFile( "D:\\Work\\Git\\SpockEngine\\Shaders\\Source\\Material.hpp" );
        lShader->AddFile( "D:\\Work\\Git\\SpockEngine\\Shaders\\Source\\ShadingData.hpp" );
        lShader->AddFile( "D:\\Work\\Git\\SpockEngine\\Shaders\\Source\\LightData.hpp" );

        auto const &lMaterialInfo = aMaterial.Get<sMaterialInfo>();

        lShader->AddFile( "D:\\Work\\Git\\SpockEngine\\Shaders\\Source\\ShadingModels\\Brdf.hpp" );

        switch( lMaterialInfo.mShadingModel )
        {
        case eShadingModel::SUBSURFACE:
            lShader->AddFile( "D:\\Work\\Git\\SpockEngine\\Shaders\\Source\\ShadingModels\\SurfaceShadingSubsurface.hpp" );
            break;
        case eShadingModel::CLOTH:
            lShader->AddFile( "D:\\Work\\Git\\SpockEngine\\Shaders\\Source\\ShadingModels\\SurfaceShadingCloth.hpp" );
            break;
        case eShadingModel::STANDARD:
        default:
            lShader->AddFile( "D:\\Work\\Git\\SpockEngine\\Shaders\\Source\\ShadingModels\\SurfaceShadingStandard.hpp" );
            break;
        }

        lShader->AddFile( "D:\\Work\\Git\\SpockEngine\\Shaders\\Source\\ShadingModels\\ShadingModelLit.hpp" );

        if( aMaterial.Has<sFragmentShader>() )
            lShader->AddCode( "//" );
        else
            lShader->AddCode( "void material( inout MaterialInputs aMaterial ) {}" );

        lShader->AddFile( "D:\\Work\\Git\\SpockEngine\\Shaders\\Source\\MainFragmentShader.hpp" );

        return lShader;
    }

    Ref<IGraphicsPipeline> NewMaterialSystem::CreateGraphicsPipeline( Material const &aMaterial, Ref<IRenderContext> aRenderPass )
    {
        auto lVertexShader   = CreateVertexShader( aMaterial );
        auto lFragmentShader = CreateFragmentShader( aMaterial );
        auto lNewPipeline    = SE::Graphics::CreateGraphicsPipeline( mGraphicContext, aRenderPass, ePrimitiveTopology::TRIANGLES );

        lVertexShader->Compile();
        lFragmentShader->Compile();

        lNewPipeline->SetShader( eShaderStageTypeFlags::VERTEX, lVertexShader, "main" );
        lNewPipeline->SetShader( eShaderStageTypeFlags::FRAGMENT, lFragmentShader, "main" );

        auto lMaterialInfo = aMaterial.Get<sMaterialInfo>();

        lNewPipeline->SetDepthParameters( true, true, eDepthCompareOperation::LESS_OR_EQUAL );

        lNewPipeline->AddInput( "Position", eBufferDataType::VEC3, 0, 0 );

        if( lMaterialInfo.mRequiresNormals )
            lNewPipeline->AddInput( "Normal", eBufferDataType::VEC3, 0, 1 );

        if( lMaterialInfo.mRequiresUV0 && !lMaterialInfo.mRequiresUV1 )
            lNewPipeline->AddInput( "UV", eBufferDataType::VEC2, 0, 2 );
        else if( lMaterialInfo.mRequiresUV1 )
            lNewPipeline->AddInput( "UV", eBufferDataType::VEC4, 0, 2 );

        lNewPipeline->AddInput( "Bones", eBufferDataType::VEC4, 0, 3 );
        lNewPipeline->AddInput( "Weights", eBufferDataType::VEC4, 0, 4 );

        if( lMaterialInfo.mIsTwoSided )
            lNewPipeline->SetCulling( eFaceCulling::NONE );
        else
            lNewPipeline->SetCulling( eFaceCulling::BACK );

        lNewPipeline->SetLineWidth( lMaterialInfo.mLineWidth );

        lNewPipeline->AddDescriptorSet( mViewParametersDescriptorLayout );
        lNewPipeline->AddDescriptorSet( mCameraParametersDescriptorLayout );
        lNewPipeline->AddDescriptorSet( mShaderMaterialsDescriptorLayout );
        lNewPipeline->AddDescriptorSet( mMaterialTexturesDescriptorLayout );
        lNewPipeline->AddDescriptorSet( mDirectionalLightsDescriptorLayout );
        lNewPipeline->AddDescriptorSet( mPunctualLightsDescriptorLayout );
        lNewPipeline->AddDescriptorSet( mDirectionalLightShadowMapDescriptorLayout );
        lNewPipeline->AddDescriptorSet( mPunctualLightShadowMapDescriptorLayout );

        lNewPipeline->AddPushConstantRange( { eShaderStageTypeFlags::FRAGMENT }, 0, sizeof( int32_t ) );

        lNewPipeline->Build();

        return lNewPipeline;
    }

    int32_t NewMaterialSystem::AppendTextureData( Ref<ISampler2D> aTexture )
    {
        mTextureData.push_back( aTexture );

        return mTextureData.size() - 1;
    }

    void NewMaterialSystem::AppendMaterialData( Material aMaterial, sMaterialInfo const &aInfo )
    {
        auto &lNew                      = mMaterialData.emplace_back();
        mMaterialIndexLookup[aMaterial] = mMaterialData.size() - 1;

        if( aMaterial.Has<sBaseColorTexture>() )
        {
            auto &lData = aMaterial.Get<sBaseColorTexture>();

            lNew.mBaseColorFactor    = lData.mFactor;
            lNew.mBaseColorUVChannel = lData.mUVChannel;
            lNew.mBaseColorTextureID = AppendTextureData( lData.mTexture );
        }

        if( aMaterial.Has<sMetalRoughTexture>() )
        {
            auto &lData = aMaterial.Get<sMetalRoughTexture>();

            lNew.mMetallicFactor     = lData.mMetallicFactor;
            lNew.mRoughnessFactor    = lData.mRoughnessFactor;
            lNew.mMetalnessUVChannel = lData.mUVChannel;
            lNew.mMetalnessTextureID = AppendTextureData( lData.mTexture );
        }

        if( aMaterial.Has<sNormalsTexture>() )
        {
            auto &lData = aMaterial.Get<sNormalsTexture>();

            lNew.mNormalUVChannel = lData.mUVChannel;
            lNew.mNormalTextureID = AppendTextureData( lData.mTexture );
        }

        if( aMaterial.Has<sOcclusionTexture>() )
        {
            auto &lData = aMaterial.Get<sOcclusionTexture>();

            lNew.mOcclusionStrength  = lData.mFactor;
            lNew.mOcclusionUVChannel = lData.mUVChannel;
            lNew.mOcclusionTextureID = AppendTextureData( lData.mTexture );
        }
        if( aMaterial.Has<sEmissiveTexture>() )
        {
            auto &lData = aMaterial.Get<sEmissiveTexture>();

            lNew.mEmissiveFactor    = math::vec4( lData.mFactor, 0.0f );
            lNew.mEmissiveUVChannel = lData.mUVChannel;
            lNew.mEmissiveTextureID = AppendTextureData( lData.mTexture );
        }
    }

    void NewMaterialSystem::UpdateMaterialData()
    {
        mMaterialData.clear();
        mTextureData.clear();

        mViewParametersDescriptor = mViewParametersDescriptorLayout->Allocate();
        mViewParametersDescriptor->Write( mViewParameters, false, 0, sizeof( ViewParameters ), 0 );

        mCameraParametersDescriptor = mCameraParametersDescriptorLayout->Allocate();
        mCameraParametersDescriptor->Write( mCameraParameters, false, 0, sizeof( CameraParameters ), 0 );

        mDirectionalLightsDescriptor = mDirectionalLightsDescriptorLayout->Allocate();
        mDirectionalLightsDescriptor->Write( mShaderDirectionalLights, false, 0, sizeof( sDirectionalLight ), 0 );

        // clang-format off
        mMaterialRegistry.ForEach<sMaterialInfo>( [&]( auto aEntity, auto const &aMaterialInfo )
        { 
            AppendMaterialData( aEntity, aMaterialInfo ); 
        });
        // clang-format on

        mShaderMaterialsDescriptor = mShaderMaterialsDescriptorLayout->Allocate( 1 );
        if( mShaderMaterials->SizeAs<sShaderMaterial>() < mMaterialData.size() )
        {
            auto lBufferSize = std::max( mMaterialData.size(), static_cast<size_t>( 1 ) ) * sizeof( sShaderMaterial );
            mShaderMaterials = CreateBuffer( mGraphicContext, eBufferType::STORAGE_BUFFER, true, false, true, true, lBufferSize );
            mShaderMaterialsDescriptor->Write( mShaderMaterials, false, 0, lBufferSize, 0 );
        }
        mShaderMaterials->Upload( mMaterialData );

        mMaterialTexturesDescriptor = mMaterialTexturesDescriptorLayout->Allocate( mTextureData.size() );
        mMaterialTexturesDescriptor->Write( mTextureData, 0 );

        if( mMaterialCudaTextures.SizeAs<Cuda::TextureSampler2D::DeviceData>() < mTextureData.size() )
        {
            mMaterialCudaTextures.Dispose();
            std::vector<Cuda::TextureSampler2D::DeviceData> lTextureDeviceData{};
            for( auto const &lCudaTextureSampler : mTextureData )
                lTextureDeviceData.push_back( lCudaTextureSampler->mDeviceData );

            mMaterialCudaTextures = Cuda::GPUMemory::Create( lTextureDeviceData );
        }
    }

    Material NewMaterialSystem::CreateMaterial( fs::path const &aMaterialPath )
    {
        SE::Logging::Info( "NewMaterialSystem::CreateMaterial( {} )", aMaterialPath.string() );
        BinaryAsset lBinaryDataFile( aMaterialPath );

        uint32_t lTextureCount = lBinaryDataFile.CountAssets() - 1;

        sMaterial lMaterialData;
        lBinaryDataFile.Retrieve( 0, lMaterialData );

        std::vector<Ref<ISampler2D>> lTextures{};
        for( uint32_t i = 0; i < lTextureCount; i++ )
        {
            auto &[lTextureData, lTextureSampler] = lBinaryDataFile.Retrieve( i + 1 );

            auto lNewInteropTexture = CreateTexture2D( mGraphicContext, lTextureData, 1, false, false, true );
            auto lNewInteropSampler = CreateSampler2D( mGraphicContext, lNewInteropTexture, lTextureSampler.mSamplingSpec );

            lTextures.push_back( lNewInteropSampler );
        }

        auto lNewMaterial = BeginMaterial( lMaterialData.mName );

        auto &lMaterialInfo            = lNewMaterial.Get<sMaterialInfo>();
        lMaterialInfo.mType            = eBlendMode::Opaque;
        lMaterialInfo.mShadingModel    = eShadingModel::STANDARD;
        lMaterialInfo.mLineWidth       = lMaterialData.mLineWidth;
        lMaterialInfo.mIsTwoSided      = lMaterialData.mIsTwoSided;
        lMaterialInfo.mRequiresNormals = true;
        lMaterialInfo.mRequiresUV0     = true;
        lMaterialInfo.mRequiresUV1     = false;

        if( lMaterialData.mBaseColorTexture.mTextureID < std::numeric_limits<uint32_t>::max() && ( lTextureCount > 0 ) )
        {
            auto &lBaseColor    = lNewMaterial.Add<sBaseColorTexture>();
            lBaseColor.mFactor  = lMaterialData.mBaseColorFactor;
            lBaseColor.mTexture = lTextures[lMaterialData.mBaseColorTexture.mTextureID];
        }

        if( lMaterialData.mEmissiveTexture.mTextureID < std::numeric_limits<uint32_t>::max() && ( lTextureCount > 0 ) )
        {
            auto &lEmissive    = lNewMaterial.Add<sEmissiveTexture>();
            lEmissive.mFactor  = lMaterialData.mEmissiveFactor;
            lEmissive.mTexture = lTextures[lMaterialData.mEmissiveTexture.mTextureID];
        }

        if( lMaterialData.mMetalRoughTexture.mTextureID < std::numeric_limits<uint32_t>::max() && ( lTextureCount > 0 ) )
        {
            auto &lMetalRough            = lNewMaterial.Add<sMetalRoughTexture>();
            lMetalRough.mMetallicFactor  = lMaterialData.mMetallicFactor;
            lMetalRough.mRoughnessFactor = lMaterialData.mRoughnessFactor;
            lMetalRough.mTexture         = lTextures[lMaterialData.mMetalRoughTexture.mTextureID];
        }

        if( lMaterialData.mOcclusionTexture.mTextureID < std::numeric_limits<uint32_t>::max() && ( lTextureCount > 0 ) )
        {
            auto &lOcclusion    = lNewMaterial.Add<sOcclusionTexture>();
            lOcclusion.mFactor  = lMaterialData.mOcclusionStrength;
            lOcclusion.mTexture = lTextures[lMaterialData.mOcclusionTexture.mTextureID];
        }

        if( lMaterialData.mNormalsTexture.mTextureID < std::numeric_limits<uint32_t>::max() && ( lTextureCount > 0 ) )
        {
            auto &lNormals    = lNewMaterial.Add<sNormalsTexture>();
            lNormals.mFactor  = math::vec3( 1.0 );
            lNormals.mTexture = lTextures[lMaterialData.mNormalsTexture.mTextureID];
        }

        EndMaterial( lNewMaterial );

        return lNewMaterial;
    }

    void NewMaterialSystem::ConfigureRenderContext( Ref<IRenderContext> aRenderPass )
    {
        aRenderPass->Bind( mViewParametersDescriptor, VIEW_PARAMETERS_BIND_POINT );
        aRenderPass->Bind( mCameraParametersDescriptor, CAMERA_PARAMETERS_BIND_POINT );
        aRenderPass->Bind( mShaderMaterialsDescriptor, MATERIAL_DATA_BIND_POINT );
        aRenderPass->Bind( mMaterialTexturesDescriptor, MATERIAL_TEXTURES_BIND_POINT );
        aRenderPass->Bind( mDirectionalLightsDescriptor, DIRECTIONAL_LIGHTS_BIND_POINT );
        aRenderPass->Bind( mPunctualLightsDescriptor, PUNCTUAL_LIGHTS_BIND_POINT );
        aRenderPass->Bind( mDirectionalLightShadowMapDescriptor, DIRECTIONAL_LIGHTS_SHADOW_MAP_BIND_POINT );
        aRenderPass->Bind( mPunctualLightShadowMapDescriptor, PUNCTUAL_LIGHTS_SHADOW_MAP_BIND_POINT );
    }

    void NewMaterialSystem::SetViewParameters( mat4 aProjection, mat4 aView, vec3 aCameraPosition )
    {
        ViewParameters lView{ aProjection, aView, aCameraPosition };
        mViewParameters->Write( lView );
    }

    void NewMaterialSystem::SetCameraParameters( float aGamma, float aExposure, vec3 aCameraPosition )
    {
        CameraParameters lCamera{ aExposure, aGamma, aCameraPosition };
        mCameraParameters->Write( lCamera );
    }

    int32_t NewMaterialSystem::GetMaterialIndex( Material aMaterial )
    {
        if( mMaterialIndexLookup.find( aMaterial ) == mMaterialIndexLookup.end() )
            return -1;

        return mMaterialIndexLookup[aMaterial];
    }

    void NewMaterialSystem::SelectMaterialInstance( Ref<IRenderContext> aRenderPass, Material aMaterialID )
    {
        aRenderPass->PushConstants( { eShaderStageTypeFlags::FRAGMENT }, 0, GetMaterialIndex( aMaterialID ) );
    }

    void NewMaterialSystem::SetShadowMap( Ref<ISampler2D> aDirectionalShadowMap )
    {
        mDirectionalLightShadowMapDescriptor->Write( aDirectionalShadowMap, 0 );
    }

    void NewMaterialSystem::SetShadowMap( std::vector<Ref<ISamplerCubeMap>> aPunctualLightShadowMaps )
    {
        mPunctualLightShadowMapDescriptor = mPunctualLightShadowMapDescriptorLayout->Allocate( aPunctualLightShadowMaps.size() );
        mPunctualLightShadowMapDescriptor->Write( aPunctualLightShadowMaps, 0 );
    }

} // namespace SE::Core