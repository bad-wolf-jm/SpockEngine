#include "NewMaterialSystem.h"

#include "Core/Logging.h"
#include "Core/Profiling/BlockTimer.h"
#include "Core/Resource.h"

#include "Scene/MaterialSystem/MaterialSystem.h"
#include "Scene/Serialize/AssetFile.h"

namespace SE::Core
{
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
    }

    void NewMaterialSystem::SetLights( std::vector<sDirectionalLightData> const &aDirectionalLights )
    {
        mDirectionalLights = aDirectionalLights;
    }

    void NewMaterialSystem::SetLights( std::vector<sPointLightData> const &aPointLights )
    {
        mPointLights = aPointLights;
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
        case eShadingModel::STANDARD:
            DefineConstant( aShaderProgram, "SHADING_MODEL_STANDARD" );
            break;
        case eShadingModel::SUBSURFACE:
            DefineConstant( aShaderProgram, "SHADING_MODEL_SUBSURFACE" );
            break;
        case eShadingModel::CLOTH:
            DefineConstant( aShaderProgram, "SHADING_MODEL_CLOTH" );
            break;
        case eShadingModel::UNLIT:
            DefineConstant( aShaderProgram, "SHADING_MODEL_UNLIT" );
            break;
        }

        if( lMaterialInfo.mRequiresUV0 )
            DefineConstant( aShaderProgram, "MATERIAL_HAS_UV0" );

        if( lMaterialInfo.mRequiresUV1 )
            DefineConstant( aShaderProgram, "MATERIAL_HAS_UV1" );

        if( lMaterialInfo.mRequiresNormals )
            DefineConstant( aShaderProgram, "MATERIAL_HAS_NORMALS" );

        // clang-format off
        DefineConstant<sBaseColorTexture>  ( aShaderProgram, aMaterial, "MATERIAL_HAS_BASE_COLOR_TEXTURE"  );
        DefineConstant<sEmissiveTexture>   ( aShaderProgram, aMaterial, "MATERIAL_HAS_EMISSIVE_TEXTURE"    );
        DefineConstant<sMetalRoughTexture> ( aShaderProgram, aMaterial, "MATERIAL_HAS_METAL_ROUGH_TEXTURE" );
        DefineConstant<sNormalsTexture>    ( aShaderProgram, aMaterial, "MATERIAL_HAS_NORMALS_TEXTURE"     );
        DefineConstant<sOcclusionTexture>  ( aShaderProgram, aMaterial, "MATERIAL_HAS_OCCLUSION_TEXTURE"   );
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

        fs::path lShaderPath = "D:\\Work\\Git\\SpockEngine\\Resources\\Shaders\\Cache";

        std::string lShaderName = CreateShaderName( aMaterial, "vertex_shader" );
        auto        lShader     = CreateShaderProgram( mGraphicContext, eShaderStageTypeFlags::VERTEX, 450, lShaderName, lShaderPath );
        mVertexShaders[GetMaterialHash( aMaterial )] = lShader;

        AddDefinitions( lShader, aMaterial );

        lShader->AddFile( "D:\\Work\\Git\\SpockEngine\\Shaders\\Source\\Renderer2\\Common\\Definitions.hpp" );
        lShader->AddFile( "D:\\Work\\Git\\SpockEngine\\Shaders\\Source\\Renderer2\\Varying.hpp" );

        if( aMaterial.Has<sVertexShader>() )
            lShader->AddCode( "//" );
        else
            lShader->AddCode( "//" );

        lShader->AddFile( "D:\\Work\\Git\\SpockEngine\\Shaders\\Source\\Renderer2\\MainVertexShader.hpp" );

        return lShader;
    }

    Ref<IShaderProgram> NewMaterialSystem::CreateFragmentShader( Material const &aMaterial )
    {

        if( mFragmentShaders.find( GetMaterialHash( aMaterial ) ) != mFragmentShaders.end() )
            return mFragmentShaders[GetMaterialHash( aMaterial )];


        fs::path lShaderPath = "D:\\Work\\Git\\SpockEngine\\Resources\\Shaders\\Cache";

        std::string lShaderName = CreateShaderName( aMaterial, "fragment_shader" );
        auto        lShader     = CreateShaderProgram( mGraphicContext, eShaderStageTypeFlags::FRAGMENT, 450, lShaderName, lShaderPath );
        mFragmentShaders[GetMaterialHash( aMaterial )] = lShader;

        AddDefinitions( lShader, aMaterial );

        lShader->AddFile( "D:\\Work\\Git\\SpockEngine\\Shaders\\Source\\Renderer2\\Common\\Definitions.hpp" );
        lShader->AddFile( "D:\\Work\\Git\\SpockEngine\\Shaders\\Source\\Renderer2\\Varying.hpp" );
        lShader->AddFile( "D:\\Work\\Git\\SpockEngine\\Shaders\\Source\\Renderer2\\Common\\ShaderMaterial.hpp" );
        lShader->AddFile( "D:\\Work\\Git\\SpockEngine\\Shaders\\Source\\Renderer2\\Common\\HelperFunctions.hpp" );
        lShader->AddFile( "D:\\Work\\Git\\SpockEngine\\Shaders\\Source\\Renderer2\\Material.hpp" );

        if( aMaterial.Has<sFragmentShader>() )
            lShader->AddCode( "//" );
        else
            lShader->AddCode( "void material( out MaterialInput aMaterial ) {}" );

        lShader->AddFile( "D:\\Work\\Git\\SpockEngine\\Shaders\\Source\\Renderer2\\MainFragmentShader.hpp" );

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

        if( lMaterialInfo.mIsTwoSided )
            lNewPipeline->SetCulling( eFaceCulling::NONE );
        else
            lNewPipeline->SetCulling( eFaceCulling::BACK );

        lNewPipeline->SetLineWidth( lMaterialInfo.mLineWidth );

        // Descriptors layout for material data
        auto lMaterialDataDescriptorLayout = CreateDescriptorSetLayout( mGraphicContext, true );
        lMaterialDataDescriptorLayout->AddBinding( 0, eDescriptorType::STORAGE_BUFFER, { eShaderStageTypeFlags::FRAGMENT } );
        lMaterialDataDescriptorLayout->Build();
        lNewPipeline->AddDescriptorSet( lMaterialDataDescriptorLayout );

        // Descriptors layout for texture array
        auto lTextureDescriptorLayout = CreateDescriptorSetLayout( mGraphicContext, true );
        lTextureDescriptorLayout->AddBinding( 0, eDescriptorType::COMBINED_IMAGE_SAMPLER, { eShaderStageTypeFlags::FRAGMENT } );
        lTextureDescriptorLayout->Build();
        lNewPipeline->AddDescriptorSet( lTextureDescriptorLayout );

        if( lMaterialInfo.mShadingModel != eShadingModel::UNLIT )
        {
            // Descriptors layout for directional lights
            auto lDirectionalLightDescriptorLayout = CreateDescriptorSetLayout( mGraphicContext, true );
            lDirectionalLightDescriptorLayout->AddBinding( 0, eDescriptorType::STORAGE_BUFFER, { eShaderStageTypeFlags::FRAGMENT } );
            lDirectionalLightDescriptorLayout->Build();
            lNewPipeline->AddDescriptorSet( lDirectionalLightDescriptorLayout );

            // Descriptors layout for pubctual lights
            auto lPunctualLightDescriptorLayout = CreateDescriptorSetLayout( mGraphicContext, true );
            lPunctualLightDescriptorLayout->AddBinding( 0, eDescriptorType::STORAGE_BUFFER, { eShaderStageTypeFlags::FRAGMENT } );
            lPunctualLightDescriptorLayout->Build();
            lNewPipeline->AddDescriptorSet( lPunctualLightDescriptorLayout );
        }

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
        auto &lNew = mMaterialData.emplace_back();

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

            lNew.mEmissiveFactor    = lData.mFactor;
            lNew.mEmissiveUVChannel = lData.mUVChannel;
            lNew.mEmissiveTextureID = AppendTextureData( lData.mTexture );
        }
    }

    void NewMaterialSystem::UpdateMaterialData()
    {
        mMaterialData.clear();
        mTextureData.clear();

        // clang-format off
        mMaterialRegistry.ForEach<sMaterialInfo>( [&]( auto aEntity, auto const &aMaterialInfo )
        { 
            AppendMaterialData( aEntity, aMaterialInfo ); 
        });
        // clang-format on
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

} // namespace SE::Core