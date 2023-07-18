#include "NewMaterialSystem.h"

#include "Core/Logging.h"
#include "Core/Profiling/BlockTimer.h"
#include "Core/Resource.h"

namespace SE::Core
{
    struct sMaterialNotReady
    {
    };

    struct sMaterialNeedsUpdate
    {
    };

    NewMaterialSystem::NewMaterialSystem( Ref<IGraphicContext> aGraphicContext )
        : mGraphicContext{ aGraphicContext }
    {
    }

    Material NewMaterialSystem::CreateMaterial( std::string const &aName )
    {
        return mMaterialRegistry.CreateEntity( aName );
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

        auto const& lMaterialInfo = aMaterial.Get<sMaterialInfo>();
        lHashValue |= lMaterialInfo.mHasUV1                << lBitOffset++;
        lHashValue |= lMaterialInfo.mIsTwoSided            << lBitOffset++; lBitOffset++;
        lHashValue |= (uint8_t)lMaterialInfo.mShadingModel << lBitOffset++; lBitOffset++;
        lHashValue |= (uint8_t)lMaterialInfo.mType         << lBitOffset;
        // clang-format on

        return lHashValue;
    }

    template <typename _Ty>
    void DefineConstant( Ref<IShaderProgram> aShaderProgram, Material aMaterial, const char *aName )
    {
        if( aMaterial.Has<_Ty>() )
            aShaderProgram->AddCode( fmt::format( "#define {}", aName ) );
    }

    static void AddDefinitions( Ref<IShaderProgram> aShaderProgram, Material aMaterial )
    {
        aShaderProgram->AddCode( "#define VULKAN_SEMANTICS" );

        auto const &lMaterialInfo = aMaterial.Get<sMaterialInfo>();

        switch( lMaterialInfo.mShadingModel )
        {
        case eShadingModel::STANDARD:
            aShaderProgram->AddCode( "#define SHADING_MODEL_STANDARD" );
            break;
        case eShadingModel::SUBSURFACE:
            aShaderProgram->AddCode( "#define SHADING_MODEL_SUBSURFACE" );
            break;
        case eShadingModel::CLOTH:
            aShaderProgram->AddCode( "#define SHADING_MODEL_CLOTH" );
            break;
        case eShadingModel::UNLIT:
            aShaderProgram->AddCode( "#define SHADING_MODEL_UNLIT" );
            break;
        }

        if( lMaterialInfo.mHasUV1 )
            aShaderProgram->AddCode( "#define MATERIAL_HAS_UV1" );

        // clang-format off
        DefineConstant<sBaseColorTexture>  ( aShaderProgram, aMaterial, "MATERIAL_HAS_BASE_COLOR_TEXTURE"  );
        DefineConstant<sEmissiveTexture>   ( aShaderProgram, aMaterial, "MATERIAL_HAS_EMISSIVE_TEXTURE"    );
        DefineConstant<sMetalRoughTexture> ( aShaderProgram, aMaterial, "MATERIAL_HAS_METAL_ROUGH_TEXTURE" );
        DefineConstant<sNormalsTexture>    ( aShaderProgram, aMaterial, "MATERIAL_HAS_NORMALS_TEXTURE"     );
        DefineConstant<sOcclusionTexture>  ( aShaderProgram, aMaterial, "MATERIAL_HAS_OCCLUSION_TEXTURE"   );
        // clang-format on
    }

    static std::string CreateShaderName( Material aMaterial, const char *aPrefix )
    {
        std::string lMateriaName = aMaterial.TryGet<sTag>( sTag{} ).mValue;
        if( !lMateriaName.empty() )
            return fmt::format( "{}_{}_{}", aPrefix, lMateriaName, GetMaterialHash( aMaterial ) );
        else
            return fmt::format( "{}_UNNAMED_{}", aPrefix, lMateriaName, GetMaterialHash( aMaterial ) );
    }

    Ref<IShaderProgram> NewMaterialSystem::CreateVertexShader( Material aMaterial )
    {
        fs::path lShaderPath = "D:\\Work\\Git\\SpockEngine\\Resources\\Shaders\\Cache";

        std::string lShaderName = CreateShaderName( aMaterial, "vertex_shader" );
        auto lShader      = CreateShaderProgram( mGraphicContext, eShaderStageTypeFlags::VERTEX, 450, lShaderName, lShaderPath );

        AddDefinitions( lShader, aMaterial );

        lShader->AddFile( "D:\\Work\\Git\\SpockEngine\\Shaders\\Source\\Renderer2\\Common\\Definitions.hpp" );
        lShader->AddFile( "D:\\Work\\Git\\SpockEngine\\Shaders\\Source\\Renderer2\\Varying.hpp" );
        lShader->AddFile( "D:\\Work\\Git\\SpockEngine\\Shaders\\Source\\Renderer2\\MainVertexShader.hpp" );

        return lShader;
    }

    Ref<IShaderProgram> NewMaterialSystem::CreateFragmentShader( Material aMaterial )
    {
        fs::path lShaderPath = "D:\\Work\\Git\\SpockEngine\\Resources\\Shaders\\Cache";

        std::string lShaderName = CreateShaderName( aMaterial, "fragment_shader" );
        auto lShader = CreateShaderProgram( mGraphicContext, eShaderStageTypeFlags::VERTEX, 450, lShaderName, lShaderPath );
        
        AddDefinitions( lShader, aMaterial );

        lShader->AddFile( "D:\\Work\\Git\\SpockEngine\\Shaders\\Source\\Renderer2\\Common\\Definitions.hpp" );
        lShader->AddFile( "D:\\Work\\Git\\SpockEngine\\Shaders\\Source\\Renderer2\\Varying.hpp" );
        lShader->AddFile( "D:\\Work\\Git\\SpockEngine\\Shaders\\Source\\Renderer2\\Common\\HelperFunctions.hpp" );
        lShader->AddFile( "D:\\Work\\Git\\SpockEngine\\Shaders\\Source\\Renderer2\\Material.hpp" );
        lShader->AddFile( "D:\\Work\\Git\\SpockEngine\\Shaders\\Source\\Renderer2\\MainFragmentShader.hpp" );

        return lShader;
    }

    // static Ref<IShaderProgram> MRTVertexShader( Ref<IGraphicContext> gc )
    // {
    //     fs::path lShaderPath   = "D:\\Work\\Git\\SpockEngine\\Resources\\Shaders\\Cache";
    //     auto     lVertexShader = CreateShaderProgram( gc, eShaderStageTypeFlags::VERTEX, 450, "geometry_vertex_shader", lShaderPath
    //     ); lVertexShader->AddCode( SE::Private::Shaders::gVertexLayout_data ); lVertexShader->AddCode(
    //     SE::Private::Shaders::gPBRMeshVertexShader_data ); lVertexShader->Compile();

    //     return lVertexShader;
    // }

    // static Ref<IShaderProgram> MRTFragmentShader( Ref<IGraphicContext> gc )
    // {
    //     fs::path lShaderPath = "D:\\Work\\Git\\SpockEngine\\Resources\\Shaders\\Cache";
    //     auto     lFragmentShader =
    //         CreateShaderProgram( gc, eShaderStageTypeFlags::FRAGMENT, 450, "geometry_fragment_shader", lShaderPath );
    //     lFragmentShader->AddCode( SE::Private::Shaders::gPBRMeshFragmentShaderPreamble_data );
    //     lFragmentShader->AddCode( SE::Private::Shaders::gToneMap_data );
    //     lFragmentShader->AddCode( SE::Private::Shaders::gPBRFunctions_data );
    //     lFragmentShader->AddCode( SE::Private::Shaders::gPBRMeshFragmentShaderCalculation_data );
    //     lFragmentShader->Compile();

    //     return lFragmentShader;
    // }

    // MeshRendererCreateInfo NewSceneRenderer::GetRenderPipelineCreateInfo( sMaterialShaderComponent &aPipelineSpecification )
    // {
    //     MeshRendererCreateInfo lCreateInfo;

    //     lCreateInfo.Opaque         = ( aPipelineSpecification.Type == eMaterialType::Opaque );
    //     lCreateInfo.IsTwoSided     = aPipelineSpecification.IsTwoSided;
    //     lCreateInfo.LineWidth      = aPipelineSpecification.LineWidth;
    //     lCreateInfo.VertexShader   = MRTVertexShader( mGraphicContext );
    //     lCreateInfo.FragmentShader = MRTFragmentShader( mGraphicContext );
    //     lCreateInfo.RenderPass     = mGeometryContext;

    //     return lCreateInfo;
    // }

    // MeshRendererCreateInfo NewSceneRenderer::GetRenderPipelineCreateInfo( sMeshRenderData &aPipelineSpecification )
    // {
    //     MeshRendererCreateInfo lCreateInfo;

    //     lCreateInfo.Opaque         = aPipelineSpecification.mOpaque;
    //     lCreateInfo.IsTwoSided     = aPipelineSpecification.mIsTwoSided;
    //     lCreateInfo.LineWidth      = aPipelineSpecification.mLineWidth;
    //     lCreateInfo.VertexShader   = MRTVertexShader( mGraphicContext );
    //     lCreateInfo.FragmentShader = MRTFragmentShader( mGraphicContext );
    //     lCreateInfo.RenderPass     = mGeometryContext;

    //     return lCreateInfo;
    // }

    // void NewMaterialSystem::CreateRenderTarget( uint32_t aOutputWidth, uint32_t aOutputHeight )
    // {
    //     sRenderTargetDescription lRenderTargetSpec{};
    //     lRenderTargetSpec.mWidth       = aOutputWidth;
    //     lRenderTargetSpec.mHeight      = aOutputHeight;
    //     lRenderTargetSpec.mSampleCount = mOutputSampleCount;
    //     mGeometryRenderTarget          = Graphics::CreateRenderTarget( mGraphicContext, lRenderTargetSpec );

    //     sAttachmentDescription lAttachmentCreateInfo{};
    //     lAttachmentCreateInfo.mIsSampled   = true;
    //     lAttachmentCreateInfo.mIsPresented = false;
    //     lAttachmentCreateInfo.mLoadOp      = eAttachmentLoadOp::CLEAR;
    //     lAttachmentCreateInfo.mStoreOp     = eAttachmentStoreOp::STORE;

    //     lAttachmentCreateInfo.mType       = eAttachmentType::COLOR;
    //     lAttachmentCreateInfo.mFormat     = eColorFormat::RGBA8_UNORM;
    //     lAttachmentCreateInfo.mClearColor = { 0.0f, 0.0f, 0.0f, 1.0f };

    //     mGeometryRenderTarget->AddAttachment( "OUTPUT", lAttachmentCreateInfo );

    //     lAttachmentCreateInfo.mType       = eAttachmentType::DEPTH;
    //     lAttachmentCreateInfo.mClearColor = { 1.0f, 0.0f, 0.0f, 0.0f };
    //     mGeometryRenderTarget->AddAttachment( "DEPTH_STENCIL", lAttachmentCreateInfo );
    //     mGeometryRenderTarget->Finalize();
    //     mGeometryContext = CreateRenderContext( mGraphicContext, mGeometryRenderTarget );
    // }

    // void NewMaterialSystem::CreateMSAARenderTarget( uint32_t aOutputWidth, uint32_t aOutputHeight )
    // {
    //     sRenderTargetDescription lRenderTargetSpec{};
    //     lRenderTargetSpec.mWidth       = aOutputWidth;
    //     lRenderTargetSpec.mHeight      = aOutputHeight;
    //     lRenderTargetSpec.mSampleCount = mOutputSampleCount;
    //     mGeometryRenderTarget          = Graphics::CreateRenderTarget( mGraphicContext, lRenderTargetSpec );

    //     sAttachmentDescription lAttachmentCreateInfo{};
    //     lAttachmentCreateInfo.mIsSampled   = true;
    //     lAttachmentCreateInfo.mIsPresented = false;
    //     lAttachmentCreateInfo.mLoadOp      = eAttachmentLoadOp::CLEAR;
    //     lAttachmentCreateInfo.mStoreOp     = eAttachmentStoreOp::STORE;

    //     lAttachmentCreateInfo.mType       = eAttachmentType::COLOR;
    //     lAttachmentCreateInfo.mFormat     = eColorFormat::RGBA8_UNORM;
    //     lAttachmentCreateInfo.mClearColor = { 0.0f, 0.0f, 0.0f, 1.0f };
    //     mGeometryRenderTarget->AddAttachment( "MSAA_OUTPUT", lAttachmentCreateInfo );

    //     lAttachmentCreateInfo.mType       = eAttachmentType::MSAA_RESOLVE;
    //     lAttachmentCreateInfo.mFormat     = eColorFormat::RGBA8_UNORM;
    //     lAttachmentCreateInfo.mClearColor = { 0.0f, 0.0f, 0.0f, 1.0f };
    //     mGeometryRenderTarget->AddAttachment( "OUTPUT", lAttachmentCreateInfo );

    //     lAttachmentCreateInfo.mType       = eAttachmentType::DEPTH;
    //     lAttachmentCreateInfo.mClearColor = { 1.0f, 0.0f, 0.0f, 0.0f };
    //     mGeometryRenderTarget->AddAttachment( "DEPTH_STENCIL", lAttachmentCreateInfo );

    //     mGeometryRenderTarget->Finalize();
    //     mGeometryContext = CreateRenderContext( mGraphicContext, mGeometryRenderTarget );
    // }

    // void NewMaterialSystem::CreateFXAARenderTarget( uint32_t aOutputWidth, uint32_t aOutputHeight )
    // {
    //     sRenderTargetDescription lFxaaSpec{};
    //     lFxaaSpec.mWidth       = aOutputWidth;
    //     lFxaaSpec.mHeight      = aOutputHeight;
    //     lFxaaSpec.mSampleCount = mOutputSampleCount;
    //     mFxaaRenderTarget      = Graphics::CreateRenderTarget( mGraphicContext, lFxaaSpec );

    //     sAttachmentDescription lAttachmentCreateInfo{};
    //     lAttachmentCreateInfo.mIsSampled   = true;
    //     lAttachmentCreateInfo.mIsPresented = false;
    //     lAttachmentCreateInfo.mLoadOp      = eAttachmentLoadOp::CLEAR;
    //     lAttachmentCreateInfo.mStoreOp     = eAttachmentStoreOp::STORE;
    //     lAttachmentCreateInfo.mType        = eAttachmentType::COLOR;
    //     lAttachmentCreateInfo.mFormat      = eColorFormat::RGBA16_FLOAT;
    //     lAttachmentCreateInfo.mClearColor  = { 0.0f, 0.0f, 0.0f, 1.0f };
    //     lAttachmentCreateInfo.mLoadOp      = eAttachmentLoadOp::CLEAR;
    //     lAttachmentCreateInfo.mStoreOp     = eAttachmentStoreOp::STORE;

    //     mFxaaRenderTarget->AddAttachment( "OUTPUT", lAttachmentCreateInfo );
    //     mFxaaRenderTarget->Finalize();

    //     mFxaaContext = CreateRenderContext( mGraphicContext, mFxaaRenderTarget );
    // }

    // void NewMaterialSystem::ResizeOutput( uint32_t aOutputWidth, uint32_t aOutputHeight )
    // {
    // sRenderTargetDescription lRenderTargetSpec{};
    // lRenderTargetSpec.mWidth       = aOutputWidth;
    // lRenderTargetSpec.mHeight      = aOutputHeight;
    // lRenderTargetSpec.mSampleCount = mOutputSampleCount;
    // mGeometryRenderTarget          = CreateRenderTarget( mGraphicContext, lRenderTargetSpec );

    // if( mOutputSampleCount == 1 )
    //     CreateRenderTarget( aOutputWidth, aOutputHeight );
    // else
    //     CreateMSAARenderTarget( aOutputWidth, aOutputHeight );

    // mFxaaSampler = CreateSampler2D( mGraphicContext, mGeometryRenderTarget->GetAttachment( "OUTPUT" ) );
    // CreateFXAARenderTarget( aOutputWidth, aOutputHeight );
    // sRenderTargetDescription lFxaaSpec{};
    // lFxaaSpec.mWidth                  = aOutputWidth;
    // lFxaaSpec.mHeight                 = aOutputHeight;
    // lFxaaSpec.mSampleCount            = mOutputSampleCount;
    // mFxaaRenderTarget                 = CreateRenderTarget( mGraphicContext, lRenderTargetSpec );
    // lAttachmentCreateInfo.mType       = eAttachmentType::COLOR;
    // lAttachmentCreateInfo.mFormat     = eColorFormat::RGBA16_FLOAT;
    // lAttachmentCreateInfo.mClearColor = { 0.0f, 0.0f, 0.0f, 1.0f };
    // lAttachmentCreateInfo.mLoadOp     = eAttachmentLoadOp::CLEAR;
    // lAttachmentCreateInfo.mStoreOp    = eAttachmentStoreOp::STORE;
    // mFxaaRenderTarget->AddAttachment( "OUTPUT", lAttachmentCreateInfo );
    // mFxaaRenderTarget->Finalize();
    // mFxaaContext = CreateRenderContext( mGraphicContext, mFxaaRenderTarget );

    // mCoordinateGridRenderer = New<CoordinateGridRenderer>( mGraphicContext, mGeometryContext );
    // mShadowSceneRenderer    = New<ShadowSceneRenderer>( mGraphicContext );

    // {
    //     fs::path lShaderPath = "D:\\Work\\Git\\SpockEngine\\Resources\\Shaders\\Cache";
    //     auto     lVertexShader =
    //         CreateShaderProgram( mGraphicContext, eShaderStageTypeFlags::VERTEX, 450, "omni_shadow_vertex_shader", lShaderPath );

    //     lVertexShader->AddCode( "layout( location = 0 ) in vec2 inUV;\n" );
    //     lVertexShader->AddCode( "layout( location = 1 ) in vec4 inConsoleUV;\n" );
    //     lVertexShader->AddCode( "layout( set = 0, binding = 0 ) uniform sampler2D sImage;\n" );
    //     lVertexShader->AddCode( "layout( location = 0 ) out vec4 outFragcolor;\n" );
    //     lVertexShader->AddCode( "#define FXAA_PC 1\n" );
    //     lVertexShader->AddCode( "#define FXAA_GLSL_130 1\n" );
    //     lVertexShader->AddCode( "#define FXAA_QUALITY__PRESET 23\n" );
    //     lVertexShader->AddCode( SE::Private::Shaders::gFXAACode_data );
    //     lVertexShader->AddCode( SE::Private::Shaders::gFXAAFragmentShader_data );
    //     lVertexShader->Compile();
    //     // mPipeline->SetShader( eShaderStageTypeFlags::VERTEX, lVertexShader, "main" );

    //     auto lFragmentShader = CreateShaderProgram( mGraphicContext, eShaderStageTypeFlags::FRAGMENT, 450,
    //                                                 "omni_shadow_fragment_shader", lShaderPath );
    //     lFragmentShader->AddCode( SE::Private::Shaders::gCopyFragmentShader_data );
    //     lFragmentShader->Compile();

    //     EffectProcessorCreateInfo lEffectProcessorCreateInfo{};
    //     lEffectProcessorCreateInfo.mVertexShader   = lVertexShader;
    //     lEffectProcessorCreateInfo.mFragmentShader = lFragmentShader;
    //     lEffectProcessorCreateInfo.RenderPass      = mFxaaContext;
    //     mFxaaRenderer = New<EffectProcessor>( mGraphicContext, mFxaaContext, lEffectProcessorCreateInfo );
    // }

    // {
    //     fs::path lShaderPath = "D:\\Work\\Git\\SpockEngine\\Resources\\Shaders\\Cache";
    //     auto     lVertexShader =
    //         CreateShaderProgram( mGraphicContext, eShaderStageTypeFlags::VERTEX, 450, "omni_shadow_vertex_shader", lShaderPath );
    //     lVertexShader->AddCode( SE::Private::Shaders::gFXAAVertexShader_data );
    //     lVertexShader->Compile();
    //     // mPipeline->SetShader( eShaderStageTypeFlags::VERTEX, lVertexShader, "main" );

    //     auto lFragmentShader = CreateShaderProgram( mGraphicContext, eShaderStageTypeFlags::FRAGMENT, 450,
    //                                                 "omni_shadow_fragment_shader", lShaderPath );
    //     lFragmentShader->AddCode( SE::Private::Shaders::gCopyFragmentShader_data );
    //     lFragmentShader->Compile();

    //     EffectProcessorCreateInfo lCopyCreateInfo{};
    //     lCopyCreateInfo.mVertexShader   = lVertexShader;
    //     lCopyCreateInfo.mFragmentShader = lFragmentShader;
    //     lCopyCreateInfo.RenderPass      = mFxaaContext;
    //     mCopyRenderer                   = New<EffectProcessor>( mGraphicContext, mFxaaContext, lCopyCreateInfo );
    // }
    // }

    // Ref<MeshRenderer> NewSceneRenderer::GetRenderPipeline( MeshRendererCreateInfo const &aPipelineSpecification )
    // {
    //     if( mMeshRenderers.find( aPipelineSpecification ) == mMeshRenderers.end() )
    //         mMeshRenderers[aPipelineSpecification] = New<MeshRenderer>( mGraphicContext, aPipelineSpecification );

    //     return mMeshRenderers[aPipelineSpecification];
    // }

    // Ref<MeshRenderer> NewSceneRenderer::GetRenderPipeline( sMaterialShaderComponent &aPipelineSpecification )
    // {
    //     MeshRendererCreateInfo lCreateInfo = GetRenderPipelineCreateInfo( aPipelineSpecification );

    //     return GetRenderPipeline( lCreateInfo );
    // }

    // Ref<MeshRenderer> NewSceneRenderer::GetRenderPipeline( sMeshRenderData &aPipelineSpecification )
    // {
    //     MeshRendererCreateInfo lCreateInfo = GetRenderPipelineCreateInfo( aPipelineSpecification );

    //     return GetRenderPipeline( lCreateInfo );
    // }

    // Ref<ParticleSystemRenderer> NewSceneRenderer::GetRenderPipeline( ParticleRendererCreateInfo &aPipelineSpecification )
    // {
    //     if( mParticleRenderers.find( aPipelineSpecification ) == mParticleRenderers.end() )
    //         mParticleRenderers[aPipelineSpecification] =
    //             New<ParticleSystemRenderer>( mGraphicContext, mGeometryContext, aPipelineSpecification );

    //     return mParticleRenderers[aPipelineSpecification];
    // }

    // Ref<ParticleSystemRenderer> NewSceneRenderer::GetRenderPipeline( sParticleShaderComponent &aPipelineSpecification )
    // {
    //     ParticleRendererCreateInfo lCreateInfo = GetRenderPipelineCreateInfo( aPipelineSpecification );

    //     return GetRenderPipeline( lCreateInfo );
    // }

    // Ref<ParticleSystemRenderer> NewSceneRenderer::GetRenderPipeline( sParticleRenderData &aPipelineSpecification )
    // {
    //     ParticleRendererCreateInfo lCreateInfo = GetRenderPipelineCreateInfo( aPipelineSpecification );

    //     return GetRenderPipeline( lCreateInfo );
    // }

    // static Ref<IShaderProgram> ParticleVertexShader( Ref<IGraphicContext> gc )
    // {
    //     fs::path lShaderPath = "D:\\Work\\Git\\SpockEngine\\Resources\\Shaders\\Cache";
    //     auto     lVertexShader =
    //         CreateShaderProgram( gc, eShaderStageTypeFlags::VERTEX, 450, "particle_system_vertex_shader", lShaderPath );
    //     lVertexShader->AddCode( SE::Private::Shaders::gParticleSystemVertexShader_data );
    //     lVertexShader->Compile();

    //     return lVertexShader;
    // }

    // static Ref<IShaderProgram> ParticleFragmentShader( Ref<IGraphicContext> gc )
    // {
    //     fs::path lShaderPath = "D:\\Work\\Git\\SpockEngine\\Resources\\Shaders\\Cache";
    //     auto     lVertexShader =
    //         CreateShaderProgram( gc, eShaderStageTypeFlags::FRAGMENT, 450, "particle_system_fragment_shader", lShaderPath );
    //     lVertexShader->AddCode( SE::Private::Shaders::gParticleSystemFragmentShader_data );
    //     lVertexShader->Compile();

    //     return lVertexShader;
    // }

    // ParticleRendererCreateInfo NewSceneRenderer::GetRenderPipelineCreateInfo( sParticleShaderComponent &aPipelineSpecification )
    // {
    //     ParticleRendererCreateInfo lCreateInfo;
    //     lCreateInfo.LineWidth      = aPipelineSpecification.LineWidth;
    //     lCreateInfo.VertexShader   = ParticleVertexShader( mGraphicContext );
    //     lCreateInfo.FragmentShader = ParticleFragmentShader( mGraphicContext );
    //     lCreateInfo.RenderPass     = mGeometryContext;

    //     return lCreateInfo;
    // }

    // ParticleRendererCreateInfo NewSceneRenderer::GetRenderPipelineCreateInfo( sParticleRenderData &aPipelineSpecification )
    // {
    //     ParticleRendererCreateInfo lCreateInfo;
    //     lCreateInfo.LineWidth      = aPipelineSpecification.mLineWidth;
    //     lCreateInfo.VertexShader   = ParticleVertexShader( mGraphicContext );
    //     lCreateInfo.FragmentShader = ParticleFragmentShader( mGraphicContext );
    //     lCreateInfo.RenderPass     = mGeometryContext;

    //     return lCreateInfo;
    // }

    // void NewMaterialSystem::Update( Ref<Scene> aWorld )
    // {
    //     BaseSceneRenderer::Update( aWorld );
    // mShadowSceneRenderer->Update( aWorld );

    // mView.PointLightCount = mPointLights.size();
    // for( uint32_t i = 0; i < mView.PointLightCount; i++ ) mView.PointLights[i] = mPointLights[i];

    // mView.DirectionalLightCount = mDirectionalLights.size();
    // for( uint32_t i = 0; i < mView.DirectionalLightCount; i++ ) mView.DirectionalLights[i] = mDirectionalLights[i];

    // mView.SpotlightCount = mSpotlights.size();
    // for( uint32_t i = 0; i < mView.SpotlightCount; i++ ) mView.Spotlights[i] = mSpotlights[i];

    // mSettings.AmbientLightIntensity = mAmbientLight.a;
    // mSettings.AmbientLightColor     = vec4( vec3( mAmbientLight ), 0.0 );
    // mSettings.Gamma                 = mGamma;
    // mSettings.Exposure              = mExposure;
    // mSettings.RenderGrayscale       = mGrayscaleRendering ? 1.0f : 0.0f;

    // mView.Projection     = mProjectionMatrix;
    // mView.CameraPosition = mCameraPosition;
    // mView.View           = mViewMatrix;

    // mCameraUniformBuffer->Write( mView );
    // mShaderParametersBuffer->Write( mSettings );

    // Update pipelines
    // }

    // void NewMaterialSystem::Render()
    // {
    //     SE_PROFILE_FUNCTION();

    //     if( !mScene ) return;

    // mScene->GetMaterialSystem()->UpdateDescriptors();

    // mShadowSceneRenderer->Render();
    // if( mShadowSceneRenderer->GetDirectionalShadowMapSamplers().size() > 0 )
    //     mLightingPassDirectionalShadowMaps->Write( mShadowSceneRenderer->GetDirectionalShadowMapSamplers(), 0 );

    // if( mShadowSceneRenderer->GetSpotlightShadowMapSamplers().size() > 0 )
    //     mLightingPassSpotlightShadowMaps->Write( mShadowSceneRenderer->GetSpotlightShadowMapSamplers(), 0 );

    // if( mShadowSceneRenderer->GetPointLightShadowMapSamplers().size() > 0 )
    //     mLightingPassPointLightShadowMaps->Write( mShadowSceneRenderer->GetPointLightShadowMapSamplers(), 0 );

    // mGeometryContext->BeginRender();
    // for( auto &lPipelineData : mOpaqueMeshQueue )
    // {
    //     auto lPipeline = GetRenderPipeline( lPipelineData );
    //     if( lPipeline->Pipeline() )
    //         mGeometryContext->Bind( lPipeline->Pipeline() );
    //     else
    //         continue;
    //     mGeometryContext->Bind( mSceneDescriptors, 0, -1 );
    //     mGeometryContext->Bind( mScene->GetMaterialSystem()->GetDescriptorSet(), 1, -1 );

    //     if( !lPipelineData.mVertexBuffer || !lPipelineData.mIndexBuffer ) continue;
    //     mGeometryContext->Bind( lPipelineData.mVertexBuffer, lPipelineData.mIndexBuffer );

    //     MeshRenderer::MaterialPushConstants lMaterialPushConstants{};
    //     lMaterialPushConstants.mMaterialID = lPipelineData.mMaterialID;

    //     mGeometryContext->PushConstants( { eShaderStageTypeFlags::FRAGMENT }, 0, lMaterialPushConstants );

    //     mGeometryContext->Draw( lPipelineData.mIndexCount, lPipelineData.mIndexOffset, lPipelineData.mVertexOffset, 1, 0 );
    // }

    // for( auto &lParticleSystem : mParticleQueue )
    // {
    //     auto &lPipeline = GetRenderPipeline( lParticleSystem );

    //     ParticleSystemRenderer::ParticleData lParticleData{};
    //     lParticleData.Model         = lParticleSystem.mModel;
    //     lParticleData.ParticleCount = lParticleSystem.mParticleCount;
    //     lParticleData.ParticleSize  = lParticleSystem.mParticleSize;
    //     lParticleData.Particles     = lParticleSystem.mParticles;

    //     lPipeline->Render( mView.Projection, mView.View, mGeometryContext, lParticleData );
    // }

    // for( auto const &lLightGizmo : mLightGizmos )
    // {
    //     switch( lLightGizmo.mType )
    //     {
    //     case eLightType::DIRECTIONAL:
    //     {
    //         // mVisualHelperRenderer->Render( lLightGizmo.mMatrix, aDirectionalLightHelperComponent, mGeometryContext );
    //         break;
    //     }
    //     case eLightType::POINT_LIGHT:
    //     {
    //         // mVisualHelperRenderer->Render( lLightGizmo.mMatrix, aPointLightHelperComponent, mGeometryContext );
    //         break;
    //     }
    //     case eLightType::SPOTLIGHT:
    //     {
    //         // mVisualHelperRenderer->Render( lLightGizmo.mMatrix, aSpotlightHelperComponent, mGeometryContext );
    //         break;
    //     }
    //     }
    // }

    // // if( mRenderCoordinateGrid ) mCoordinateGridRenderer->Render( mView.Projection, mView.View, mGeometryContext );
    // mGeometryContext->EndRender();

    // mFxaaContext->BeginRender();
    // if( mUseFXAA )
    // {
    //     mFxaaRenderer->Render( mFxaaSampler, mFxaaContext );
    // }
    // else
    // {
    //     mCopyRenderer->Render( mFxaaSampler, mFxaaContext );
    // }
    // mFxaaContext->EndRender();
    // }

    // Ref<ITexture2D> NewMaterialSystem::GetOutputImage()
    // {
    //     //
    //     return mGeometryRenderTarget->GetAttachment( "OUTPUT" );
    // }
} // namespace SE::Core