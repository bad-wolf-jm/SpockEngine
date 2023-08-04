#include "ShadowRenderer.h"

#include <chrono>
#include <gli/gli.hpp>

#include "Core/Logging.h"
#include "Core/Profiling/BlockTimer.h"
#include "Core/Resource.h"

#include "Shaders/gDirectionalShadowVertexShader.h"
#include "Shaders/gOmniDirectionalShadowFragmentShader.h"
#include "Shaders/gOmniDirectionalShadowVertexShader.h"
#include "Shaders/gVertexLayout.h"
namespace SE::Core
{
    using namespace math;
    using namespace SE::Core::EntityComponentSystem::Components;
    using namespace SE::Core::Primitives;

    ShadowMeshRenderer::ShadowMeshRenderer( Ref<IGraphicContext> aGraphicContext, ShadowMeshRendererCreateInfo const &aCreateInfo )
        : mGraphicContext( aGraphicContext )
        , Spec{ aCreateInfo }
    {

        mPipeline = CreateGraphicsPipeline( mGraphicContext, Spec.RenderPass, ePrimitiveTopology::TRIANGLES );

        mPipeline->SetCulling( eFaceCulling::BACK );
        mPipeline->SetDepthParameters( true, true, eDepthCompareOperation::LESS_OR_EQUAL );

        fs::path lShaderPath = "D:\\Work\\Git\\SpockEngine\\Resources\\Shaders\\Cache";
        auto     lVertexShader =
            CreateShaderProgram( mGraphicContext, eShaderStageTypeFlags::VERTEX, 450, "shadow_vertex_shader", lShaderPath );
        lVertexShader->AddCode( "#define __GLSL__" );
        lVertexShader->AddCode( "#define VULKAN_SEMANTICS" );
        lVertexShader->AddCode( "#define DIRECTIONAL_LIGHT_SHADOW_VERTEX_SHADER" );
        lVertexShader->AddCode( "#define MATERIAL_HAS_NORMALS" );
        lVertexShader->AddCode( "#define MATERIAL_HAS_UV0" );
        lVertexShader->AddFile( "D:\\Work\\Git\\SpockEngine\\Shaders\\Source\\Common\\Definitions.hpp" );
        lVertexShader->AddFile( "D:\\Work\\Git\\SpockEngine\\Shaders\\Source\\Varying.hpp" );
        lVertexShader->AddFile( "D:\\Work\\Git\\SpockEngine\\Shaders\\Source\\Shadow.hpp" );
        lVertexShader->Compile();

        mPipeline->SetShader( eShaderStageTypeFlags::VERTEX, lVertexShader, "main" );
        mPipeline->AddInput( "Position", eBufferDataType::VEC3, 0, 0 );
        mPipeline->AddInput( "Normal", eBufferDataType::VEC3, 0, 1 );
        mPipeline->AddInput( "TexCoord_0", eBufferDataType::VEC2, 0, 2 );
        mPipeline->AddInput( "Bones", eBufferDataType::VEC4, 0, 3 );
        mPipeline->AddInput( "Weights", eBufferDataType::VEC4, 0, 4 );

        mCameraBuffer = CreateBuffer( mGraphicContext, eBufferType::UNIFORM_BUFFER, true, true, true, true, sizeof( ShadowMatrices ) );
        mCameraSetLayout = CreateDescriptorSetLayout( aGraphicContext );
        mCameraSetLayout->AddBinding( 0, eDescriptorType::UNIFORM_BUFFER, { eShaderStageTypeFlags::VERTEX } );
        mCameraSetLayout->Build();
        mCameraDescriptors = mCameraSetLayout->Allocate();

        mPipeline->AddDescriptorSet( mCameraSetLayout );

        mPipeline->Build();
    }

    void ShadowMeshRenderer::SetView( ShadowMatrices const &aView )
    {
        mCameraBuffer->Write( aView );
    }

    OmniShadowMeshRenderer::OmniShadowMeshRenderer( Ref<IGraphicContext>                aGraphicContext,
                                                    ShadowMeshRendererCreateInfo const &aCreateInfo )
        : mGraphicContext( aGraphicContext )
        , Spec{ aCreateInfo }
    {
        mPipeline = CreateGraphicsPipeline( mGraphicContext, Spec.RenderPass, ePrimitiveTopology::TRIANGLES );

        mPipeline->SetCulling( eFaceCulling::BACK );
        mPipeline->SetDepthParameters( true, true, eDepthCompareOperation::LESS_OR_EQUAL );

        fs::path lShaderPath = "D:\\Work\\Git\\SpockEngine\\Resources\\Shaders\\Cache";
        auto     lVertexShader =
            CreateShaderProgram( mGraphicContext, eShaderStageTypeFlags::VERTEX, 450, "omni_shadow_vertex_shader", lShaderPath );

        lVertexShader->AddCode( "#define __GLSL__" );
        lVertexShader->AddCode( "#define VULKAN_SEMANTICS" );
        lVertexShader->AddCode( "#define PUNCTUAL_LIGHT_SHADOW_VERTEX_SHADER" );
        lVertexShader->AddCode( "#define MATERIAL_HAS_NORMALS" );
        lVertexShader->AddCode( "#define MATERIAL_HAS_UV0" );
        lVertexShader->AddFile( "D:\\Work\\Git\\SpockEngine\\Shaders\\Source\\Common\\Definitions.hpp" );
        lVertexShader->AddFile( "D:\\Work\\Git\\SpockEngine\\Shaders\\Source\\Varying.hpp" );
        lVertexShader->AddFile( "D:\\Work\\Git\\SpockEngine\\Shaders\\Source\\Shadow.hpp" );
        lVertexShader->Compile();
        mPipeline->SetShader( eShaderStageTypeFlags::VERTEX, lVertexShader, "main" );

        auto lFragmentShader =
            CreateShaderProgram( mGraphicContext, eShaderStageTypeFlags::FRAGMENT, 450, "omni_shadow_fragment_shader", lShaderPath );
        lFragmentShader->AddCode( "#define __GLSL__" );
        lFragmentShader->AddCode( "#define VULKAN_SEMANTICS" );
        lFragmentShader->AddCode( "#define PUNCTUAL_LIGHT_SHADOW_FRAGMENT_SHADER" );
        lFragmentShader->AddFile( "D:\\Work\\Git\\SpockEngine\\Shaders\\Source\\Common\\Definitions.hpp" );
        lFragmentShader->AddFile( "D:\\Work\\Git\\SpockEngine\\Shaders\\Source\\Shadow.hpp" );
        lFragmentShader->Compile();

        mPipeline->SetShader( eShaderStageTypeFlags::FRAGMENT, lFragmentShader, "main" );
        mPipeline->AddInput( "Position", eBufferDataType::VEC3, 0, 0 );
        mPipeline->AddInput( "Normal", eBufferDataType::VEC3, 0, 1 );
        mPipeline->AddInput( "TexCoord_0", eBufferDataType::VEC2, 0, 2 );
        mPipeline->AddInput( "Bones", eBufferDataType::VEC4, 0, 3 );
        mPipeline->AddInput( "Weights", eBufferDataType::VEC4, 0, 4 );

        mCameraSetLayout = CreateDescriptorSetLayout( aGraphicContext );
        mCameraSetLayout->AddBinding( 0, eDescriptorType::UNIFORM_BUFFER, { eShaderStageTypeFlags::VERTEX } );
        mCameraSetLayout->Build();

        mPipeline->AddDescriptorSet( mCameraSetLayout );
        mPipeline->Build();
    }

    Ref<IDescriptorSet> OmniShadowMeshRenderer::AllocateDescriptors()
    {
        return mCameraSetLayout->Allocate();
    }

    static math::mat4 CreateCubeFaceViewMatrix( int aFace )
    {
        glm::mat4 lViewMatrix = glm::mat4( 1.0f );
        switch( aFace )
        {
        case 0:
            // POSITIVE_X
            lViewMatrix = glm::rotate( lViewMatrix, glm::radians( 90.0f ), glm::vec3( 0.0f, 1.0f, 0.0f ) );
            lViewMatrix = glm::rotate( lViewMatrix, glm::radians( 180.0f ), glm::vec3( 1.0f, 0.0f, 0.0f ) );
            break;
        case 1:
            // NEGATIVE_X
            lViewMatrix = glm::rotate( lViewMatrix, glm::radians( -90.0f ), glm::vec3( 0.0f, 1.0f, 0.0f ) );
            lViewMatrix = glm::rotate( lViewMatrix, glm::radians( 180.0f ), glm::vec3( 1.0f, 0.0f, 0.0f ) );
            break;
        case 2:
            // POSITIVE_Y
            lViewMatrix = glm::rotate( lViewMatrix, glm::radians( -90.0f ), glm::vec3( 1.0f, 0.0f, 0.0f ) );
            break;
        case 3:
            // NEGATIVE_Y
            lViewMatrix = glm::rotate( lViewMatrix, glm::radians( 90.0f ), glm::vec3( 1.0f, 0.0f, 0.0f ) );
            break;
        case 4:
            // POSITIVE_Z
            lViewMatrix = glm::rotate( lViewMatrix, glm::radians( 180.0f ), glm::vec3( 1.0f, 0.0f, 0.0f ) );
            break;
        case 5:
            // NEGATIVE_Z
            lViewMatrix = glm::rotate( lViewMatrix, glm::radians( 180.0f ), glm::vec3( 0.0f, 0.0f, 1.0f ) );
            break;
        }

        return lViewMatrix;
    }

    ShadowSceneRenderer::ShadowSceneRenderer( Ref<IGraphicContext> aGraphicContext )
        : BaseSceneRenderer( aGraphicContext, eColorFormat::UNDEFINED, 1 )
    {
        auto lDirectionalShadowMaps        = NewRenderTarget( 1024, 1024 );
        mDirectionalShadowMapRenderContext = CreateRenderContext( mGraphicContext, lDirectionalShadowMaps );
        ShadowMeshRendererCreateInfo lCreateInfo{};
        lCreateInfo.RenderPass = mDirectionalShadowMapRenderContext;

        mRenderPipeline              = New<ShadowMeshRenderer>( mGraphicContext, lCreateInfo );
        mDirectionalShadowMapSampler = CreateSampler2D( mGraphicContext, lDirectionalShadowMaps->GetAttachment( "SHADOW_MAP" ) );

        constexpr int32_t        mOmniShadowResolution = 1024;
        sRenderTargetDescription lRenderTargetSpec{};
        lRenderTargetSpec.mWidth       = mOmniShadowResolution;
        lRenderTargetSpec.mHeight      = mOmniShadowResolution;
        lRenderTargetSpec.mSampleCount = 1;

        auto lRenderTarget = CreateRenderTarget( mGraphicContext, lRenderTargetSpec );
        lRenderTarget->AddAttachment( "SHADOW_MAP", sAttachmentDescription( eAttachmentType::COLOR, eColorFormat::R32_FLOAT, true ) );
        lRenderTarget->AddAttachment( "DEPTH", sAttachmentDescription( eAttachmentType::DEPTH, eColorFormat::UNDEFINED ) );
        lRenderTarget->Finalize();
        lCreateInfo.RenderPass = CreateRenderContext( mGraphicContext, lRenderTarget );
        mOmniRenderPipeline    = New<OmniShadowMeshRenderer>( mGraphicContext, lCreateInfo );
    }

    void ShadowSceneRenderer::ResizeOutput( uint32_t aOutputWidth, uint32_t aOutputHeight )
    {
    }

    Ref<IRenderTarget> ShadowSceneRenderer::NewRenderTarget( uint32_t aOutputWidth, uint32_t aOutputHeight )
    {
        sRenderTargetDescription lRenderTargetSpec{};
        lRenderTargetSpec.mWidth       = aOutputWidth;
        lRenderTargetSpec.mHeight      = aOutputHeight;
        lRenderTargetSpec.mSampleCount = 1;

        auto lRenderTarget = CreateRenderTarget( mGraphicContext, lRenderTargetSpec );
        lRenderTarget->AddAttachment( "SHADOW_MAP", sAttachmentDescription( eAttachmentType::DEPTH, eColorFormat::UNDEFINED ) );
        lRenderTarget->Finalize();

        return lRenderTarget;
    }

    void ShadowSceneRenderer::SetLights( sDirectionalLight const &aDirectionalLights )
    {
        mDirectionalLight = aDirectionalLights;
    }

    void ShadowSceneRenderer::SetLights( std::vector<sPunctualLight> const &aPointLights )
    {
        mPointLights = aPointLights;
    }

    void ShadowSceneRenderer::Update( Ref<Scene> aWorld )
    {
        BaseSceneRenderer::Update( aWorld );

        constexpr int32_t mOmniShadowResolution = 1024;
        if( mPointLights.size() != mPointLightsShadowMapRenderContext.size() )
        {
            mPointLightsShadowMapRenderContext.clear();
            mPointLightsShadowCameraUniformBuffer.clear();
            mPointLightsShadowSceneDescriptors.clear();
            mPointLightShadowMapSamplers.clear();

            sTextureCreateInfo lCreateInfo{};
            lCreateInfo.mFormat = eColorFormat::RGBA32_FLOAT;
            lCreateInfo.mWidth  = mOmniShadowResolution;
            lCreateInfo.mHeight = mOmniShadowResolution;
            lCreateInfo.mDepth  = 1;
            lCreateInfo.mLayers = 6;

            for( uint32_t i = 0; i < mPointLights.size(); i++ )
            {
                mPointLightsShadowMapRenderContext.emplace_back();
                mPointLightsShadowSceneDescriptors.emplace_back();
                mPointLightsShadowCameraUniformBuffer.emplace_back();

                auto lShadowMap = CreateTexture2D( mGraphicContext, lCreateInfo, 1, false, true, false, false );
                mPointLightShadowMapSamplers.emplace_back();
                mPointLightShadowMapSamplers.back() = CreateSamplerCubeMap( mGraphicContext, lShadowMap );

                for( uint32_t f = 0; f < 6; f++ )
                {
                    sRenderTargetDescription lRenderTargetSpec{};
                    lRenderTargetSpec.mWidth       = mOmniShadowResolution;
                    lRenderTargetSpec.mHeight      = mOmniShadowResolution;
                    lRenderTargetSpec.mSampleCount = 1;

                    auto lRenderTarget = CreateRenderTarget( mGraphicContext, lRenderTargetSpec );
                    lRenderTarget->AddAttachment( "SHADOW_MAP",
                                                  sAttachmentDescription( eAttachmentType::COLOR, eColorFormat::R32_FLOAT, true ),
                                                  lShadowMap, static_cast<eCubeFace>( f ) );
                    lRenderTarget->AddAttachment( "DEPTH", sAttachmentDescription( eAttachmentType::DEPTH, eColorFormat::UNDEFINED ) );
                    lRenderTarget->Finalize();

                    mPointLightsShadowMapRenderContext.back()[f]    = CreateRenderContext( mGraphicContext, lRenderTarget );
                    mPointLightsShadowSceneDescriptors.back()[f]    = mOmniRenderPipeline->AllocateDescriptors();
                    mPointLightsShadowCameraUniformBuffer.back()[f] = CreateBuffer( mGraphicContext, eBufferType::UNIFORM_BUFFER, true,
                                                                                    true, true, true, sizeof( OmniShadowMatrices ) );
                    mPointLightsShadowSceneDescriptors.back()[f]->Write( mPointLightsShadowCameraUniformBuffer.back()[f], false, 0,
                                                                         sizeof( OmniShadowMatrices ), 0 );
                }
            }
        }
    }

    void ShadowSceneRenderer::Render()
    {
        SE_PROFILE_FUNCTION();

        if( !mScene )
            return;

        if( mRenderPipeline->Pipeline() )
        {
            uint32_t lLightIndex = 0;
            View.mMVP            = mDirectionalLight.mTransform;
            mRenderPipeline->SetView( View );

            mDirectionalShadowMapRenderContext->BeginRender();
            mDirectionalShadowMapRenderContext->Bind( mRenderPipeline->Pipeline() );
            mDirectionalShadowMapRenderContext->Bind( mRenderPipeline->View(), 0, -1 );

            mScene->ForEach<sStaticMeshComponent, sMaterialComponent>(
                [&]( auto aEntity, auto &aStaticMeshComponent, auto &aMaterial )
                {
                    if( !aStaticMeshComponent.mVertexBuffer || !aStaticMeshComponent.mIndexBuffer )
                        return;

                    mDirectionalShadowMapRenderContext->Bind( aStaticMeshComponent.mTransformedBuffer,
                                                              aStaticMeshComponent.mIndexBuffer );
                    mDirectionalShadowMapRenderContext->Draw( aStaticMeshComponent.mIndexCount, aStaticMeshComponent.mIndexOffset,
                                                              aStaticMeshComponent.mVertexOffset, 1, 0 );
                } );

            mDirectionalShadowMapRenderContext->EndRender();
        }

        if( mOmniRenderPipeline->Pipeline() )
        {
            uint32_t lLightIndex = 0;
            for( auto &lContext : mPointLightsShadowMapRenderContext )
            {
                RenderPunctualShadowMap( mPointLights[lLightIndex].mPosition, lContext, mPointLightsShadowCameraUniformBuffer[lLightIndex],
                                         mPointLightsShadowSceneDescriptors[lLightIndex] );
                lLightIndex++;
            }
        }
    }

    void ShadowSceneRenderer::RenderPunctualShadowMap( math::vec3 aLightPosition, std::array<Ref<IRenderContext>, 6> aContext,
                                                       std::array<Ref<IGraphicBuffer>, 6> const &aUniforms,
                                                       std::array<Ref<IDescriptorSet>, 6> const &aDescriptors )
    {
        math::mat4 lProjection = math::Perspective( math::radians( 90.0f ), 1.0f, .2f, 1000.0f );
        mOmniView.mLightPos    = math::vec4( aLightPosition, 0.0f );

        for( uint32_t f = 0; f < 6; f++ )
        {
            glm::mat4 viewMatrix = CreateCubeFaceViewMatrix( f );
            mOmniView.mMVP       = lProjection * math::Translate( viewMatrix, -aLightPosition );
            aUniforms[f]->Write( mOmniView );
            
            RenderCubeFace( viewMatrix, lProjection, aContext[f], aDescriptors[f] );
        }
    }

    void ShadowSceneRenderer::RenderCubeFace( math::mat4 viewMatrix, math::mat4 lProjection, Ref<IRenderContext> lContext,
                                              Ref<IDescriptorSet> aDescriptors )
    {
        lContext->BeginRender();
        lContext->Bind( mOmniRenderPipeline->Pipeline() );
        lContext->Bind( aDescriptors, 0, -1 );

        mScene->ForEach<sStaticMeshComponent, sMaterialComponent>(
            [&]( auto aEntity, auto &aStaticMeshComponent, auto &aMaterial )
            {
                if( !aStaticMeshComponent.mVertexBuffer || !aStaticMeshComponent.mIndexBuffer )
                    return;

                lContext->Bind( aStaticMeshComponent.mTransformedBuffer, aStaticMeshComponent.mIndexBuffer );
                lContext->Draw( aStaticMeshComponent.mIndexCount, aStaticMeshComponent.mIndexOffset,
                                aStaticMeshComponent.mVertexOffset, 1, 0 );
            } );

        lContext->EndRender();
    }

    Ref<ITexture2D> ShadowSceneRenderer::GetOutputImage()
    {
        //
        return mGeometryRenderTarget->GetAttachment( "OUTPUT" );
    }
} // namespace SE::Core