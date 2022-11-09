#include "SceneRenderer.h"

#include <chrono>
#include <gli/gli.hpp>

#include "Core/Vulkan/VkPipeline.h"

#include "Scene/Components/VisualHelpers.h"
#include "Scene/Primitives/Primitives.h"
#include "Scene/VertexData.h"

#include "Core/Profiling/BlockTimer.h"

#include "Core/Logging.h"
#include "Core/Resource.h"

#include "MeshRenderer.h"
#include "ParticleSystemRenderer.h"

namespace LTSE::Core
{

    using namespace math;
    using namespace LTSE::Core::EntityComponentSystem::Components;
    using namespace LTSE::Core::Primitives;

    // DirectionalLightData::DirectionalLightData( const sDirectionalLightComponent &aSpec, math::mat4 aTransform )
    // {
    //     float lAzimuth   = math::radians( aSpec.Azimuth );
    //     float lElevation = math::radians( aSpec.Elevation );

    //     Direction = math::vec3{ math::sin( lElevation ) * math::cos( lAzimuth ), math::cos( lElevation ),
    //         math::sin( lElevation ) * math::sin( lAzimuth ) };
    //     Color     = aSpec.Color;
    //     Intensity = aSpec.Intensity;
    // }

    // PointLightData::PointLightData( const sPointLightComponent &aSpec, math::mat4 aTransform )
    // {
    //     WorldPosition = aTransform * math::vec4( aSpec.Position, 1.0f );
    //     Color         = aSpec.Color;
    //     Intensity     = aSpec.Intensity;
    // }

    // SpotlightData::SpotlightData( const sSpotlightComponent &aSpec, math::mat4 aTransform )
    // {
    //     float lAzimuth   = math::radians( aSpec.Azimuth );
    //     float lElevation = math::radians( aSpec.Elevation );

    //     WorldPosition   = aTransform * math::vec4( aSpec.Position, 1.0f );
    //     LookAtDirection = math::vec3{ math::sin( lElevation ) * math::cos( lAzimuth ), math::cos( lElevation ),
    //         math::sin( lElevation ) * math::sin( lAzimuth ) };
    //     Color           = aSpec.Color;
    //     Intensity       = aSpec.Intensity;
    //     Cone            = math::cos( math::radians( aSpec.Cone / 2 ) );
    // }

    SceneRenderer::SceneRenderer( Ref<Scene> aWorld, RenderContext &aRenderContext )
        : mGraphicContext{ aWorld->GetGraphicContext() }
        , mWorld{ aWorld }
    {
        mSceneDescriptors = New<DescriptorSet>( mGraphicContext, MeshRenderer::GetCameraSetLayout( mGraphicContext ) );

        mCameraUniformBuffer =
            New<Buffer>( mGraphicContext, eBufferBindType::UNIFORM_BUFFER, true, false, true, true, sizeof( WorldMatrices ) );
        mShaderParametersBuffer =
            New<Buffer>( mGraphicContext, eBufferBindType::UNIFORM_BUFFER, true, false, true, true, sizeof( CameraSettings ) );
        mSceneDescriptors->Write( mCameraUniformBuffer, false, 0, sizeof( WorldMatrices ), 0 );
        mSceneDescriptors->Write( mShaderParametersBuffer, false, 0, sizeof( CameraSettings ), 1 );

        CoordinateGridRendererCreateInfo lCoordinateGridRendererCreateInfo{};
        lCoordinateGridRendererCreateInfo.RenderPass = aRenderContext.GetRenderPass();
        mCoordinateGridRenderer = New<CoordinateGridRenderer>( mGraphicContext, aRenderContext, lCoordinateGridRendererCreateInfo );
        mVisualHelperRenderer   = New<VisualHelperRenderer>( mGraphicContext, aRenderContext.GetRenderPass() );
    }

    MeshRendererCreateInfo SceneRenderer::GetRenderPipelineCreateInfo(
        RenderContext &aRenderContext, sMaterialShaderComponent &aPipelineSpecification )
    {
        MeshRendererCreateInfo lCreateInfo;

        lCreateInfo.Opaque         = ( aPipelineSpecification.Type == eCMaterialType::Opaque );
        lCreateInfo.IsTwoSided     = aPipelineSpecification.IsTwoSided;
        lCreateInfo.LineWidth      = aPipelineSpecification.LineWidth;
        lCreateInfo.VertexShader   = "Shaders\\PBRMeshShader.vert.spv";
        lCreateInfo.FragmentShader = "Shaders\\PBRMeshShader.frag.spv";
        lCreateInfo.RenderPass     = aRenderContext.GetRenderPass();

        return lCreateInfo;
    }

    MeshRenderer &SceneRenderer::GetRenderPipeline(
        RenderContext &aRenderContext, MeshRendererCreateInfo const &aPipelineSpecification )
    {
        if( mMeshRenderers.find( aPipelineSpecification ) == mMeshRenderers.end() )
            mMeshRenderers[aPipelineSpecification] = MeshRenderer( mGraphicContext, aPipelineSpecification );

        return mMeshRenderers[aPipelineSpecification];
    }

    MeshRenderer &SceneRenderer::GetRenderPipeline( RenderContext &aRenderContext, sMaterialShaderComponent &aPipelineSpecification )
    {
        MeshRendererCreateInfo lCreateInfo = GetRenderPipelineCreateInfo( aRenderContext, aPipelineSpecification );

        return GetRenderPipeline( aRenderContext, lCreateInfo );
    }

    ParticleSystemRenderer &SceneRenderer::GetRenderPipeline(
        RenderContext &aRenderContext, sParticleShaderComponent &aPipelineSpecification )
    {
        ParticleRendererCreateInfo lCreateInfo = GetRenderPipelineCreateInfo( aRenderContext, aPipelineSpecification );

        if( mParticleRenderers.find( lCreateInfo ) == mParticleRenderers.end() )
            mParticleRenderers[lCreateInfo] = ParticleSystemRenderer( mGraphicContext, aRenderContext, lCreateInfo );

        return mParticleRenderers[lCreateInfo];
    }

    ParticleRendererCreateInfo SceneRenderer::GetRenderPipelineCreateInfo(
        RenderContext &aRenderContext, sParticleShaderComponent &aPipelineSpecification )
    {
        ParticleRendererCreateInfo lCreateInfo;
        lCreateInfo.LineWidth      = aPipelineSpecification.LineWidth;
        lCreateInfo.VertexShader   = "Shaders\\ParticleSystem.vert.spv";
        lCreateInfo.FragmentShader = "Shaders\\ParticleSystem.frag.spv";
        lCreateInfo.RenderPass     = aRenderContext.GetRenderPass();
        ;

        return lCreateInfo;
    }

    void SceneRenderer::Render( RenderContext &aRenderContext )
    {
        LTSE_PROFILE_FUNCTION();

        UpdateDescriptorSets( aRenderContext );
        mWorld->GetMaterialSystem()->UpdateDescriptors();

        int lDirectionalLightCount = 0;
        int lSpotlightCount        = 0;
        int lPointLightCount       = 0;

        mWorld->ForEach<sDirectionalLightComponent>(
            [&]( auto aEntity, auto &aComponent )
            {
                View.DirectionalLights[lDirectionalLightCount] = DirectionalLightData( aComponent, math::mat4( 1.0f ) );
                lDirectionalLightCount++;
            } );

        mWorld->ForEach<sPointLightComponent>(
            [&]( auto aEntity, auto &aComponent )
            {
                math::mat4 l_TransformMatrix = math::mat4( 1.0f );
                if( aEntity.Has<sTransformMatrixComponent>() ) l_TransformMatrix = aEntity.Get<sTransformMatrixComponent>().Matrix;

                View.PointLights[lPointLightCount] = PointLightData( aComponent, l_TransformMatrix );
                lPointLightCount++;
            } );

        mWorld->ForEach<sSpotlightComponent>(
            [&]( auto aEntity, auto &aComponent )
            {
                math::mat4 l_TransformMatrix = math::mat4( 1.0f );
                if( aEntity.Has<sTransformMatrixComponent>() ) l_TransformMatrix = aEntity.Get<sTransformMatrixComponent>().Matrix;

                View.Spotlights[lSpotlightCount] = SpotlightData( aComponent, l_TransformMatrix );
                lSpotlightCount++;
            } );

        View.PointLightCount       = lPointLightCount;
        View.DirectionalLightCount = lDirectionalLightCount;
        View.SpotlightCount        = lSpotlightCount;

        if( mWorld->Environment.Has<sAmbientLightingComponent>() )
        {
            auto &lComponent = mWorld->Environment.Get<sAmbientLightingComponent>();

            Settings.AmbientLightIntensity = lComponent.Intensity;
            Settings.AmbientLightColor     = math::vec4( lComponent.Color, 0.0 );
        }

        mCameraUniformBuffer->Write( View );
        mShaderParametersBuffer->Write( Settings );

        std::unordered_map<MeshRendererCreateInfo, std::vector<Entity>, MeshRendererCreateInfoHash> lOpaqueMeshQueue{};
        mWorld->ForEach<sStaticMeshComponent, sMaterialShaderComponent>(
            [&]( auto aEntity, auto &aStaticMeshComponent, auto &aMaterialData )
            {
                auto &l_PipelineCreateInfo = GetRenderPipelineCreateInfo( aRenderContext, aMaterialData );
                if( lOpaqueMeshQueue.find( l_PipelineCreateInfo ) == lOpaqueMeshQueue.end() )
                    lOpaqueMeshQueue[l_PipelineCreateInfo] = std::vector<Entity>{};
                lOpaqueMeshQueue[l_PipelineCreateInfo].push_back( aEntity );
            } );

        if( mWorld->mVertexBuffer && mWorld->mIndexBuffer )
        {
            aRenderContext.Bind( mWorld->mTransformedVertexBuffer, mWorld->mIndexBuffer );
            for( auto &lPipelineData : lOpaqueMeshQueue )
            {
                auto &lPipeline = GetRenderPipeline( aRenderContext, lPipelineData.first );
                if( lPipeline.Pipeline )
                    aRenderContext.Bind( lPipeline.Pipeline );
                else
                    continue;

                aRenderContext.Bind( mSceneDescriptors, 0, -1 );
                aRenderContext.Bind( mWorld->GetMaterialSystem()->GetDescriptorSet(), 1, -1 );

                for( auto &lMeshInformation : lPipelineData.second )
                {
                    if( lMeshInformation.Has<NodeDescriptorComponent>() )
                        aRenderContext.Bind( lMeshInformation.Get<NodeDescriptorComponent>().Descriptors, 2, -1 );

                    MeshRenderer::MaterialPushConstants l_MaterialPushConstants{};
                    l_MaterialPushConstants.mMaterialID = lMeshInformation.Get<sMaterialComponent>().mMaterialID;

                    aRenderContext.PushConstants(
                        { Graphics::Internal::eShaderStageTypeFlags::FRAGMENT }, 0, l_MaterialPushConstants );

                    auto &l_StaticMeshComponent = lMeshInformation.Get<sStaticMeshComponent>();
                    aRenderContext.Draw( l_StaticMeshComponent.mIndexCount, l_StaticMeshComponent.mIndexOffset,
                        l_StaticMeshComponent.mVertexOffset, 1, 0 );
                }
            }
        }

        mWorld->ForEach<sParticleSystemComponent, sParticleShaderComponent>(
            [&]( auto aEntity, auto &aParticleSystemComponent, auto &aParticleShaderComponent )
            {
                auto &lPipeline = GetRenderPipeline( aRenderContext, aParticleShaderComponent );

                ParticleSystemRenderer::ParticleData lParticleData{};
                lParticleData.Model         = math::mat4( 1.0f );
                lParticleData.ParticleCount = aParticleSystemComponent.ParticleCount;
                lParticleData.ParticleSize  = aParticleSystemComponent.ParticleSize;
                lParticleData.Particles     = aParticleSystemComponent.Particles;

                lPipeline.Render( View.Projection, View.View, aRenderContext, lParticleData );
            } );

        if( RenderGizmos )
        {
            mVisualHelperRenderer->View       = View.View;
            mVisualHelperRenderer->Projection = View.Projection;
            mWorld->ForEach<DirectionalLightHelperComponent>(
                [&]( auto aEntity, auto &a_DirectionalLightHelperComponent )
                {
                    math::mat4 l_Transform = math::mat4( 1.0f );
                    if( aEntity.Has<sTransformMatrixComponent>() ) l_Transform = aEntity.Get<sTransformMatrixComponent>().Matrix;
                    mVisualHelperRenderer->Render( l_Transform, a_DirectionalLightHelperComponent, aRenderContext );
                } );

            mWorld->ForEach<SpotlightHelperComponent>(
                [&]( auto aEntity, auto &a_SpotlightHelperComponent )
                {
                    math::mat4 l_Transform = math::mat4( 1.0f );
                    if( aEntity.Has<sTransformMatrixComponent>() ) l_Transform = aEntity.Get<sTransformMatrixComponent>().Matrix;
                    mVisualHelperRenderer->Render( l_Transform, a_SpotlightHelperComponent, aRenderContext );
                } );

            mWorld->ForEach<PointLightHelperComponent>(
                [&]( auto aEntity, auto &a_PointLightHelperComponent )
                {
                    math::mat4 l_Transform = math::mat4( 1.0f );
                    if( aEntity.Has<sTransformMatrixComponent>() ) l_Transform = aEntity.Get<sTransformMatrixComponent>().Matrix;
                    mVisualHelperRenderer->Render( l_Transform, a_PointLightHelperComponent, aRenderContext );
                } );
        }

        if( RenderCoordinateGrid ) mCoordinateGridRenderer->Render( View.Projection, View.View, aRenderContext );
    }

    void SceneRenderer::UpdateDescriptorSets( RenderContext &aRenderContext )
    {
        LTSE_PROFILE_FUNCTION();

        mWorld->ForEach<sTransformMatrixComponent>(
            [&]( auto aEntity, auto &aComponent )
            {
                if( !( aEntity.Has<NodeDescriptorComponent>() ) )
                {
                    auto &l_NodeDescriptor = aEntity.Add<NodeDescriptorComponent>();
                    l_NodeDescriptor.Descriptors =
                        New<DescriptorSet>( mGraphicContext, MeshRenderer::GetNodeSetLayout( mGraphicContext ) );
                    l_NodeDescriptor.UniformBuffer = New<Buffer>(
                        mGraphicContext, eBufferBindType::UNIFORM_BUFFER, true, false, true, true, sizeof( NodeMatrixDataComponent ) );

                    l_NodeDescriptor.Descriptors->Write(
                        l_NodeDescriptor.UniformBuffer, false, 0, sizeof( NodeMatrixDataComponent ), 0 );
                }

                auto                   &l_NodeDescriptor = aEntity.Get<NodeDescriptorComponent>();
                NodeMatrixDataComponent l_NodeTransform{};
                l_NodeTransform.Transform = aComponent.Matrix;
                aEntity.IfExists<sSkeletonComponent>(
                    [&]( auto &l_SkeletonComponent )
                    {
                        l_NodeTransform.JointCount = l_SkeletonComponent.BoneCount;
                        for( uint32_t i = 0; i < l_SkeletonComponent.BoneCount; i++ )
                            l_NodeTransform.Joints[i] = l_SkeletonComponent.JointMatrices[i];
                    } );

                l_NodeDescriptor.UniformBuffer->Write( l_NodeTransform );
            } );
    }

} // namespace LTSE::Core