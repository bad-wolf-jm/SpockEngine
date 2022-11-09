#include "DeferredSceneRenderer.h"

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

    DeferredSceneRenderer::DeferredSceneRenderer( Ref<Scene> aWorld, DeferredRenderContext &aRenderContext )
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
    }

    MeshRendererCreateInfo DeferredSceneRenderer::GetRenderPipelineCreateInfo(
        DeferredRenderContext &aRenderContext, sMaterialShaderComponent &aPipelineSpecification )
    {
        MeshRendererCreateInfo lCreateInfo;

        lCreateInfo.Opaque         = ( aPipelineSpecification.Type == eCMaterialType::Opaque );
        lCreateInfo.IsTwoSided     = aPipelineSpecification.IsTwoSided;
        lCreateInfo.LineWidth      = aPipelineSpecification.LineWidth;
        lCreateInfo.VertexShader   = "Shaders\\Deferred\\MRT.vert.spv";
        lCreateInfo.FragmentShader = "Shaders\\Deferred\\MRT.frag.spv";
        lCreateInfo.RenderPass     = aRenderContext.GetRenderPass();

        return lCreateInfo;
    }

    MeshRenderer &DeferredSceneRenderer::GetRenderPipeline(
        DeferredRenderContext &aRenderContext, MeshRendererCreateInfo const &aPipelineSpecification )
    {
        if( mMeshRenderers.find( aPipelineSpecification ) == mMeshRenderers.end() )
            mMeshRenderers[aPipelineSpecification] = MeshRenderer( mGraphicContext, aPipelineSpecification );

        return mMeshRenderers[aPipelineSpecification];
    }

    MeshRenderer &DeferredSceneRenderer::GetRenderPipeline(
        DeferredRenderContext &aRenderContext, sMaterialShaderComponent &aPipelineSpecification )
    {
        MeshRendererCreateInfo lCreateInfo = GetRenderPipelineCreateInfo( aRenderContext, aPipelineSpecification );

        return GetRenderPipeline( aRenderContext, lCreateInfo );
    }

    void DeferredSceneRenderer::Render( DeferredRenderContext &aRenderContext )
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
                math::mat4 lTransformMatrix = math::mat4( 1.0f );
                if( aEntity.Has<sTransformMatrixComponent>() ) lTransformMatrix = aEntity.Get<sTransformMatrixComponent>().Matrix;

                View.DirectionalLights[lDirectionalLightCount] = DirectionalLightData( aComponent, lTransformMatrix );
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

                    MeshRenderer::MaterialPushConstants lMaterialPushConstants{};
                    lMaterialPushConstants.mMaterialID = lMeshInformation.Get<sMaterialComponent>().mMaterialID;

                    aRenderContext.PushConstants( { Graphics::Internal::eShaderStageTypeFlags::FRAGMENT }, 0, lMaterialPushConstants );

                    auto &lStaticMeshComponent = lMeshInformation.Get<sStaticMeshComponent>();
                    aRenderContext.Draw( lStaticMeshComponent.mIndexCount, lStaticMeshComponent.mIndexOffset,
                        lStaticMeshComponent.mVertexOffset, 1, 0 );
                }
            }
        }
    }

    void DeferredSceneRenderer::UpdateDescriptorSets( DeferredRenderContext &aRenderContext )
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