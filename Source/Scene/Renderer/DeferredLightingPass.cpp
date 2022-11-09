#include "DeferredLightingPass.h"

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

    DeferredLightingPass::DeferredLightingPass( Ref<Scene> aWorld, DeferredRenderContext &aRenderContext )
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

        DeferredLightingRendererCreateInfo mLightingRendererCI{};
        mLightingRendererCI.RenderPass = aRenderContext.GetLightingRenderPass();
        mRenderer = DeferredLightingRenderer( mGraphicContext, mLightingRendererCI );
    }

    void DeferredLightingPass::Render( DeferredRenderContext &aRenderContext )
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
    }

    void DeferredLightingPass::UpdateDescriptorSets( DeferredRenderContext &aRenderContext )
    {
        LTSE_PROFILE_FUNCTION();
    }

} // namespace LTSE::Core