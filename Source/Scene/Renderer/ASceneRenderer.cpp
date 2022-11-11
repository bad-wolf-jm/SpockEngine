#include "ASceneRenderer.h"

#include <chrono>
#include <gli/gli.hpp>

#include "Core/Profiling/BlockTimer.h"

#include "Core/Logging.h"
#include "Core/Resource.h"

namespace LTSE::Core
{
    using namespace math;
    using namespace LTSE::Core::EntityComponentSystem::Components;

    void ASceneRenderer::SetProjection( math::mat4 aProjectionMatrix ) { mProjectionMatrix = aProjectionMatrix; }

    void ASceneRenderer::SetView( math::mat4 aViewMatrix ) { mViewMatrix = aViewMatrix; }

    void ASceneRenderer::SetGamma( float aGamma ) { mGamma = aGamma; }

    void ASceneRenderer::SetExposure( float aExposure ) { mExposure = aExposure; }

    void ASceneRenderer::SetAmbientLighting( math::vec4 aAmbientLight ) { mAmbientLight = aAmbientLight; }

    void ASceneRenderer::Update( Ref<Scene> aScene )
    {
        LTSE_PROFILE_FUNCTION();

        if( !aScene ) return;

        // if( aScene->Environment.Has<sAmbientLightingComponent>() )
        // {
        //     auto &lComponent = aScene->Environment.Get<sAmbientLightingComponent>();

        //     mAmbientLight = vec4( lComponent.Color, lComponent.Intensity );
        // }

        mDirectionalLights.clear();
        mPointLights.clear();
        mSpotlights.clear();

        aScene->ForEach<sDirectionalLightComponent>(
            [&]( auto aEntity, auto &aComponent )
            {
                mat4 lTransformMatrix = mat4( 1.0f );
                if( aEntity.Has<sTransformMatrixComponent>() ) lTransformMatrix = aEntity.Get<sTransformMatrixComponent>().Matrix;

                mDirectionalLights.emplace_back( aComponent, lTransformMatrix );
            } );

        aScene->ForEach<sPointLightComponent>(
            [&]( auto aEntity, auto &aComponent )
            {
                mat4 lTransformMatrix = mat4( 1.0f );
                if( aEntity.Has<sTransformMatrixComponent>() ) lTransformMatrix = aEntity.Get<sTransformMatrixComponent>().Matrix;

                mPointLights.emplace_back( aComponent, lTransformMatrix );
            } );

        aScene->ForEach<sSpotlightComponent>(
            [&]( auto aEntity, auto &aComponent )
            {
                mat4 lTransformMatrix = mat4( 1.0f );
                if( aEntity.Has<sTransformMatrixComponent>() ) lTransformMatrix = aEntity.Get<sTransformMatrixComponent>().Matrix;

                mSpotlights.emplace_back( aComponent, lTransformMatrix );
            } );
    }

    // void ASceneRenderer::Update( mat4 aProjectionMatrix, mat4 aViewMatrix )
    // {
    //     LTSE_PROFILE_FUNCTION();

    //     mProjectionMatrix = aProjectionMatrix;
    //     mViewMatrix       = aViewMatrix;
    // }

    // void ASceneRenderer::Update( Ref<Scene> aScene, mat4 aProjectionMatrix, mat4 aViewMatrix )
    // {
    //     LTSE_PROFILE_FUNCTION();

    //     Update( aProjectionMatrix, aViewMatrix );
    //     mProjectionMatrix = aProjectionMatrix;
    //     mViewMatrix       = aViewMatrix;

    //     Update( aScene );
    // }
} // namespace LTSE::Core