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

    ASceneRenderer::ASceneRenderer( GraphicContext aGraphicContext, eColorFormat aOutputFormat, uint32_t aOutputSampleCount )
        : mGraphicContext{ aGraphicContext }
        , mOutputFormat{ aOutputFormat }
        , mOutputSampleCount{ aOutputSampleCount }
    {
    }

    void ASceneRenderer::SetProjection( math::mat4 aProjectionMatrix )
    {
        mProjectionMatrix = aProjectionMatrix;
        mProjectionMatrix[1][1] *= -1.0f;
    }

    void ASceneRenderer::SetView( math::mat4 aViewMatrix )
    {
        mViewMatrix     = aViewMatrix;
        mCameraPosition = math::vec3( math::Inverse( mViewMatrix )[3] );
    }

    void ASceneRenderer::SetGamma( float aGamma ) { mGamma = aGamma; }

    void ASceneRenderer::SetExposure( float aExposure ) { mExposure = aExposure; }

    void ASceneRenderer::SetAmbientLighting( math::vec4 aAmbientLight ) { mAmbientLight = aAmbientLight; }

    void ASceneRenderer::ResizeOutput( uint32_t aOutputWidth, uint32_t aOutputHeight )
    {
        mOutputWidth  = aOutputWidth;
        mOutputHeight = aOutputHeight;
    }

    void ASceneRenderer::Update( Ref<Scene> aScene )
    {
        LTSE_PROFILE_FUNCTION();

        mScene = aScene;

        if( !aScene ) return;

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

    void ASceneRenderer::Render() {}

} // namespace LTSE::Core