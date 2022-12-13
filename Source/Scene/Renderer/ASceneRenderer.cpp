#include "ASceneRenderer.h"

#include <chrono>
#include <gli/gli.hpp>

#include "Core/Profiling/BlockTimer.h"

#include "Core/Logging.h"
#include "Core/Resource.h"

namespace SE::Core
{
    using namespace math;
    using namespace SE::Core::EntityComponentSystem::Components;

    ASceneRenderer::ASceneRenderer( Ref<VkGraphicContext> aGraphicContext, eColorFormat aOutputFormat, uint32_t aOutputSampleCount )
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
        SE_PROFILE_FUNCTION();

        mScene = aScene;

        if( !aScene ) return;

        mDirectionalLights.clear();
        mPointLights.clear();
        mSpotlights.clear();

        aScene->ForEach<sLightComponent>(
            [&]( auto aEntity, auto &aComponent )
            {
                mat4 lTransformMatrix = mat4( 1.0f );
                if( aEntity.Has<sTransformMatrixComponent>() ) lTransformMatrix = aEntity.Get<sTransformMatrixComponent>().Matrix;

                switch( aComponent.mType )
                {
                case eLightType::DIRECTIONAL: mDirectionalLights.emplace_back( aComponent, lTransformMatrix ); break;
                case eLightType::POINT_LIGHT: mPointLights.emplace_back( aComponent, lTransformMatrix ); break;
                case eLightType::SPOTLIGHT: mSpotlights.emplace_back( aComponent, lTransformMatrix ); break;
                }
            } );
    }

    void ASceneRenderer::Render() {}

} // namespace SE::Core