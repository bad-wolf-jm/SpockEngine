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

    BaseSceneRenderer::BaseSceneRenderer( ref_t<IGraphicContext> aGraphicContext, eColorFormat aOutputFormat,
                                          uint32_t aOutputSampleCount )
        : mGraphicContext{ aGraphicContext }
        , mOutputFormat{ aOutputFormat }
        , mOutputSampleCount{ aOutputSampleCount }
    {
    }

    void BaseSceneRenderer::SetProjection( mat4 aProjectionMatrix )
    {
        mProjectionMatrix = aProjectionMatrix;
        mProjectionMatrix[1][1] *= -1.0f;
    }

    void BaseSceneRenderer::SetView( mat4 aViewMatrix )
    {
        mViewMatrix     = aViewMatrix;
        mCameraPosition = vec3( Inverse( mViewMatrix )[3] );
    }

    void BaseSceneRenderer::SetGamma( float aGamma )
    {
        mGamma = aGamma;
    }

    void BaseSceneRenderer::SetExposure( float aExposure )
    {
        mExposure = aExposure;
    }

    void BaseSceneRenderer::SetAmbientLighting( vec4 aAmbientLight )
    {
        mAmbientLight = aAmbientLight;
    }

    void BaseSceneRenderer::ResizeOutput( uint32_t aOutputWidth, uint32_t aOutputHeight )
    {
        mOutputWidth  = aOutputWidth;
        mOutputHeight = aOutputHeight;
    }

    void BaseSceneRenderer::Update( ref_t<Scene> aScene )
    {
        SE_PROFILE_SCOPE( "FOO" );

        mScene = aScene;

        if( !mScene )
            return;
    }

    void BaseSceneRenderer::Render()
    {
    }
} // namespace SE::Core