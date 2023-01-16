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

    void ASceneRenderer::SetProjection( mat4 aProjectionMatrix )
    {
        mProjectionMatrix = aProjectionMatrix;
        mProjectionMatrix[1][1] *= -1.0f;
    }

    void ASceneRenderer::SetView( mat4 aViewMatrix )
    {
        mViewMatrix     = aViewMatrix;
        mCameraPosition = vec3( Inverse( mViewMatrix )[3] );
    }

    void ASceneRenderer::SetGamma( float aGamma ) { mGamma = aGamma; }

    void ASceneRenderer::SetExposure( float aExposure ) { mExposure = aExposure; }

    void ASceneRenderer::SetAmbientLighting( vec4 aAmbientLight ) { mAmbientLight = aAmbientLight; }

    void ASceneRenderer::ResizeOutput( uint32_t aOutputWidth, uint32_t aOutputHeight )
    {
        mOutputWidth  = aOutputWidth;
        mOutputHeight = aOutputHeight;
    }

    void ASceneRenderer::AddLightGizmo( mat4 const &aTransform, sLightComponent &aLightComponent )
    {
        switch( aLightComponent.mType )
        {
        case eLightType::DIRECTIONAL:
            mDirectionalLights.emplace_back( aLightComponent, aTransform );
            mLightGizmos.emplace_back( eLightType::DIRECTIONAL, mDirectionalLights.size() - 1, aTransform );
            break;
        case eLightType::POINT_LIGHT:
            mPointLights.emplace_back( aLightComponent, aTransform );
            mLightGizmos.emplace_back( eLightType::POINT_LIGHT, mPointLights.size() - 1, aTransform );
            break;
        case eLightType::SPOTLIGHT:
            mSpotlights.emplace_back( aLightComponent, aTransform );
            mLightGizmos.emplace_back( eLightType::SPOTLIGHT, mSpotlights.size() - 1, aTransform );
            break;
        }
    }

    void ASceneRenderer::Update( Ref<Scene> aScene )
    {
        SE_PROFILE_FUNCTION( "FOO" );

        mScene = aScene;

        if( !mScene ) return;

        if( mScene->Environment.Has<sAmbientLightingComponent>() )
        {
            auto &lAmbientLightComponent = mScene->Environment.Get<sAmbientLightingComponent>();
            SetAmbientLighting( vec4( lAmbientLightComponent.Color, lAmbientLightComponent.Intensity ) );
        }

        mDirectionalLights.clear();
        mPointLights.clear();
        mSpotlights.clear();
        mLightGizmos.clear();
        mScene->ForEach<sLightComponent>(
            [&]( auto aEntity, auto &aComponent )
            {
                mat4 lTransformMatrix = mScene->GetFinalTransformMatrix( aEntity );

                AddLightGizmo( lTransformMatrix, aComponent );
            } );

        mOpaqueMeshQueue.clear();
        mTransparentMeshQueue.clear();
        mScene->ForEach<sStaticMeshComponent, sMaterialComponent, sMaterialShaderComponent>(
            [&]( auto aEntity, auto &aStaticMeshComponent, auto &aMaterial, auto &aMaterialData )
            {
                if( aMaterialData.Type == eCMaterialType::Opaque )
                    mOpaqueMeshQueue.emplace_back( aStaticMeshComponent, aMaterial, aMaterialData );
                else
                    mTransparentMeshQueue.emplace_back( aStaticMeshComponent, aMaterial, aMaterialData );
            } );

        mParticleQueue.clear();
        mScene->ForEach<sParticleSystemComponent, sParticleShaderComponent>(
            [&]( auto aEntity, auto &aParticleSystemComponent, auto &aParticleShaderComponent )
            { mParticleQueue.emplace_back( aParticleSystemComponent, aParticleShaderComponent ); } );
    }

    void ASceneRenderer::Render() {}

} // namespace SE::Core