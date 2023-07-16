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

    BaseSceneRenderer::BaseSceneRenderer( Ref<IGraphicContext> aGraphicContext, eColorFormat aOutputFormat, uint32_t aOutputSampleCount )
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

    void BaseSceneRenderer::SetGamma( float aGamma ) { mGamma = aGamma; }

    void BaseSceneRenderer::SetExposure( float aExposure ) { mExposure = aExposure; }

    void BaseSceneRenderer::SetAmbientLighting( vec4 aAmbientLight ) { mAmbientLight = aAmbientLight; }

    void BaseSceneRenderer::ResizeOutput( uint32_t aOutputWidth, uint32_t aOutputHeight )
    {
        mOutputWidth  = aOutputWidth;
        mOutputHeight = aOutputHeight;
    }

    void BaseSceneRenderer::AddLight( mat4 const &aTransform, sLightComponent &aLightComponent )
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
        // case eLightType::SPOTLIGHT:
        //     mSpotlights.emplace_back( aLightComponent, aTransform );
        //     mLightGizmos.emplace_back( eLightType::SPOTLIGHT, mSpotlights.size() - 1, aTransform );
        //     break;
        }
    }

    void BaseSceneRenderer::Update( Ref<Scene> aScene )
    {
        SE_PROFILE_SCOPE( "FOO" );

        mScene = aScene;

        if( !mScene ) return;

        if( mScene->Environment.Has<sAmbientLightingComponent>() )
        {
            auto &lAmbientLightComponent = mScene->Environment.Get<sAmbientLightingComponent>();
            SetAmbientLighting( vec4( lAmbientLightComponent.Color, lAmbientLightComponent.Intensity ) );
        }

        // clang-format off
        mDirectionalLights.clear();
        mPointLights.clear();
        // mSpotlights.clear();
        mLightGizmos.clear();
        mScene->ForEach<sLightComponent>( [&]( auto aEntity, auto &aComponent ) { 
            AddLight( mScene->GetFinalTransformMatrix( aEntity ), aComponent ); 
        } );
        // clang-format on

        // clang-format off
        mOpaqueMeshQueue.clear();
        mTransparentMeshQueue.clear();
        mScene->ForEach<sStaticMeshComponent, sMaterialComponent, sMaterialShaderComponent>(
            [&]( auto aEntity, auto &aStaticMeshComponent, auto &aMaterial, auto &aMaterialData ) {
                if( aMaterialData.Type == eMaterialType::Opaque )
                    mOpaqueMeshQueue.emplace_back( aStaticMeshComponent, aMaterial, aMaterialData );
                else
                    mTransparentMeshQueue.emplace_back( aStaticMeshComponent, aMaterial, aMaterialData );
            });
        // clang-format on

        // clang-format off
        mParticleQueue.clear();
        mScene->ForEach<sParticleSystemComponent, sParticleShaderComponent>(
            [&]( auto aEntity, auto &aParticleSystemComponent, auto &aParticleShaderComponent ) { 
                mParticleQueue.emplace_back( aParticleSystemComponent, aParticleShaderComponent ); 
            } );
        // clang-format on

    }

    void BaseSceneRenderer::Render() {}

} // namespace SE::Core