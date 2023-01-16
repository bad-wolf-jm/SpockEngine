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
        mLightGizmos.clear();

        mScene->ForEach<sLightComponent>(
            [&]( auto aEntity, auto &aComponent )
            {
                mat4 lTransformMatrix = mScene->GetFinalTransformMatrix( aEntity );

                switch( aComponent.mType )
                {
                case eLightType::DIRECTIONAL:
                {
                    mDirectionalLights.emplace_back( aComponent, lTransformMatrix );
                    mLightGizmos.emplace_back( eLightType::DIRECTIONAL, mDirectionalLights.size() - 1, lTransformMatrix );
                    break;
                }
                case eLightType::POINT_LIGHT:
                {
                    mPointLights.emplace_back( aComponent, lTransformMatrix );
                    mLightGizmos.emplace_back( eLightType::POINT_LIGHT, mPointLights.size() - 1, lTransformMatrix );
                    break;
                }
                case eLightType::SPOTLIGHT:
                {
                    mSpotlights.emplace_back( aComponent, lTransformMatrix );
                    mLightGizmos.emplace_back( eLightType::SPOTLIGHT, mSpotlights.size() - 1, lTransformMatrix );
                    break;
                }
                }
            } );

        mOpaqueMeshQueue.clear();
        mScene->ForEach<sStaticMeshComponent, sMaterialShaderComponent>(
            [&]( auto aEntity, auto &aStaticMeshComponent, auto &aMaterialData )
            {
                MaterialShaderCreateInfo lMaterialShaderCreateInfo{};
                lMaterialShaderCreateInfo.Opaque     = ( aMaterialData.Type == eCMaterialType::Opaque );
                lMaterialShaderCreateInfo.IsTwoSided = aMaterialData.IsTwoSided;
                lMaterialShaderCreateInfo.LineWidth  = aMaterialData.LineWidth;

                if( mOpaqueMeshQueue.find( lMaterialShaderCreateInfo ) == mOpaqueMeshQueue.end() )
                    mOpaqueMeshQueue[lMaterialShaderCreateInfo] = std::vector<sStaticMeshComponent>{};
                mOpaqueMeshQueue[lMaterialShaderCreateInfo].push_back( aStaticMeshComponent );
            } );

        mScene->ForEach<sParticleSystemComponent, sParticleShaderComponent>(
            [&]( auto aEntity, auto &aParticleSystemComponent, auto &aParticleShaderComponent )
            {
                // auto &lPipeline = GetRenderPipeline( aParticleShaderComponent );

                // ParticleSystemRenderer::ParticleData lParticleData{};
                // lParticleData.Model         = math::mat4( 1.0f );
                // lParticleData.ParticleCount = aParticleSystemComponent.ParticleCount;
                // lParticleData.ParticleSize  = aParticleSystemComponent.ParticleSize;
                // lParticleData.Particles     = aParticleSystemComponent.Particles;

                // lPipeline.Render( View.Projection, View.View, mGeometryContext, lParticleData );
            } );
    }

    void ASceneRenderer::Render() {}

} // namespace SE::Core