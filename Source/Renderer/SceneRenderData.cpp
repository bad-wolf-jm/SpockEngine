#include "SceneRenderData.h"

#include <chrono>
#include <gli/gli.hpp>

#include "Graphics/Vulkan/VkPipeline.h"

#include "Scene/Components/VisualHelpers.h"
#include "Scene/Primitives/Primitives.h"
#include "Scene/VertexData.h"

#include "Core/Profiling/BlockTimer.h"

#include "Core/Logging.h"
#include "Core/Resource.h"

namespace SE::Core
{

    using namespace math;
    using namespace SE::Core::EntityComponentSystem::Components;
    using namespace SE::Core::Primitives;

    DirectionalLightData::DirectionalLightData( const sLightComponent &aSpec, mat4 aTransform )
    {
        Direction = mat3( aTransform ) * vec3{ 0.0f, 1.0f, 0.0f };
        Color     = aSpec.mColor;
        Intensity = aSpec.mIntensity;

        // clang-format off
        const float aEntries[] = { 1.0f,  0.0f, 0.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f,  0.0f, 1.0f, 0.0f, 0.0f,  0.0f, 0.0f, 1.0f };
        math::mat4  lClip = math::MakeMat4( aEntries );
        // clang-format on

        math::mat4 lProjection =
            math::Orthogonal( math::vec2{ -10.0f, 10.0f }, math::vec2{ -10.0f, 10.0f }, math::vec2{ -10.0f, 10.0f } );
        math::mat4 lView = math::LookAt( Direction * 5.0f, math::vec3{ 0.0f, 0.0f, 0.0f }, math::vec3{ 0.0f, 1.0f, 0.0f } );
        Transform        = lClip * lProjection * lView;
    }

    PointLightData::PointLightData( const sLightComponent &aSpec, mat4 aTransform )
    {
        WorldPosition = vec3( aTransform[3] );
        Color         = aSpec.mColor;
        Intensity     = aSpec.mIntensity;
    }

    SpotlightData::SpotlightData( const sLightComponent &aSpec, mat4 aTransform )
    {
        WorldPosition   = vec3( aTransform[3] );
        LookAtDirection = -vec3{ aTransform[0][2], aTransform[1][2], aTransform[2][2] };
        Color           = aSpec.mColor;
        Intensity       = aSpec.mIntensity;
        Cone            = math::cos( radians( aSpec.mCone / 2 ) );

        // clang-format off
        const float aEntries[] = { 1.0f,  0.0f, 0.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f,  0.0f, 1.0f, 0.0f, 0.0f,  0.0f, 0.0f, 1.0f };
        math::mat4  lClip = math::MakeMat4( aEntries );
        // clang-format on

        math::mat4 lProjection = math::Perspective( math::radians( aSpec.mCone ), 1.0f, 0.0001f, 10.0f );
        math::mat4 lView       = math::LookAt( WorldPosition, WorldPosition + LookAtDirection, math::vec3{ 0.0f, 1.0f, 0.0f } );

        Transform = lClip * lProjection * lView;
    }

    sLightGizmo::sLightGizmo( eLightType aType, uint64_t aLightDataIndex, mat4 aMatrix )
        : mType{ aType }
        , mLightDataIndex{ aLightDataIndex }
        , mMatrix{ aMatrix }
    {
    }

    sMeshRenderData::sMeshRenderData( sStaticMeshComponent const &aMesh, sMaterialComponent const &aMaterialID,
                                      sMaterialShaderComponent const &aShader )
        : mOpaque{ ( aShader.Type == eCMaterialType::Opaque ) }
        , mIsTwoSided{ aShader.IsTwoSided }
        , mLineWidth{ aShader.LineWidth }
        , mMaterialID{ aMaterialID.mMaterialID }
        , mIndexBuffer{ aMesh.mIndexBuffer }
        , mVertexBuffer{ aMesh.mTransformedBuffer }
        , mVertexOffset{ aMesh.mVertexOffset }
        , mVertexCount{ aMesh.mVertexCount }
        , mIndexOffset{ aMesh.mIndexOffset }
        , mIndexCount{ aMesh.mIndexCount }
    {
    }

    sParticleRenderData::sParticleRenderData( sParticleSystemComponent const &aParticles, sParticleShaderComponent const &aShader )
        : mModel{ mat4( 1.0f ) }
        , mParticleCount{ aParticles.ParticleCount }
        , mLineWidth{ aShader.LineWidth }
        , mParticleSize{ aParticles.ParticleSize }
        , mParticles{ aParticles.Particles }

    {
    }
} // namespace SE::Core