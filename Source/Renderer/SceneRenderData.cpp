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

    DirectionalLightData::DirectionalLightData( const sLightComponent &aSpec, math::mat4 aTransform )
    {
        Direction = math::mat3( aTransform ) * math::vec3{ 0.0f, 1.0f, 0.0f };
        Color     = aSpec.mColor;
        Intensity = aSpec.mIntensity;
    }

    PointLightData::PointLightData( const sLightComponent &aSpec, math::mat4 aTransform )
    {
        WorldPosition = math::vec3( aTransform[3] );
        Color         = aSpec.mColor;
        Intensity     = aSpec.mIntensity;
    }

    SpotlightData::SpotlightData( const sLightComponent &aSpec, math::mat4 aTransform )
    {
        WorldPosition   = math::vec3( aTransform[3] ); // * math::vec4( aSpec.Position, 1.0f );
        LookAtDirection = math::mat3( aTransform ) * math::vec3{ 0.0f, -1.0f, 0.0f };
        Color           = aSpec.mColor;
        Intensity       = aSpec.mIntensity;
        Cone            = math::cos( math::radians( aSpec.mCone / 2 ) );
    }

    sLightGizmo::sLightGizmo( eLightType aType, uint64_t aLightDataIndex, math::mat4 aMatrix )
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
        : mModel{ math::mat4( 1.0f ) }
        , mParticleCount{ aParticles.ParticleCount }
        , mLineWidth{ aShader.LineWidth }
        , mParticleSize{ aParticles.ParticleSize }
        , mParticles{ aParticles.Particles }

    {
    }
} // namespace SE::Core