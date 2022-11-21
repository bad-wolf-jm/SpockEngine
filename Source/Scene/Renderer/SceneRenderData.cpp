#include "SceneRenderData.h"

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
} // namespace SE::Core