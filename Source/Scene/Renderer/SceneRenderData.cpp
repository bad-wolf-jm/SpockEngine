#include "SceneRenderer.h"

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

namespace LTSE::Core
{

    using namespace math;
    using namespace LTSE::Core::EntityComponentSystem::Components;
    using namespace LTSE::Core::Primitives;

    DirectionalLightData::DirectionalLightData( const sDirectionalLightComponent &aSpec, math::mat4 aTransform )
    {
        float lAzimuth   = math::radians( aSpec.Azimuth );
        float lElevation = math::radians( aSpec.Elevation );

        Direction = math::vec3{ math::sin( lElevation ) * math::cos( lAzimuth ), math::cos( lElevation ),
            math::sin( lElevation ) * math::sin( lAzimuth ) };
        Color     = aSpec.Color;
        Intensity = aSpec.Intensity;
    }

    PointLightData::PointLightData( const sPointLightComponent &aSpec, math::mat4 aTransform )
    {
        WorldPosition = aTransform * math::vec4( aSpec.Position, 1.0f );
        Color         = aSpec.Color;
        Intensity     = aSpec.Intensity;
    }

    SpotlightData::SpotlightData( const sSpotlightComponent &aSpec, math::mat4 aTransform )
    {
        float lAzimuth   = math::radians( aSpec.Azimuth );
        float lElevation = math::radians( aSpec.Elevation );

        WorldPosition   = aTransform * math::vec4( aSpec.Position, 1.0f );
        LookAtDirection = math::vec3{ math::sin( lElevation ) * math::cos( lAzimuth ), math::cos( lElevation ),
            math::sin( lElevation ) * math::sin( lAzimuth ) };
        Color           = aSpec.Color;
        Intensity       = aSpec.Intensity;
        Cone            = math::cos( math::radians( aSpec.Cone / 2 ) );
    }
} // namespace LTSE::Core