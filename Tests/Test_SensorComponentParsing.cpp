#include <catch2/catch_test_macros.hpp>
#include <filesystem>

#include "TestUtils.h"

#include "Serialize/FileIO.h"
#include "Serialize/SensorAsset.h"
#include "Serialize/SensorComponents.h"

using namespace math;
using namespace TestUtils;
using namespace LTSE::SensorModel;

namespace fs = std::filesystem;

TEST_CASE( "Parse sampler component", "[CORE_SENSOR_MODEL]" )
{
    const std::string lCode = R""""(components:
  sampler_1:
    type: sampler
    name: sampler_1
    data:
        length: 2345
        frequency: 12345678.9
)"""";

    ConfigurationReader lReader = ConfigurationReader( YAML::Load( lCode ) );
    ConfigurationNode lRoot     = lReader.GetRoot()["components.sampler_1"];

    sSensorComponentData lSensorComponent = ReadComponent( lRoot );

    REQUIRE( lSensorComponent.Type() == eComponentType::SAMPLER );
    try
    {
        lSensorComponent.Get<sSamplerComponentData>();
        REQUIRE( true );
    }
    catch( ... )
    {
        REQUIRE( false );
    }

    REQUIRE( lSensorComponent.mName == "sampler_1" );
    REQUIRE( lSensorComponent.Get<sSamplerComponentData>().mLength == 2345 );
    REQUIRE( lSensorComponent.Get<sSamplerComponentData>().mFrequency == 12345678.9f );
}
