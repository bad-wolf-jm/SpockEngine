#include <catch2/catch_test_macros.hpp>

#include <algorithm>

#include "Core/Memory.h"

#include "Serialize/FileIO.h"

#include "yaml-cpp/yaml.h"

#include "TestUtils.h"

using namespace math;
using namespace TestUtils;
using namespace LTSE;
using namespace LTSE::Core;
using namespace LTSE::SensorModel;

const char *YamlTest0 = R""""(
root:
  first: 2
  second:
    a: 5
    b: 6
    seq: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  third: 9
  fourth:
    r: 12
    g: 13
    b: 14
    a: 15
)"""";

TEST_CASE( "Read YAML configuration", "[CORE_SENSOR_MODEL]" )
{
    ConfigurationReader lTestReader = ConfigurationReader( YAML::Load( YamlTest0 ) );

    SECTION( "Retrieving root node" )
    {
        ConfigurationNode lRoot = lTestReader.GetRoot();
        REQUIRE( true );
    }

    SECTION( "Retrieving root node value" )
    {
        ConfigurationNode lRoot      = lTestReader.GetRoot();
        ConfigurationNode lRootNode = lRoot["root"];
        REQUIRE( true );
    }

    SECTION( "Retrieving nodes by path" )
    {
        ConfigurationNode lRoot = lTestReader.GetRoot();

        REQUIRE( ( ( lRoot["root.first"].As<uint32_t>( 0u ) ) == 2u ) );
        REQUIRE( ( ( lRoot["root.third"].As<uint32_t>( 0u ) ) == 9u ) );
        REQUIRE( ( ( lRoot["root.second.a"].As<uint32_t>( 0u ) ) == 5u ) );
        REQUIRE( ( ( lRoot["root.second.b"].As<uint32_t>( 0u ) ) == 6u ) );
    }

    SECTION( "Non-existing paths yield Null nodes" )
    {
        ConfigurationNode lRoot = lTestReader.GetRoot();
        REQUIRE( lRoot["first"].IsNull() );
        REQUIRE( lRoot["first.x"].IsNull() );
        REQUIRE( lRoot["root.x"].IsNull() );
        REQUIRE( lRoot["root.x.first.x"].IsNull() );
    }

    SECTION( "Retrieving vectors" )
    {
        ConfigurationNode lRoot = lTestReader.GetRoot();

        REQUIRE( ( lRoot["root.fourth"].Vec( { "r", "g" }, math::vec2{ -1.0f, -1.0f } ) == math::vec2{ 12.0f, 13.0f } ) );
        REQUIRE( ( lRoot["root.fourth"].Vec( { "r", "g", "b" }, math::vec3{ -1.0f, -1.0f, -1.0f } ) == math::vec3{ 12.0f, 13.0f, 14.0f } ) );
        REQUIRE( ( lRoot["root.fourth"].Vec( { "r", "g", "b", "a" }, math::vec4{ -1.0f, -1.0f, -1.0f, -1.0f } ) == math::vec4{ 12.0f, 13.0f, 14.0f, 15.0f } ) );
    }

    SECTION( "Retrieving vectors with missing keys" )
    {
        ConfigurationNode lRoot = lTestReader.GetRoot();

        constexpr math::vec2 lValue0 = math::vec2{ -1.0f, -1.0f };
        REQUIRE( ( lRoot["root.fourth"].Vec( { "r", "foo" }, lValue0 ) ==  lValue0 ) );
        REQUIRE( ( lRoot["root.fourth"].Vec( { "foo", "g" }, lValue0 ) ==  lValue0 ) );

        constexpr math::vec3 lValue1 = math::vec3{ -1.0f, -1.0f, -1.0f };
        REQUIRE( ( lRoot["root.fourth"].Vec( { "foo", "g", "b" }, lValue1 ) == lValue1 ) );
        REQUIRE( ( lRoot["root.fourth"].Vec( { "r", "foo", "b" }, lValue1 ) == lValue1 ) );
        REQUIRE( ( lRoot["root.fourth"].Vec( { "r", "g", "foo" }, lValue1 ) == lValue1 ) );
        REQUIRE( ( lRoot["root.fourth"].Vec( { "foo", "foo", "b" }, lValue1 ) == lValue1 ) );
        REQUIRE( ( lRoot["root.fourth"].Vec( { "foo", "g", "foo" }, lValue1 ) == lValue1 ) );
        REQUIRE( ( lRoot["root.fourth"].Vec( { "r", "foo", "foo" }, lValue1 ) == lValue1 ) );
        REQUIRE( ( lRoot["root.fourth"].Vec( { "foo", "foo", "foo" }, lValue1 ) == lValue1 ) );

        constexpr math::vec4 lValue2 = math::vec4{ -1.0f, -1.0f, -1.0f, -1.0f };
        REQUIRE( ( lRoot["root.fourth"].Vec( { "foo", "g", "b", "a" }, lValue2 ) == lValue2 ) );
        REQUIRE( ( lRoot["root.fourth"].Vec( { "r", "foo", "b", "a" }, lValue2 ) == lValue2 ) );
        REQUIRE( ( lRoot["root.fourth"].Vec( { "r", "g", "foo", "a" }, lValue2 ) == lValue2 ) );
        REQUIRE( ( lRoot["root.fourth"].Vec( { "r", "g", "b", "foo" }, lValue2 ) == lValue2 ) );
        REQUIRE( ( lRoot["root.fourth"].Vec( { "foo", "foo", "b", "a" }, lValue2 ) == lValue2 ) );
        REQUIRE( ( lRoot["root.fourth"].Vec( { "foo", "g", "foo", "a" }, lValue2 ) == lValue2 ) );
        REQUIRE( ( lRoot["root.fourth"].Vec( { "foo", "g", "b", "foo" }, lValue2 ) == lValue2 ) );
        REQUIRE( ( lRoot["root.fourth"].Vec( { "foo", "foo", "foo", "a" }, lValue2 ) == lValue2 ) );
        REQUIRE( ( lRoot["root.fourth"].Vec( { "foo", "foo", "b", "foo" }, lValue2 ) == lValue2 ) );
        REQUIRE( ( lRoot["root.fourth"].Vec( { "foo", "foo", "foo", "foo" }, lValue2 ) == lValue2 ) );
    }

    SECTION( "Retrieving vectors with paths" )
    {
        ConfigurationNode lRoot = lTestReader.GetRoot();

        REQUIRE( ( lRoot.Vec( { "root.fourth.r", "root.fourth.g" }, math::vec2{ -1.0f, -1.0f } ) == math::vec2{ 12.0f, 13.0f } ) );
        REQUIRE( ( lRoot.Vec( { "root.fourth.r", "root.fourth.g", "root.fourth.b" }, math::vec3{ -1.0f, -1.0f, -1.0f } ) == math::vec3{ 12.0f, 13.0f, 14.0f } ) );
        REQUIRE( ( lRoot.Vec( { "root.fourth.r", "root.fourth.g", "root.fourth.b", "root.fourth.a" }, math::vec4{ -1.0f, -1.0f, -1.0f, -1.0f } ) ==
                   math::vec4{ 12.0f, 13.0f, 14.0f, 15.0f } ) );
    }

    SECTION( "Retrieving vectors with paths and missing elements" )
    {
        ConfigurationNode lRoot = lTestReader.GetRoot();

        constexpr math::vec2 lValue0 = math::vec2{ -1.0f, -1.0f };
        REQUIRE( ( lRoot["root.fourth"].Vec( { "root.fourth.r", "root.foo.r" }, lValue0 ) ==  lValue0 ) );
        REQUIRE( ( lRoot["root.fourth"].Vec( { "root.foo.r", "root.fourth.g" }, lValue0 ) ==  lValue0 ) );

        constexpr math::vec3 lValue1 = math::vec3{ -1.0f, -1.0f, -1.0f };
        REQUIRE( ( lRoot["root.fourth"].Vec( { "root.foo.r", "root.fourth.g", "root.fourth.b" }, lValue1 ) == lValue1 ) );
        REQUIRE( ( lRoot["root.fourth"].Vec( { "root.fourth.r", "root.foo.r", "root.fourth.b" }, lValue1 ) == lValue1 ) );
        REQUIRE( ( lRoot["root.fourth"].Vec( { "root.fourth.r", "root.fourth.g", "root.foo.r" }, lValue1 ) == lValue1 ) );
        REQUIRE( ( lRoot["root.fourth"].Vec( { "root.foo.r", "root.foo.r", "root.fourth.b" }, lValue1 ) == lValue1 ) );
        REQUIRE( ( lRoot["root.fourth"].Vec( { "root.foo.r", "root.fourth.g", "root.foo.r" }, lValue1 ) == lValue1 ) );
        REQUIRE( ( lRoot["root.fourth"].Vec( { "root.fourth.r", "root.foo.r", "root.foo.r" }, lValue1 ) == lValue1 ) );
        REQUIRE( ( lRoot["root.fourth"].Vec( { "root.foo.r", "root.foo.r", "root.foo.r" }, lValue1 ) == lValue1 ) );

        constexpr math::vec4 lValue2 = math::vec4{ -1.0f, -1.0f, -1.0f, -1.0f };
        REQUIRE( ( lRoot["root.fourth"].Vec( { "root.foo.r", "root.fourth.g", "root.fourth.b", "root.fourth.a" }, lValue2 ) == lValue2 ) );
        REQUIRE( ( lRoot["root.fourth"].Vec( { "root.fourth.r", "root.foo.r", "root.fourth.b", "root.fourth.a" }, lValue2 ) == lValue2 ) );
        REQUIRE( ( lRoot["root.fourth"].Vec( { "root.fourth.r", "root.fourth.g", "root.foo.r", "root.fourth.a" }, lValue2 ) == lValue2 ) );
        REQUIRE( ( lRoot["root.fourth"].Vec( { "root.fourth.r", "root.fourth.g", "root.fourth.b", "root.foo.r" }, lValue2 ) == lValue2 ) );
        REQUIRE( ( lRoot["root.fourth"].Vec( { "root.foo.r", "root.foo.r", "root.fourth.b", "root.fourth.a" }, lValue2 ) == lValue2 ) );
        REQUIRE( ( lRoot["root.fourth"].Vec( { "root.foo.r", "root.fourth.g", "root.foo.r", "root.fourth.a" }, lValue2 ) == lValue2 ) );
        REQUIRE( ( lRoot["root.fourth"].Vec( { "root.foo.r", "root.fourth.g", "root.fourth.b", "root.foo.r" }, lValue2 ) == lValue2 ) );
        REQUIRE( ( lRoot["root.fourth"].Vec( { "root.foo.r", "root.foo.r", "root.foo.r", "root.fourth.a" }, lValue2 ) == lValue2 ) );
        REQUIRE( ( lRoot["root.fourth"].Vec( { "root.foo.r", "root.foo.r", "root.fourth.b", "root.foo.r" }, lValue2 ) == lValue2 ) );
        REQUIRE( ( lRoot["root.fourth"].Vec( { "root.foo.r", "root.foo.r", "root.foo.r", "root.foo.r" }, lValue2 ) == lValue2 ) );
    }


    SECTION( "Iterating over mappings" )
    {
        ConfigurationNode lRoot                = lTestReader.GetRoot();
        std::vector<std::string> lKeys              = {};
        std::vector<ConfigurationNode> lValues = {};

        lRoot["root"].ForEach<std::string>(
            [&]( std::string aKey, ConfigurationNode &aValue )
            {
                lKeys.push_back( aKey );
                lValues.push_back( aValue );
            } );
        REQUIRE( lKeys == std::vector<std::string>{ "first", "second", "third", "fourth" } );
        REQUIRE( lValues.size() == 4 );
        REQUIRE( lValues[1]["a"].As<uint32_t>( 0 ) == 5 );
    }

    SECTION( "Iterating over sequences" )
    {
        ConfigurationNode lRoot  = lTestReader.GetRoot();
        std::vector<uint32_t> lValues = {};

        lRoot["root.second.seq"].ForEach( [&]( ConfigurationNode &aValue ) { lValues.push_back( aValue.As<uint32_t>( 100000000 ) ); } );

        REQUIRE( lValues == std::vector<uint32_t>{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 } );
    }
}