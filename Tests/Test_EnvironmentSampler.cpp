#include <catch2/catch_test_macros.hpp>

#include <algorithm>

#include "Core/Memory.h"

#include "Core/Cuda/MultiTensor.h"

#include "LidarSensorModel/AcquisitionContext/AcquisitionContext.h"
#include "LidarSensorModel/Components.h"
#include "LidarSensorModel/EnvironmentSampler.h"
#include "LidarSensorModel/SensorModelBase.h"


#include "TensorOps/NodeComponents.h"
#include "TensorOps/Scope.h"

#include "TestUtils.h"

using namespace math;
using namespace TestUtils;
using namespace LTSE;
using namespace LTSE::Core;
using namespace LTSE::TensorOps;
using namespace LTSE::SensorModel;

class TestEnvironmentSampler : public EnvironmentSampler
{
  public:
    TestEnvironmentSampler( const sCreateInfo &aSpec, Ref<Scope> aScope, const AcquisitionContext &aFlashList )
        : EnvironmentSampler( aSpec, aScope, aFlashList ){};

    TestEnvironmentSampler( const sCreateInfo &aSpec, uint32_t aPoolSize, const AcquisitionContext &aFlashList )
        : EnvironmentSampler( aSpec, aPoolSize, aFlashList ){};

    OpNode &Check( std::string aName ) { return ( *mScope )[aName]; }
    Cuda::MultiTensor &Retrieve( std::string aName ) { return ( *mScope )[aName].Get<sMultiTensorComponent>().mValue; }

    void CreateGraph() { EnvironmentSampler::CreateGraph(); }
};

TEST_CASE( "Environment sampling", "[CORE_SENSOR_MODEL]" )
{
    SensorModelBase lSensorModelBase{};
    uint32_t lPoolSize = 5 * 1024 * 1024;

    SECTION( "Create environment sampler" )
    {
        Entity lTile1  = lSensorModelBase.CreateTile( "1", vec2{ 1.0f, 0.0f } );
        Entity lFlash0 = lSensorModelBase.CreateFlash( lTile1, "0", vec2{ -1.0f, 0.0f }, vec2{ 2.0f, 3.0f } );
        Entity lFlash1 = lSensorModelBase.CreateFlash( lTile1, "1", vec2{ 1.0f, 0.0f }, vec2{ 4.0f, 5.0f } );
        Entity lFlash2 = lSensorModelBase.CreateFlash( lTile1, "2", vec2{ 2.0f, 0.0f }, vec2{ 1.0f, 4.0f } );

        AcquisitionSpecification lAcqCreateInfo{};
        AcquisitionContext lLaserFlashList( lAcqCreateInfo, lTile1, math::vec2{ 2.0f, 2.0f }, 0.0f );

        Ref<Scope> lScope = New<Scope>( lPoolSize );

        EnvironmentSampler::sCreateInfo lCreateInfo;
        lCreateInfo.mUseRegularMultiSampling = false;
        lCreateInfo.mSamplingResolution      = { 0.2f, 0.2f };
        lCreateInfo.mMultiSamplingFactor     = 3;
        TestEnvironmentSampler lSampler( lCreateInfo, lScope, lLaserFlashList );
        lSampler.CreateGraph();
        REQUIRE( true );
    }

    SECTION( "Create environment sampler" )
    {
        Entity lTile1 = lSensorModelBase.CreateTile( "1", vec2{ 1.0f, 0.0f } );

        vec2 lExtent0  = vec2{ 2.0f, 3.0f };
        vec2 lSize0    = lExtent0 * 2.0f;
        Entity lFlash0 = lSensorModelBase.CreateFlash( lTile1, "0", vec2{ -1.0f, 0.0f }, vec2{ 2.0f, 3.0f } );

        vec2 lExtent1  = vec2{ 4.0f, 5.0f };
        vec2 lSize1    = lExtent1 * 2.0f;
        Entity lFlash1 = lSensorModelBase.CreateFlash( lTile1, "1", vec2{ 1.0f, 0.0f }, vec2{ 4.0f, 5.0f } );

        vec2 lExtent2  = vec2{ 1.0f, 4.0f };
        vec2 lSize2    = lExtent2 * 2.0f;
        Entity lFlash2 = lSensorModelBase.CreateFlash( lTile1, "2", vec2{ 2.0f, 0.0f }, vec2{ 1.0f, 4.0f } );

        AcquisitionSpecification lAcqCreateInfo{};
        AcquisitionContext lLaserFlashList( lAcqCreateInfo, lTile1, math::vec2{ 2.0f, 2.0f }, 0.0f );

        EnvironmentSampler::sCreateInfo lCreateInfo;
        lCreateInfo.mUseRegularMultiSampling = false;
        lCreateInfo.mSamplingResolution      = { 0.18237654f, 0.192083764f };
        lCreateInfo.mMultiSamplingFactor     = 3;
        TestEnvironmentSampler lSampler( lCreateInfo, lPoolSize, lLaserFlashList );
        lSampler.CreateGraph();

        constexpr uint32_t lExpectedTensorLayers = 3;
        constexpr uint32_t lExpectedTensorRank   = 1;

        std::vector<uint32_t> lDim1{ static_cast<uint32_t>( std::ceil( lSize0.y / lCreateInfo.mSamplingResolution.y ) ) *
                                     static_cast<uint32_t>( std::ceil( lSize0.x / lCreateInfo.mSamplingResolution.x ) ) * lCreateInfo.mMultiSamplingFactor };
        std::vector<uint32_t> lDim2{ static_cast<uint32_t>( std::ceil( lSize1.y / lCreateInfo.mSamplingResolution.y ) ) *
                                     static_cast<uint32_t>( std::ceil( lSize1.x / lCreateInfo.mSamplingResolution.x ) ) * lCreateInfo.mMultiSamplingFactor };
        std::vector<uint32_t> lDim3{ static_cast<uint32_t>( std::ceil( lSize2.y / lCreateInfo.mSamplingResolution.y ) ) *
                                     static_cast<uint32_t>( std::ceil( lSize2.x / lCreateInfo.mSamplingResolution.x ) ) * lCreateInfo.mMultiSamplingFactor };

        sTensorShape lExpectedTensorShape( { lDim1, lDim2, lDim3 }, sizeof( float ) );

        REQUIRE( ( lSampler.Check( "Azimuth" ) ) );
        REQUIRE( ( lSampler.Check( "Elevation" ) ) );
        REQUIRE( ( lSampler.Check( "Intensity" ) ) );
        REQUIRE( ( lSampler.Check( "FlashId" ) ) );
        REQUIRE( ( lSampler.Check( "Timestamp" ) ) );

        REQUIRE( lSampler.Retrieve( "Azimuth" ).Shape().CountLayers() == lExpectedTensorLayers );
        REQUIRE( lSampler.Retrieve( "Azimuth" ).Shape().mRank == lExpectedTensorRank );
        REQUIRE( lSampler.Retrieve( "Azimuth" ).Shape().mShape == lExpectedTensorShape.mShape );

        REQUIRE( lSampler.Retrieve( "Elevation" ).Shape().CountLayers() == lExpectedTensorLayers );
        REQUIRE( lSampler.Retrieve( "Elevation" ).Shape().mRank == lExpectedTensorRank );
        REQUIRE( lSampler.Retrieve( "Elevation" ).Shape().mShape == lSampler.Retrieve( "Azimuth" ).Shape().mShape );

        REQUIRE( lSampler.Retrieve( "Intensity" ).Shape().CountLayers() == lExpectedTensorLayers );
        REQUIRE( lSampler.Retrieve( "Intensity" ).Shape().mRank == lExpectedTensorRank );
        REQUIRE( lSampler.Retrieve( "Intensity" ).Shape().mShape == lSampler.Retrieve( "Azimuth" ).Shape().mShape );

        REQUIRE( lSampler.Retrieve( "FlashId" ).Shape().CountLayers() == lExpectedTensorLayers );
        REQUIRE( lSampler.Retrieve( "FlashId" ).Shape().mRank == lExpectedTensorRank );
        REQUIRE( lSampler.Retrieve( "FlashId" ).Shape().mShape == lSampler.Retrieve( "Azimuth" ).Shape().mShape );

        REQUIRE( lSampler.Retrieve( "Timestamp" ).Shape().CountLayers() == lExpectedTensorLayers );
        REQUIRE( lSampler.Retrieve( "Timestamp" ).Shape().mRank == lExpectedTensorRank );
        REQUIRE( lSampler.Retrieve( "Timestamp" ).Shape().mShape == lSampler.Retrieve( "Azimuth" ).Shape().mShape );
    }

    SECTION( "Sample single tile" )
    {
        Entity lTile1  = lSensorModelBase.CreateTile( "1", vec2{ 1.0f, 0.0f } );
        Entity lFlash0 = lSensorModelBase.CreateFlash( lTile1, "0", vec2{ -1.0f, 0.0f }, vec2{ 2.0f, 3.0f } );
        Entity lFlash1 = lSensorModelBase.CreateFlash( lTile1, "1", vec2{ 1.0f, 0.0f }, vec2{ 4.0f, 5.0f } );
        Entity lFlash2 = lSensorModelBase.CreateFlash( lTile1, "2", vec2{ 2.0f, 0.0f }, vec2{ 1.0f, 4.0f } );

        AcquisitionSpecification lAcqCreateInfo{};
        AcquisitionContext lLaserFlashList( lAcqCreateInfo, lTile1, math::vec2{ 2.0f, 2.0f }, 0.0f );

        Ref<Scope> lScope = New<Scope>( lPoolSize );

        EnvironmentSampler::sCreateInfo lCreateInfo;
        lCreateInfo.mUseRegularMultiSampling = false;
        lCreateInfo.mSamplingResolution      = { 0.19238746f, 0.41283764f };
        lCreateInfo.mMultiSamplingFactor     = 3;
        TestEnvironmentSampler lSampler( lCreateInfo, lScope, lLaserFlashList );
        lSampler.Run();

        {
            std::vector<float> lAzimuths0 = lSampler.Retrieve( "Azimuth" ).FetchFlattened<float>();
            std::vector<float> lMS0( lAzimuths0.size() / 3 );
            std::vector<float> lMS1( lAzimuths0.size() / 3 );
            std::vector<float> lMS2( lAzimuths0.size() / 3 );

            for( uint32_t i = 0; i < lMS0.size(); i++ )
            {
                lMS0[i] = lAzimuths0[3 * i + 0];
                lMS1[i] = lAzimuths0[3 * i + 1];
                lMS2[i] = lAzimuths0[3 * i + 2];
            }

            REQUIRE( ArrayMax( Absol( lMS1 - lMS0 ) ) <= lCreateInfo.mSamplingResolution.x );
            REQUIRE( ArrayMax( Absol( lMS2 - lMS1 ) ) <= lCreateInfo.mSamplingResolution.x );
            REQUIRE( ArrayMax( Absol( lMS2 - lMS0 ) ) <= lCreateInfo.mSamplingResolution.x );
        }

        {
            std::vector<float> lElevations0 = lSampler.Retrieve( "Elevation" ).FetchFlattened<float>();
            std::vector<float> lMS0( lElevations0.size() / 3 );
            std::vector<float> lMS1( lElevations0.size() / 3 );
            std::vector<float> lMS2( lElevations0.size() / 3 );

            for( uint32_t i = 0; i < lMS0.size(); i++ )
            {
                lMS0[i] = lElevations0[3 * i + 0];
                lMS1[i] = lElevations0[3 * i + 1];
                lMS2[i] = lElevations0[3 * i + 2];
            }

            REQUIRE( ArrayMax( Absol( lMS1 - lMS0 ) ) <= lCreateInfo.mSamplingResolution.y );
            REQUIRE( ArrayMax( Absol( lMS2 - lMS1 ) ) <= lCreateInfo.mSamplingResolution.y );
            REQUIRE( ArrayMax( Absol( lMS2 - lMS0 ) ) <= lCreateInfo.mSamplingResolution.y );
        }
    }
}