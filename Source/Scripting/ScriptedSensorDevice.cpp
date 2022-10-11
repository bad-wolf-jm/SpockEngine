#include "ScriptedSensorDevice.h"

namespace LTSE::Core
{
    LuaSensorDevice::LuaSensorDevice( uint32_t aMemorySize, fs::path const &aImplementation )
        : SensorDeviceBase( aMemorySize )
    {
        mScripting        = New<ScriptingEngine>();
        mSensorDefinition = New<SensorModelBase>();

        auto lDeviceType                 = mScripting->RegisterType<LuaSensorDevice>( "LuaSensorDevice" );
        lDeviceType["sensor_model"]      = sol::readonly( &LuaSensorDevice::mSensorDefinition );
        lDeviceType["computation_scope"] = sol::readonly( &LuaSensorDevice::mComputationScope );

        mEnvironment = mScripting->LoadFile( aImplementation.string() );

        mEnvironment["configure"]( *this );
    }

    sol::table LuaSensorDevice::Sample( std::vector<Entity> const &aTile, std::vector<math::vec2> const &aPosition, std::vector<float> const &aTimestamp )
    {
        auto x = mEnvironment["sample"]( *this, aTile, aPosition, aTimestamp );
        return x.get<sol::table>();
    }

    OpNode LuaSensorDevice::Process( Timestep const &aTs, std::vector<Entity> const &aTile, std::vector<math::vec2> const &aPosition, std::vector<float> const &aTimestamp,
                                   OpNode const &aAzimuth, OpNode const &aElevation, OpNode const &aIntensity, OpNode const &aDistance )
    {
        auto x = mEnvironment["process_environment_returns"]( *this, aTs, aTile, aPosition, aTimestamp, aAzimuth, aElevation, aIntensity, aDistance );

        return x.get<OpNode>();
    }

} // namespace LTSE::Core