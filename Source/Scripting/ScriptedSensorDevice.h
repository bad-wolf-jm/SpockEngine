#pragma once

#include "LidarSensorModel/SensorDeviceBase.h"
#include "LidarSensorModel/SensorModelBase.h"

#include "ScriptingEngine.h"

namespace LTSE::Core
{
    class LuaSensorDevice : public SensorDeviceBase
    {
      public:
        LuaSensorDevice( uint32_t aMemorySize, fs::path const &aImplementation );

        ~LuaSensorDevice() = default;

        Ref<ScriptingEngine> mScripting = nullptr;
        ScriptEnvironment mEnvironment;

        sol::table Sample( std::vector<Entity> const &aTile, std::vector<math::vec2> const &aPosition, std::vector<float> const &aTimestamp );

        OpNode Process( Timestep const &aTs, std::vector<Entity> const &aTile, std::vector<math::vec2> const &aPosition, std::vector<float> const &aTimestamp,
                                OpNode const &aAzimuth, OpNode const &aElevation, OpNode const &aIntensity, OpNode const &aDistance );
    };
} // namespace LTSE::Core