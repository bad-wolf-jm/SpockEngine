#pragma once

#include "Core/Memory.h"

#include "LidarSensorModel/EnvironmentSampler.h"
#include "LidarSensorModel/SensorDeviceBase.h"
#include "LidarSensorModel/SensorModelBase.h"

namespace LTSE::Editor
{
    using namespace LTSE::Core;
    using namespace LTSE::SensorModel;

    class SensorController
    {
      public:
        EnvironmentSampler::sCreateInfo EnvironmentSamplingParameter{};
        Entity CurrentTileLayout{};
        Entity SensorPawn{};
        bool RunSensorSimulation = false;

      public:
        Ref<SensorDeviceBase> m_ControlledSensor = nullptr;
        Ref<EnvironmentSampler> Sample();

        template <typename... Args> Ref<EnvironmentSampler> Sample( Args... aArgs )
        {
            return m_ControlledSensor->Sample( EnvironmentSamplingParameter, std::forward<Args>( aArgs )... );
        }

      public:
        SensorController( Ref<SensorDeviceBase> a_ControlledSensor );
        ~SensorController() = default;

      private:
    };
} // namespace LTSE::Editor