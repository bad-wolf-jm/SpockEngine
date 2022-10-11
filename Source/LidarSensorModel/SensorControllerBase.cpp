/// @file   SensorControllerBase.cpp
///
/// @brief  Default implementation for sensor controller functions
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2022 LeddarTech Inc. All rights reserved.

#include "SensorControllerBase.h"

namespace LTSE::SensorModel
{
    void SensorControllerBase::PowerOn() { mIsPoweredOn = true; }

    void SensorControllerBase::Shutdown() { mIsPoweredOn = false; }

    void SensorControllerBase::Connect( Ref<SensorDeviceBase> aControlledSensor ) { mControlledSensor = aControlledSensor; }

    void SensorControllerBase::Disconnect() { mControlledSensor = nullptr; }

    Ref<EnvironmentSampler> SensorControllerBase::Emit( Timestep const &aTs ) { return nullptr; }

    void SensorControllerBase::Receive( Timestep const &aTs, Scope &aScope, AcquisitionContext const &aFlashList, OpNode const &aAzimuth, OpNode const &aElevation,
                                        OpNode const &aIntensity, OpNode const &aDistance )
    {
        if( !mControlledSensor )
            return;

        mControlledSensor->Process( aTs, aScope, aFlashList, aAzimuth, aElevation, aIntensity, aDistance );
    }

    Ref<SensorDeviceBase> SensorControllerBase::ControlledSensor() { return mControlledSensor; }

} // namespace LTSE::SensorModel