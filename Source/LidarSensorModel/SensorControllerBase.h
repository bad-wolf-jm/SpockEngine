/// @file   SensorControllerBase.h
///
/// @brief  Basic interface for sensor controller
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2022 LeddarTech Inc. All rights reserved.

#pragma once

#include "Core/Math/Types.h"
#include "Core/Memory.h"

#include "EnvironmentSampler.h"
#include "SensorDeviceBase.h"

#include "TensorOps/Scope.h"

namespace LTSE::SensorModel
{
    using namespace LTSE::Core;
    using namespace LTSE::TensorOps;

    class SensorDeviceBase;

    /// @brief Sensor controller abstraction
    ///
    /// The sensor controller encapsulates the life cycle of a lidar sensor.
    ///
    class SensorControllerBase
    {
      public:
        EnvironmentSampler::sCreateInfo EnvironmentSamplingParameter{}; //!< Parameters to use for enviroment sampling

      public:
        /// @brief Default constructor
        SensorControllerBase() = default;

        /// @brief Default destructor
        ~SensorControllerBase() = default;

        /// @brief Power on the sensor
        ///
        /// The default implementation only sets a flag indicating that the device is powered on. It should be called in
        /// derived classes. In this method the controller instance should perform all steps necessary to initialize itself,
        /// and be ready to have the `Emit` and `Receive` methods called.
        ///
        virtual void PowerOn() = 0;

        /// @brief Shutdoen the sensor
        ///
        /// The default implementation sets the `mIsPoweredOn` flag to false, and should be called in derived classes.
        /// In this method the controller instance should return to a clean state, and be ready to be powered on again
        /// at a later time if required.
        ///
        virtual void Shutdown() = 0;

        /// @brief Connect a sensor to the controller instance
        ///
        /// @param aControlledSensor The sensor device to control.
        ///
        virtual void Connect( Ref<SensorDeviceBase> aControlledSensor ) = 0;

        /// @brief Disconnect the currently connected sensor from the controller instance
        virtual void Disconnect() = 0;

        /// @brief Fire lasers!
        ///
        /// The default implementation does nothing. When implementng this method in a derived class, one should produce
        /// a set of azimuths, elevations, laser intensities and timestamps to be sent out to a simulated 3D environment
        /// according to the specification laid out in @ref EnvironmentSamplingParameters.
        ///
        /// @param aTs Time elapsed since the last call to `Emit`
        ///
        virtual Ref<EnvironmentSampler> Emit( Timestep const &aTs ) = 0;

        /// @brief Receive data from the outside world
        ///
        /// The default implementation does nothing. This is the method to implement to funnel data retrieved from
        /// simulated 3D environments back into the sensor for further processing.
        ///
        /// @param aTs Time elapsed since the last call to `Receive`
        /// @param aScope Computation scope
        /// @param aFlashList Flash list used to produce the azimuth and elevation parameters
        /// @param aAzimuth OpNode representing the `x` coordinates of the points to process
        /// @param aElevation OpNode representing the `y` coordinates of the points to process
        /// @param aIntensity OpNode representing the intensity of the reflected laser pulse
        /// @param aDistance OpNode representing the distance of the object that reflected the laser pulse
        ///
        virtual void Receive( Timestep const &aTs, Scope &aScope, AcquisitionContext const &aFlashList, OpNode const &aAzimuth, OpNode const &aElevation, OpNode const &aIntensity,
                              OpNode const &aDistance );

        Ref<SensorDeviceBase> ControlledSensor();

      private:
        Ref<SensorDeviceBase> mControlledSensor = nullptr;
        bool mIsPoweredOn                       = false;
    };
} // namespace LTSE::SensorModel