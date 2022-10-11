/// @file   LTPostProcessingInterface.h
///
/// @brief  Definition file for the post processing interface.
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2021 LeddarTech Inc. All rights reserved.

#pragma once

#include "fmt/core.h"

#include "Core/Math/Types.h"

#include "LidarPostProcessor.h"

namespace OptixSensorLidar
{
    /// @brief Abstract base class for Leddartech sensor post processors
    ///
    /// This class defines virtual life cycle methods to be overridden in the actrual sensor implementation
    ///
    class LTPostProcessingInterface : public LidarPostProcessor
    {
      public:
        /// @brief Default constructor
        LTPostProcessingInterface() = default;

        /// @brief Default constructor
        ~LTPostProcessingInterface() = default;

        /// @brief Sensor initialization
        ///
        /// This function is called immediately after the constructor, before the post processing plugin is returned to
        /// the dSpace process. The implementation should perform the initial setup of the sensor model, and load all static
        /// data
        ///
        virtual void Initialize() = 0;

        /// @brief Sensor configuration
        ///
        /// This is called after the sensor is loaded into the dSpace environment simulator.
        ///
        virtual void Configure() = 0;

        /// @brief Sensor shutdown
        ///
        /// Called when the sensor gets destroyed
        ///
        virtual void Shutdown() = 0;

        /// @brief Sensor error
        virtual void OnError() = 0;

        /// @brief Sensor frame start
        ///
        /// This function will be called during the first iteration of the simulation loop. At this point the initialization
        /// of the sensor should be complete, and the implementation should generate initial batch of environment samples to
        /// be sent to the environment simulator
        ///.
        virtual void Begin() = 0;

        /// @brief Sensor loop
        ///
        /// Called every simulation frame with the time elapsed since the last frame. In this function the sensor
        /// implementation should fetch data from the dSapce environment to be further processed, and prepare a new
        /// batch of environment sample points to be sent to the dSpace environment simulator.
        ///
        virtual void Update( Timestep ts ) = 0;

      protected:
        // Driver functions, called by the dSpace environment
        PP_ErrorCode OnFirstFrame() override;
        PP_ErrorCode OnProcessFrame( const PostProcessingSensorProperty *aPropertyArray, uint32_t aArraySize ) override;
        PP_ErrorCode LoadJsonConfig( const char *aJsonUtf8 ) override;

        /// @brief Current simulation time in nanoseconds.
        int64_t GetCurrentSimulationTime();

      protected:
        int64_t mCurrentSimulationTime        = 0; //!< Current simulation time
        int64_t mTimeSinceLastSimulationFrame = 0; //!< Time elapsed since the last frame, in simulation time
    };

} // namespace OptixSensorLidar
