/// @file   AcquisitionSpecification.h
///
/// @brief AcquisitionSpecification definition
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2022 LeddarTech Inc. All rights reserved.

#pragma once

#include "Core/Math/Types.h"
#include "Core/Memory.h"
#include "Cuda/Texture2D.h"

namespace LTSE::SensorModel
{
    /// @brief Specification structure for a single acquisition
    ///
    /// Collects all data that is common to all items in a given acquisition
    ///
    struct AcquisitionSpecification
    {
        uint32_t mBasePoints   = 1; //!< Base length of a raw waveform
        uint32_t mOversampling = 1; //!< Multiplier for the base length. This value also effectively multiplies the sampling frequency of the raw waveform
        uint32_t mAccumulation = 1; //!< Accumulation factor. This value influences the amount of noise that is introduced during sampling

        float mAPDBias = 0.0f; //!< Gain for the photodetector
        float mTIAGain = 1.0f; //!< GAin factor for the TIA

        float mTemperature = 0.0f; //!< Current system temperature

        struct
        {
            float mDC = 0.0f; //!< Ambiant noise, direct current component
            float mAC = 0.0f; //!< Ambiand noise, alternating current component
        } mAmbientNoise;

        /// @brief Default constructor
        AcquisitionSpecification() = default;

        /// @brief Default destructor
        ~AcquisitionSpecification() = default;

        /// @brief Copy destructor
        AcquisitionSpecification( AcquisitionSpecification const & ) = default;
    };

} // namespace LTSE::SensorModel