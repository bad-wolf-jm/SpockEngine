/// @file   FPGAConfiguration.h
///
/// @brief  Waveform generator.
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2022 LeddarTech Inc. All rights reserved.

#pragma once
namespace LTSE::SensorModel
{

    /// @brief sFPGAConfiguration
    ///
    /// Configuration structure for the FPGA
    ///
    struct sFPGAConfiguration
    {
        uint32_t mBaseLevelSampleCount = 8; //!< Number of waveform samples to consider for the computation of baseline and noise
    };

} // namespace LTSE::SensorModel