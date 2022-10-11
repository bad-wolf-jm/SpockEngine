/// @file   FPGAKernels.h
///
/// @brief  Kernel definitions for the FPGA.
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2022 LeddarTech Inc. All rights reserved.


#pragma once

#include <cstdint>

#include "Cuda/MemoryPool.h"
#include "Cuda/MultiTensor.h"

#include "TensorOps/ScalarTypes.h"

#include "FPGAConfiguration.h"

namespace LTSE::SensorModel
{

    using MultiTensor  = LTSE::Cuda::MultiTensor;
    using MemoryBuffer = LTSE::Cuda::MemoryBuffer;

    /// @brief Kernel launcher
    ///
    /// See for @ref sFPGAProcess.
    ///
    /// @param aOut             Output MultiTensor.
    /// @param aConfig          Configuration structure for the FPGA
    /// @param aWaveforms       Waveform buffer
    /// @param aSegmentCounts   Segment count for every layer.
    /// @param aWaveformLengths Length of waveforms for every layer
    /// @param aMaxSegmentCount Largest segment count
    ///
    /// @return The newly created computation node
    ///
    void FPGAProcessOp( MultiTensor &aOut, sFPGAConfiguration &aConfig, MultiTensor &aWaveforms, MemoryBuffer &aSegmentCounts, MemoryBuffer &aWaveformLengths, uint32_t aMaxSegmentCount );
}