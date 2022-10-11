/// @file   WaveformGeneratorKernels.h
///
/// @brief  Cuda kernel launcher.
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2022 LeddarTech Inc. All rights reserved.

#pragma once

#include "Cuda/MemoryPool.h"
#include "Cuda/MultiTensor.h"

#include "TensorOps/ScalarTypes.h"

namespace LTSE::SensorModel::Kernels
{

    using MultiTensor  = LTSE::Cuda::MultiTensor;
    using MemoryBuffer = LTSE::Cuda::MemoryBuffer;

    /// @brief Kernel launcher
    ///
    /// See for @ref sResolveAbstractWaveforms.
    ///
    /// @param aOut                  Output MultiTensor.
    /// @param aReturnIntensities    Return times (ns)
    /// @param aReturnTimes          Return intensity (unit TBD)
    /// @param aSummandCount         Number of summands to process.
    /// @param mAPDSizes             Size of the photodetector matrix
    /// @param aAPDTemplatePositions Position of the template list to use for the corresponding photodetector
    /// @param aSamplingLength       Length of the waveform to generate
    /// @param aSamplingInterval     Time interval between consecutive samples
    /// @param aPulseTemplates       Pulse template to use for each layer of the above MultiTensors
    /// @param aMaxAPDSize           Maximum photodetector matrix size
    /// @param aMaxWaveformLength    Maximum length of the waveforms to generator
    ///
    ///
    void ResolveElectronicCrosstalkWaveformsOp( MultiTensor &aOut, MultiTensor &aReturnIntensities, MultiTensor &aReturnTimes, MemoryBuffer &aSummandCount, MemoryBuffer &mAPDSizes,
                                                MemoryBuffer &aAPDTemplatePositions, MemoryBuffer &aSamplingLength, MemoryBuffer &aSamplingInterval, MemoryBuffer &aPulseTemplates,
                                                int32_t aMaxAPDSize, uint32_t aMaxWaveformLength );

} // namespace LTSE::SensorModel::Kernels