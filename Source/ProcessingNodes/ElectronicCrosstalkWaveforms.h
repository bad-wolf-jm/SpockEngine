/// @file   WaveformGenerator.h
///
/// @brief  Waveform generator.
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2022 LeddarTech Inc. All rights reserved.

#pragma once

#include "Core/Math/Types.h"

#include "Core/EntityRegistry/Registry.h"

#include "TensorOps/NodeComponents.h"
#include "TensorOps/Scope.h"

namespace LTSE::SensorModel
{
    using OpNode = LTSE::TensorOps::OpNode;
    using Scope  = LTSE::TensorOps::Scope;

    /// @brief sResolveAbstractWaveforms
    ///
    /// Compute full waveforms given matched lists of return times and return intensities.
    /// The expected layout of the input tensors mReturnTimes and mReturnIntensities is as
    /// follows. Both should have the same number @f[ N @f] of layers corresponding to
    /// individual flashes. Each layer @f[ i @f] has dimension @f[ (M_i, K_i) @f], where
    /// @f[ M_i @f] corresponds to the number of photodetectors assigned to the flash,
    /// and @f[ K_i @f] represents the total number of detections collected during flash
    /// @f[ i @f]. Note that None of the values @f[ M_i @f] and @f[ K_i @f] are required
    /// to be equal, though in practice then will be.
    ///
    /// The output is a @ref MultiTensor with @f[ N @f] layer, where layer @f[ i @f] has dimension
    /// @f[ (M_i, L_i) @f]. In this case @f[ L_i @f] is given by the @f[ i @f]-th entry in
    /// the sampling length parameter. Once again the values @f[ L_i @f] are not required to
    /// be equal, though in practive then often are. In the special case where @f[ L_i = L @f].
    /// @f[ M_i = M @f] for every @f[ i @f], the output will be a @ref MultiTensor with @f[ N @f]
    /// layers, each of dimension @f[ (M, L) @f], which is fundamentally the sams as a
    /// @ref MultiTensor with just 1 layer, and which has dimension @f[ (N, K, L) @f], which itself
    /// is the same as a simple @f[ (N, K, L) @f] tensor.
    ///
    struct sResolveElectronicCrosstalkWaveforms
    {
        OpNode mReturnTimes{};          //!< Return times (ns)
        OpNode mReturnIntensities{};    //!< Return intensity (unit TBD)
        OpNode mSamplingLength{};       //!< Number of samples to generate per waveform
        OpNode mSamplingInterval{};     //!< Time interval between two samples (ns)
        OpNode mPulseTemplates{};       //!< Pulse templates to use for each layer of the above MultiTensors
        OpNode mAPDSizes{};             //!< Length of the APD matrix for each layer of the above MultiTensors
        OpNode mAPDTemplatePositions{}; //!< Position of the template vector
        OpNode mSummandCount{};         //!< Number of detections to use for each layer of the above MultiTensors (calculated automatically upon construction)

        sResolveElectronicCrosstalkWaveforms()                                               = default;
        sResolveElectronicCrosstalkWaveforms( const sResolveElectronicCrosstalkWaveforms & ) = default;
    };

    struct sResolveElectronicCrosstalkWaveformsController : public LTSE::TensorOps::sGraphOperationController
    {
        void Run();
    };

    /// @brief Create a waveform resolution node
    ///
    /// See for @ref sResolveAbstractWaveforms.
    ///
    /// @param aScope             Computation scope
    /// @param aReturnTimes       Return times (ns)
    /// @param aReturnIntensities Return intensity (unit TBD)
    /// @param aSamplingLength    Number of samples to generate per waveform
    /// @param aSamplingInterval  Time interval between two samples (ns)
    /// @param aPulseTemplates    Pulse templates to use for each layer of the above MultiTensors
    ///
    /// @return The newly created computation node
    ///
    OpNode ResolveElectronicCrosstalkWaveforms( Scope &aScope, OpNode aReturnTimes, OpNode aReturnIntensities, OpNode aSamplingLength, OpNode aSamplingInterval,
                                                OpNode aPulseTemplates );

} // namespace LTSE::SensorModel
