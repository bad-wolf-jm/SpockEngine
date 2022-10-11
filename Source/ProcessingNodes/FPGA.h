/// @file   FPGA.h
///
/// @brief  FPGA processing.
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2022 LeddarTech Inc. All rights reserved.

#pragma once

#include "Core/Math/Types.h"

#include "Core/EntityRegistry/Registry.h"

#include "FPGAConfiguration.h"
#include "TensorOps/NodeComponents.h"
#include "TensorOps/Scope.h"
#include "WaveformType.h"


namespace LTSE::SensorModel
{
    using OpNode = LTSE::TensorOps::OpNode;
    using Scope  = LTSE::TensorOps::Scope;

    /// @brief sFPGAProcess
    ///
    struct sFPGAProcess
    {
        OpNode mWaveforms{};      //!< OpNode holding the waveform buffer
        OpNode mSegmentCount{};   //!< OpNode containing the number of segments (APD cells). These are automatically generated.
        OpNode mWaveformLength{}; //!< OpNode containing the length of waveforms. These are automatically generated.

        sFPGAProcess()                       = default;
        sFPGAProcess( const sFPGAProcess & ) = default;
    };

    struct sFPGAProcessController : public LTSE::TensorOps::sGraphOperationController
    {
        void Run();
    };

    /// @brief Create a FPGA processingnode
    ///
    /// See for @ref sFPGAProcess.
    ///
    /// @param aScope     Computation scope
    /// @param aConfig    Configuration
    /// @param aWaveforms Waveform buffer
    ///
    /// @return The newly created computation node
    ///
    OpNode CreateFPGAProcessNode( Scope &aScope, sFPGAConfiguration const &aConfig, OpNode const &aWaveforms );

} // namespace LTSE::SensorModel