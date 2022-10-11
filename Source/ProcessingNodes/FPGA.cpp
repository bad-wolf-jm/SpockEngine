/// @file   FPGA.cpp
///
/// @brief  FPGA processing.
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2022 LeddarTech Inc. All rights reserved.

#include "FPGA.h"
#include "FPGAKernels.h"

#include "Cuda/MultiTensor.h"

#include "TensorOps/NodeComponents.h"

using namespace LTSE::Cuda;

namespace LTSE::SensorModel
{

    using namespace LTSE::Core;
    using namespace LTSE::TensorOps;

    void sFPGAProcessController::Run()
    {
        auto &lValue         = GetControlledEntity().Get<sMultiTensorComponent>().mValue;
        auto &lOperandData   = GetControlledEntity().Get<sFPGAProcess>();
        auto &lConfiguration = GetControlledEntity().Get<sFPGAConfiguration>();

        uint32_t lMaxAPDSize = 0;
        for( auto lSize : lOperandData.mSegmentCount.Get<sVectorComponent<uint32_t>>().mValue )
            lMaxAPDSize = std::max( lMaxAPDSize, lSize );

        auto &lWaveforms      = lOperandData.mWaveforms.Get<sMultiTensorComponent>().mValue;
        auto &lSegmentCounts  = lOperandData.mSegmentCount.Get<sVectorComponent<uint32_t>>().mData;
        auto &lWaveformLength = lOperandData.mWaveformLength.Get<sVectorComponent<uint32_t>>().mData;

        FPGAProcessOp( lValue, lConfiguration, lWaveforms, lSegmentCounts, lWaveformLength, lMaxAPDSize );
    }

    OpNode CreateFPGAProcessNode( Scope &aScope, sFPGAConfiguration const &aConfig, OpNode const &aWaveforms )
    {
        uint32_t lLayerCount = aWaveforms.Get<sMultiTensorComponent>().mValue.Shape().CountLayers();

        std::vector<std::vector<uint32_t>> lWaveformShape = aWaveforms.Get<sMultiTensorComponent>().mValue.Shape().mShape;
        std::vector<uint32_t> lWaveformLengths( lLayerCount );
        std::vector<uint32_t> lSegmentCounts( lLayerCount );
        for( uint32_t i = 0; i < lLayerCount; i++ )
        {
            lSegmentCounts[i]   = lWaveformShape[i][0];
            lWaveformLengths[i] = lWaveformShape[i][1];
        }

        auto lNewEntity = aScope.CreateNode();
        lNewEntity.Add<sFPGAConfiguration>( aConfig );

        auto &lOperandData           = lNewEntity.Add<sFPGAProcess>();
        lOperandData.mSegmentCount   = VectorValue( aScope, lSegmentCounts );
        lOperandData.mWaveformLength = VectorValue( aScope, lWaveformLengths );

        // Calculate the appropriate dimension for the output tensor
        std::vector<std::vector<uint32_t>> lTargetShape( lLayerCount );
        for( uint32_t i = 0; i < lLayerCount; i++ )
            lTargetShape[i] = { lWaveformShape[0][0], 1 };

        lOperandData.mWaveforms = aWaveforms;

        auto &lOperands     = lNewEntity.Add<sOperandComponent>();
        lOperands.mOperands = { aWaveforms, lOperandData.mSegmentCount, lOperandData.mWaveformLength };

        auto &lValue  = lNewEntity.Add<sMultiTensorComponent>();
        lValue.mValue = Cuda::MultiTensor( aScope.mPool, sTensorShape( lTargetShape, sizeof( sWaveformPacket ) ) );

        lNewEntity.Add<sGraphOperationComponent>().Bind<sFPGAProcessController>();
        return lNewEntity;
    }
} // namespace LTSE::SensorModel