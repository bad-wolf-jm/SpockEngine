/// @file   WaveformGenerator.cpp
///
/// @brief  Implementation of the waveform generator.
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2022 LeddarTech Inc. All rights reserved.

#include <numeric>

#include "ElectronicCrosstalkWaveforms.h"
#include "ElectronicCrosstalkWaveformKernels.h"

#include "Cuda/MultiTensor.h"
#include "Cuda/Texture2D.h"

#include "TensorOps/ScalarTypes.h"

namespace LTSE::SensorModel
{

    using namespace LTSE::Core;
    using namespace LTSE::TensorOps;

    void sResolveElectronicCrosstalkWaveformsController::Run()
    {
        auto &lValue       = GetControlledEntity().Get<sMultiTensorComponent>().mValue;
        auto &lOperandData = GetControlledEntity().Get<sResolveElectronicCrosstalkWaveforms>();

        uint32_t lMaxAPDSize = 0;
        for( auto lSize : lOperandData.mAPDSizes.Get<sVectorComponent<uint32_t>>().mValue )
            lMaxAPDSize = std::max( lMaxAPDSize, lSize );

        uint32_t lMaxWaveformLength = 0;
        for( auto lParam : lOperandData.mSamplingLength.Get<sVectorComponent<uint32_t>>().mValue )
            lMaxWaveformLength = std::max( lMaxWaveformLength, lParam );

        auto &lReturnIntensities    = lOperandData.mReturnIntensities.Get<sMultiTensorComponent>().mValue;
        auto &lReturnTimes          = lOperandData.mReturnTimes.Get<sMultiTensorComponent>().mValue;
        auto &lSummandCount         = lOperandData.mSummandCount.Get<sVectorComponent<uint32_t>>().mData;
        auto &lAPDSizes             = lOperandData.mAPDSizes.Get<sVectorComponent<uint32_t>>().mData;
        auto &lAPDTemplatePositions = lOperandData.mAPDTemplatePositions.Get<sVectorComponent<uint32_t>>().mData;
        auto &lSamplingLength       = lOperandData.mSamplingLength.Get<sVectorComponent<uint32_t>>().mData;
        auto &lSamplingInterval     = lOperandData.mSamplingInterval.Get<sVectorComponent<float>>().mData;
        auto &lPulseTemplates       = lOperandData.mPulseTemplates.Get<sVectorComponent<Cuda::TextureSampler2D::DeviceData>>().mData;

        Kernels::ResolveElectronicCrosstalkWaveformsOp( lValue, lReturnIntensities, lReturnTimes, lSummandCount, lAPDSizes, lAPDTemplatePositions, lSamplingLength,
                                                        lSamplingInterval, lPulseTemplates, lMaxAPDSize, lMaxWaveformLength );
    }

    OpNode ResolveElectronicCrosstalkWaveforms( LTSE::TensorOps::Scope &aScope, OpNode aReturnTimes, OpNode aReturnIntensities, OpNode aSamplingLength, OpNode aSamplingInterval,
                                                OpNode aPulseTemplates )
    {
        auto lNewEntity = aScope.CreateNode();

        auto &lType  = lNewEntity.Add<sTypeComponent>();
        lType.mValue = eScalarType::FLOAT32;

        auto &lOperandData              = lNewEntity.Add<sResolveElectronicCrosstalkWaveforms>();
        lOperandData.mReturnTimes       = aReturnTimes;
        lOperandData.mReturnIntensities = aReturnIntensities;
        lOperandData.mSamplingLength    = aSamplingLength;
        lOperandData.mSamplingInterval  = aSamplingInterval;
        lOperandData.mPulseTemplates    = aPulseTemplates;

        auto &lTimesTensorShape = aReturnTimes.Get<sMultiTensorComponent>().mValue.Shape();
        uint32_t lLayerCount    = lTimesTensorShape.CountLayers();

        std::vector<uint32_t> lSummandCount( lLayerCount );

        std::vector<uint32_t> lAPDSizes = lTimesTensorShape.GetDimension( 0 );
        std::vector<uint32_t> lAPDTemplatePosition( lAPDSizes.size() );
        std::exclusive_scan( lAPDSizes.begin(), lAPDSizes.end(), lAPDTemplatePosition.begin(), 0u );

        for( uint32_t i = 0; i < lLayerCount; i++ )
            lSummandCount[i] = std::accumulate( lTimesTensorShape.mShape[i].begin() + 1, lTimesTensorShape.mShape[i].end(), 1, std::multiplies<uint32_t>() );

        lOperandData.mAPDSizes             = VectorValue( aScope, lAPDSizes );
        lOperandData.mAPDTemplatePositions = VectorValue( aScope, lAPDTemplatePosition );
        lOperandData.mSummandCount         = VectorValue( aScope, lSummandCount );

        auto &lOperands     = lNewEntity.Add<sOperandComponent>();
        lOperands.mOperands = { aReturnTimes,
                                aReturnIntensities,
                                aSamplingLength,
                                aSamplingInterval,
                                aPulseTemplates,
                                lOperandData.mAPDSizes,
                                lOperandData.mAPDTemplatePositions,
                                lOperandData.mSummandCount };

        auto &lValue = lNewEntity.Add<sMultiTensorComponent>();
        sTensorShape lTargetShape( lAPDSizes, SizeOf( lType.mValue ) );
        lTargetShape.InsertDimension( -1, aSamplingLength.Get<sVectorComponent<uint32_t>>().mValue );
        lValue.mValue = Cuda::MultiTensor( aScope.mPool, lTargetShape );

        lNewEntity.Add<sGraphOperationComponent>().Bind<sResolveElectronicCrosstalkWaveformsController>();
        return lNewEntity;
    }

} // namespace LTSE::SensorModel