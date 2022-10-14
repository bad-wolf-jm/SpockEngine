/// @file   ReturnDataDestructer.cpp
///
/// @brief  Implementation file for converting dSpace ray tracing return data into LTSE nodes
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2021 LeddarTech Inc. All rights reserved.

#include "HitRecordProcessor.h"
#include "HitRecordProcessorKernels.h"

#include "Core/Cuda/MultiTensor.h"

#include "TensorOps/ScalarTypes.h"

namespace LTSE::SensorModel
{
    using namespace LTSE::Core;
    using namespace LTSE::Cuda;
    using namespace LTSE::TensorOps;

    void sExtractReflectivityController::Run()
    {
        auto &lOutputTensor = Get<sMultiTensorComponent>().mValue;

        ExtractReflectivityOp( lOutputTensor, Get<sDestructHitRecordStructures>().mReturnData );
    }

    void sExtractDistanceController::Run()
    {
        auto &lOutputTensor = Get<sMultiTensorComponent>().mValue;

        ExtractDistanceOp( lOutputTensor, Get<sDestructHitRecordStructures>().mReturnData );
    }

    namespace
    {
        OpNode GenericDestructer( Scope &aScope, MultiTensor &aReturnData )
        {
            auto lNewEntity       = aScope.CreateNode();
            auto &lAmplitudeType  = lNewEntity.Add<sTypeComponent>();
            lAmplitudeType.mValue = eScalarType::FLOAT32;

            auto &lAmplitudeOperandData       = lNewEntity.Add<sDestructHitRecordStructures>();
            lAmplitudeOperandData.mReturnData = aReturnData;

            auto lShape = aReturnData.Shape().mShape;
            auto &lAmplitudeValue  = lNewEntity.Add<sMultiTensorComponent>();
            lAmplitudeValue.mValue = MultiTensor( aScope.mPool, sTensorShape(lShape, SizeOf(lAmplitudeType.mValue)) );

            return lNewEntity;
        }

    } // namespace

    OpNode RetrieveDistance( Scope &aScope, MultiTensor &aReturnData )
    {
        auto lNewEntity = GenericDestructer( aScope, aReturnData );

        lNewEntity.Add<sGraphOperationComponent>().Bind<sExtractDistanceController>();

        return lNewEntity;
    }

    OpNode RetrieveIntensities( Scope &aScope, MultiTensor &aReturnData )
    {
        auto lNewEntity = GenericDestructer( aScope, aReturnData );

        lNewEntity.Add<sGraphOperationComponent>().Bind<sExtractReflectivityController>();

        return lNewEntity;
    }

} // namespace LTSE::SensorModel