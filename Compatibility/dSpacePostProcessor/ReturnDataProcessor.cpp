/// @file   ReturnDataDestructer.cpp
///
/// @brief  Implementation file for converting dSpace ray tracing return data into LTSE nodes
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2021 LeddarTech Inc. All rights reserved.

#include "ReturnDataProcessor.h"
#include "ReturnDataProcessorKernels.h"

#include "Core/Cuda/MultiTensor.h"

#include "LidarSensorConfig.h"
#include "TensorOps/ScalarTypes.h"

namespace LTSE::dSpaceCompatibility
{
    using namespace LTSE::Core;
    using namespace LTSE::Cuda;
    using namespace LTSE::TensorOps;

    void sExtractReflectivityController::Run()
    {
        auto &lOutputTensor = Get<sMultiTensorComponent>().mValue;

        ExtractReflectivityOp( lOutputTensor, Get<sDestructLidarReturnStructures>().mReturnData );
    }

    void sExtractDistanceController::Run()
    {
        auto &lOutputTensor = Get<sMultiTensorComponent>().mValue;

        ExtractDistanceOp( lOutputTensor, Get<sDestructLidarReturnStructures>().mReturnData );
    }

    namespace
    {
        OpNode GenericDestructer( Scope &aScope, sTensorShape &aShape, MemoryBuffer &aReturnData )
        {
            auto lNewEntity = aScope.CreateNode();
            auto &lType     = lNewEntity.Add<sTypeComponent>();
            lType.mValue    = eScalarType::FLOAT32;

            auto &lAmplitudeOperandData       = lNewEntity.Add<sDestructLidarReturnStructures>();
            lAmplitudeOperandData.mReturnData = aReturnData;

            auto &lValue  = lNewEntity.Add<sMultiTensorComponent>();
            lValue.mValue = MultiTensor( aScope.mPool, aShape );

            return lNewEntity;
        }

    } // namespace

    OpNode RetrieveDistance( Scope &aScope, sTensorShape &aShape, MemoryBuffer &aReturnData )
    {
        auto lNewEntity = GenericDestructer( aScope, aShape, aReturnData );

        lNewEntity.Add<sGraphOperationComponent>().Bind<sExtractDistanceController>();

        return lNewEntity;
    }

    OpNode RetrieveIntensities( Scope &aScope, sTensorShape &aShape, MemoryBuffer &aReturnData )
    {
        auto lNewEntity = GenericDestructer( aScope, aShape, aReturnData );

        lNewEntity.Add<sGraphOperationComponent>().Bind<sExtractReflectivityController>();

        return lNewEntity;
    }

} // namespace LTSE::dSpaceCompatibility