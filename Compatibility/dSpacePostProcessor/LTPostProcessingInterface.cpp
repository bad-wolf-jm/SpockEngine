/// @file   LTPostProcessingInterface.cpp
///
/// @brief  Implementation file for the post processing interface
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2021 LeddarTech Inc. All rights reserved.

#include "LTPostProcessingInterface.h"
#include "LidarSensorConfig.h"

namespace OptixSensorLidar
{
    int64_t LTPostProcessingInterface::GetCurrentSimulationTime() { return static_cast<int64_t>( m_SimulationTime * 1000000 ); }

    PP_ErrorCode LTPostProcessingInterface::LoadJsonConfig( const char *aJsonUtf8 )
    {
        Configure();
        return DefaultErrorCodes::PP_SUCCESS;
    }

    PP_ErrorCode LTPostProcessingInterface::OnFirstFrame()
    {
        mCurrentSimulationTime = GetCurrentSimulationTime();

        ClearPattern();
        Begin();

        return DefaultErrorCodes::PP_SUCCESS;
    }

    PP_ErrorCode LTPostProcessingInterface::OnProcessFrame( [[maybe_unused]] const PostProcessingSensorProperty *aPropertyArray, [[maybe_unused]] uint32_t aArraySize )
    {
        int64_t lNow                  = GetCurrentSimulationTime();
        mTimeSinceLastSimulationFrame = lNow - mCurrentSimulationTime;
        mCurrentSimulationTime        = lNow;
        LOG_PP( LL_ERROR ) << "======= OnProcessFrame =======" << FLUSHSL;
        LOG_PP( LL_ERROR ) << mTimeSinceLastSimulationFrame << FLUSHSL;

        try
        {
            Update( mTimeSinceLastSimulationFrame );

            LOG_PP( LL_ERROR ) << "======= OnProcessFrameEnd =======" << FLUSHSL;

            return DefaultErrorCodes::PP_SUCCESS;
        }
        catch( const std::exception &e )
        {
            LOG_PP( LL_ERROR ) << "Failed to process frame: " << e.what() << FLUSHSL;
            OnError();

            return DefaultErrorCodes::PP_ERROR;
        }
    }
} // namespace OptixSensorLidar
