#pragma once
/** @file RayGenerator.h
 *
 * @brief FILE
 */

#include <vector>

#include "Core/Math/Types.h"

#include "Core/Cuda/CudaBuffer.h"
#include "Core/Cuda/ExternalMemory.h"
#include "Core/Cuda/MultiTensor.h"

#include "Scene/ParticleData.h"

namespace LTSE::SensorModel::Dev
{

    using namespace LTSE::Cuda;

    /**
     *  @struct LidarSamplePoint
     */
    struct sLidarCartesianSamplePoint
    {
        uint32_t   mFlashID   = 0xffffffff;
        int        mType      = -1;
        math::vec3 mDirection = { 0.0f, 0.0f, 0.0f };
        float      mDistance  = 0.0f;
        float      mIntensity = 0.0f;

        sLidarCartesianSamplePoint()                                     = default;
        sLidarCartesianSamplePoint( const sLidarCartesianSamplePoint & ) = default;
    };

    struct sFlashVisualizationData
    {
        math::vec2 mFlashPosition = { 0.0f, 0.0f };
        math::vec2 mFlashExtent   = { 0.0f, 0.0f };
        math::vec3 mColor         = { 1.0f, 0.2f, 0.8f };
        float      mMinValue      = 0.0f;
        float      mMaxValue      = 1.0f;
    };

    class sPointCloudVisualizer
    {
      public:
        bool  mHighlightFlashFOV  = false;
        bool  mHeatmapColoring    = true;
        float mHeatmapColorOffset = 120.0f;
        bool  mLogScale           = true;
        bool  mInvertZAxis        = false;
        float mPointDensity       = 0.5;
        float mResolution         = 0.1;

        sPointCloudVisualizer()  = default;
        ~sPointCloudVisualizer() = default;

        void Visualize( math::mat4 a_PointCloudTransform, MultiTensor &a_LidarReturnPoints, GPUExternalMemory &a_Particles );
    };

} // namespace LTSE::SensorModel::Dev
