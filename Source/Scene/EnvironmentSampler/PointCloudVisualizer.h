#pragma once
/** @file RayGenerator.h
 *
 * @brief FILE
 */

#include <vector>

#include "Core/Math/Types.h"

#include "Core/CUDA/Array/CudaBuffer.h"
#include "Core/CUDA/Array/MultiTensor.h"

#include "Scene/ParticleData.h"

namespace SE::SensorModel::Dev
{
    using namespace math;
    using namespace SE::Cuda;

    /**
     *  @struct LidarSamplePoint
     */
    struct sLidarCartesianSamplePoint
    {
        uint32_t mFlashID   = 0xffffffff;
        int      mType      = -1;
        vec3     mDirection = { 0.0f, 0.0f, 0.0f };
        float    mDistance  = 0.0f;
        float    mIntensity = 0.0f;

        sLidarCartesianSamplePoint()                                     = default;
        sLidarCartesianSamplePoint( const sLidarCartesianSamplePoint & ) = default;
    };

    struct sFlashVisualizationData
    {
        vec2  mFlashPosition = { 0.0f, 0.0f };
        vec2  mFlashExtent   = { 0.0f, 0.0f };
        vec3  mColor         = { 1.0f, 0.2f, 0.8f };
        float mMinValue      = 0.0f;
        float mMaxValue      = 1.0f;
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

        void Visualize( mat4 a_PointCloudTransform, MultiTensor &a_LidarReturnPoints, GPUMemory &a_Particles );
    };

} // namespace SE::SensorModel::Dev
