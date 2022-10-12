/// @file   EnvironmentSampler.h
///
/// @brief  Environment sampler implementation
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2021 LeddarTech Inc. All rights reserved.

#pragma once

#include "Core/EntityRegistry/Registry.h"
#include "Core/Math/Types.h"
#include "Core/Memory.h"

#include "AcquisitionContext/AcquisitionContext.h"

#include "TensorOps/Scope.h"

/** @brief */
namespace LTSE::SensorModel
{
    using namespace LTSE::Core;
    using namespace LTSE::TensorOps;

    /// @class EnvironmentSampler
    ///
    /// The environment sampler class creates a set of azimuths, elevations and ray intensities to be used
    /// to query a simulated 3D environment in order to get baseline detections for the simulated lidar sensor.
    /// The sampling is based on the information provided by a list of flashes, and is returned as four dataframe
    /// columns containing the requested information information.
    ///
    /// Environment sampling produces a disjoint union of points sampled from a list of rectangles in the plane.
    /// This sampling of each rectangle can be done using one of three methods, which can be chosen by appropriately
    /// setting values in the definition structure. All three methods first subdivide each rectangle into regular
    /// (non-overlapping) subrectangles, which we shall call pixels, and pick a certain number of points in each.
    /// In each pixel, we can either pick the center point, pick a random finite set of points, or further subdivide
    /// each pixel into subpixels, and pick the center point of those. Note that the first and third methods, though
    /// they seem similar, have a subtle difference. Using the third method, all samples coming from the same pixel are
    /// contiguous in memory, which makes this form of regular sampling amenable to filtering on a per-pixel basis.
    ///
    class EnvironmentSampler
    {
      public:
        /// @class sCreateInfo
        ///
        /// Definition parameters for the environment sampler. This structure determined the method by which the environment will be sampled.
        struct sCreateInfo
        {
            float mLaserPower = 1.0f; //!< Laser power

            math::vec2 mSamplingResolution = { 0.1f, 0.1f }; //!< Size of each pixel in world units, in this case, degrees. The rectangle defined by the laser flash to sample will
                                                             //!< be divided into a regular grid of pixels from which the samples are chosen. This is done to ensire a uniform
                                                             //!< distribution of samples throughout the rectangle, no matter what the distribution of the sampling is.
            bool mUseRegularMultiSampling =
                false; //!< If this is set to `true`, and `mMultiSamplingFactor` is set to a value greater than 1, each pixel in the area to sample will be further divided into a
                       //!< regular grid of size @f$ \lceil\sqrt{N}\rceil\times\lceil\sqrt{N}\rceil @f$. The center point of each of the finer subpixels is picked as the sample.
            uint32_t mMultiSamplingFactor = 1; //!< Multisampling factor for the current frame. This value determined the number of sample points contained in eaxh pixel. If set to
                                               //!< 1, the center of each pixel is picked as the sample.
        };

        sCreateInfo mSpec; //!< Copy of the parameter structure used to create the instance of the environment sampler.

        /// @brief Default constructor.
        EnvironmentSampler() = default;

        /// @brief Constructs an environment sampler using the provided computation scope
        ///
        /// @param aSpec Parameter structure.
        /// @param aScope Computation scope.
        /// @param aFlashList Laser flashes to sample.
        ///.
        EnvironmentSampler( sCreateInfo const &aSpec, Ref<Scope> aScope, AcquisitionContext const &aFlashList );

        /// @brief Constructs an environment sampler using an internal computation scope
        ///
        /// @param aSpec Parameter structure.
        /// @param aPoolSize Size of the scope to create.
        /// @param aFlashList Laser flashes to sample.
        ///
        EnvironmentSampler( sCreateInfo const &aSpec, uint32_t aPoolSize, AcquisitionContext const &aFlashList );

        /// @brief Default destructor.
        ~EnvironmentSampler() = default;

        /// @brief Performs the actual sampling
        ///
        /// The values for azimuth, elevation, intensity and timestamp are only valid after this function is called.
        ///
        void Run();

        /// @brief Retrieve a node by name
        OpNode operator[]( std::string aNodeName );

        /// @brief Returns the number of flashes
        uint32_t GetScheduledFlashCount() { return mScheduledFlashCount; }

        /// @brief Retrieve the acquisition context used to generate the environment sampling
        AcquisitionContext &GetScheduledFlashes() { return mFlashList; }

        sTensorShape mSamplingShape;

      protected:
        struct
        {
            std::vector<float> mMin = {};
            std::vector<float> mMax = {};
        } mWorldAzimuth;

        struct
        {
            std::vector<float> mMin = {};
            std::vector<float> mMax = {};
        } mWorldElevation;

        std::vector<float> mTimestamp = {};

        uint32_t mScheduledFlashCount = 0;
        AcquisitionContext mFlashList;

      public:
        Ref<Scope> mScope; //!< Reference to the computation scope.

        OpNode mFlashIdLUT{};           //!< TensorNode representing a mapping between generated rays to the index of the rectangle they came from
        OpNode mSampledAzimuths{};      //!< TensorNode representing the resulting sampled azimuths, in degrees, with negative values pointing to the left.
        OpNode mSampledElevations{};    //!< TensorNode representing the resulting sampled elevations, in degrees, with negative values pointing down.
        OpNode mRelativeAzimuths{};     //!< TensorNode representing the resulting sampled azimuths, in degrees, with negative values pointing to the left.
        OpNode mRelativeElevations{};   //!< TensorNode representing the resulting sampled elevations, in degrees, with negative values pointing down.
        OpNode mSampleRayIntensities{}; //!< TensorNode representing the sample ray intensities.
        OpNode mTimestamps{};           //!< TensorNode representing the sample ray timestamps.

      protected:
        /// @brief Creates the computation graph.
        void CreateGraph();

        /// @brief Ranges a range node.
        OpNode CreateRangeNode( Scope &aScope, std::vector<float> const &aStart, std::vector<float> const &aEnd, float aDelta );

        /// @brief Created two columns representing the cartesian product of the two inputs.
        std::tuple<OpNode, OpNode> Prod( Scope &aScope, OpNode const &aLeft, OpNode const &aRight );
    };

} // namespace LTSE::SensorModel
