/// @file   AcquisitionContext.h
///
/// @brief  AcquisitionContext definition
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2022 LeddarTech Inc. All rights reserved.

#pragma once

#include "AcquisitionSpecification.h"

#include "Core/EntityRegistry/Registry.h"
#include "Core/Math/Types.h"

#include "LidarSensorModel/Components.h"

namespace LTSE::SensorModel
{
    using namespace LTSE::Core;

    template <typename _Ty> struct sValueArray : public std::vector<_Ty>
    {
        void Reserve( size_t aNewSize ) { reserve( aNewSize ); }

        void Append( _Ty aValue ) { push_back( aValue ); }

        size_t Size() { return size(); }
    };

    using TextureSampler = Cuda::TextureSampler2D::DeviceData;

    using sUInt32Array  = sValueArray<uint32_t>;
    using sFloat32Array = sValueArray<float>;
    using sStringArray  = sValueArray<std::string>;
    using sTextureArray = sValueArray<TextureSampler>;

    /// @brief Holds a list of positions as a structure of arrays
    struct sPositionArray
    {
        sFloat32Array mX{}; //!< X coordinates
        sFloat32Array mY{}; //!< Y coordinates

        sPositionArray()  = default;
        ~sPositionArray() = default;

        /// @brief Reserve size
        ///
        /// @param aNewsize Size to reserve in the array
        ///
        void Reserve( size_t aNewsize );

        /// @brief Append a point
        ///
        /// @param aPoint Point to append.
        ///
        void Append( math::vec2 aPoint );

        /// @brief Number of points
        size_t Size();
    };

    /// @brief Holds a list of sizes as a structure of arrays
    struct sSizeArray
    {
        sFloat32Array mWidth{};  //!< Widths
        sFloat32Array mHeight{}; //!< Heights

        sSizeArray()  = default;
        ~sSizeArray() = default;

        /// @brief Reserve size
        ///
        /// @param aNewsize Size to reserve in the array
        ///
        void Reserve( size_t aNewsize );

        /// @brief Append a size
        ///
        /// @param aPoint Size to append, encoded as a 2-dimensional vector.
        ///
        void Append( math::vec2 aPoint );

        /// @brief Append a size
        ///
        /// @param aWidth Width to append.
        /// @param aHeight Height to append.
        ///
        void Append( float aWidth, float aHeight );

        /// @brief Number of points
        size_t Size();
    };

    struct sIntervalArray
    {
        sFloat32Array mMin{};
        sFloat32Array mMax{};

        sIntervalArray()  = default;
        ~sIntervalArray() = default;

        /// @brief Reserve size
        ///
        /// @param aNewsize Size to reserve in the array
        ///
        void Reserve( size_t aNewsize );

        /// @brief Append an interval
        ///
        /// @param aPoint Interval to append, encoded as a 2-dimensional vector.
        ///
        void Append( math::vec2 aPoint );

        /// @brief Append a size
        ///
        /// @param aMin Lower bound to append.
        /// @param aMax Upper bound to append.
        ///
        void Append( float aMin, float aMax );

        /// @brief Number of points
        size_t Size();
    };

    class AcquisitionContext
    {
      public:
        AcquisitionSpecification mSpec{};

        /// @brief Create an acquisition context with a single tile
        ///
        /// @param aSpec Acquisition specification.
        /// @param aTile Tile to sample.
        /// @param aPosition Position of the center of the tiole, in world space.
        /// @param aTimestamp Time for the start of the acquisition for this tile.
        ///
        AcquisitionContext( AcquisitionSpecification const &aSpec, Entity const &aTile, math::vec2 const &aPosition, float aTimestamp );

        /// @brief Create an acquisition context with a single tile submitted at multiple positions
        ///
        /// @param aSpec Acquisition specification.
        /// @param aTile Tile to sample.
        /// @param aPositions Positions of the center of the tile, in world space.
        /// @param aTimestamp Time for the start of the acquisition for each sampling of the tile.
        ///
        AcquisitionContext( AcquisitionSpecification const &aSpec, Entity const &aTile, std::vector<math::vec2> const &aPositions, std::vector<float> const &aTimestamp );

        /// @brief Create an acquisition context with a single tile submitted at multiple positions
        ///
        /// @param aSpec Acquisition specification.
        /// @param aTileSequence Tiles to sample.
        /// @param aPositions Positions of the center of the tile, in world space.
        /// @param aTimestamp Time for the start of the acquisition for each sampling of the tile.
        ///
        AcquisitionContext( AcquisitionSpecification const &aSpec, std::vector<Entity> const &aTileSequence, std::vector<math::vec2> const &aPositions,
                            std::vector<float> const &aTimestamp );

      public:
        sValueArray<Entity> mScheduledFlashEntities = {}; //!< List of of all flash entities scheduled for the current frame.

        struct
        {
            sStringArray mTileID{};  //!< ID of the parent tile
            sStringArray mFlashID{}; //!< ID of the flash

            sPositionArray mTilePosition{};  //!< Position of the parent tile
            sPositionArray mWorldPosition{}; //!< Position of the flash, in world space
            sPositionArray mLocalPosition{}; //!< Position of the flash, relative to its parent tile

            sIntervalArray mWorldAzimuth{};   //!< Azimuth interval. This takes diffusion into account if applicable.
            sIntervalArray mWorldElevation{}; //!< Elevation interval. This takes diffusion into account if applicable.
            sSizeArray mFlashSize{};          //!< Width and height of the flash, taking diffusion into account.

            sFloat32Array mTimestamp{}; //!< mTimestamp for this laser flash

            sTextureArray mDiffusion{}; //!< Interpolator for laser diffusion

            /// @brief Sampling information
            struct
            {
                sUInt32Array mLength{};     //!< Number of samples to collect
                sFloat32Array mInterval{};  //!< Sampling interval, equal to (1 / frequency)
                sFloat32Array mFrequency{}; //!< Sampling frequency
            } mSampling;

            /// @brief Laser information
            struct
            {
                sTextureArray mPulseTemplate{}; //!< Pulse template to use
                sFloat32Array mTimebaseDelay{}; //!< Timebase delay
                sFloat32Array mFlashTime{};     //!< Flash time
            } mLaser;

        } mEnvironmentSampling;

        struct
        {
            sValueArray<uint32_t> mPhotoDetectorCellCount{}; //!< Grouping of photodetector cells

            struct
            {
                sUInt32Array mFlashIndex{};                 //!< Index of the flash associated to this photodetector cell.
                sPositionArray mCellPositions{};            //!< Position of the photodetector cell, relative to the center of the photodetector array.
                sPositionArray mCellWorldPositions{};       //!< Position of the photodetector cell, in world space.
                sPositionArray mCellTilePositions{};        //!< Position of the photodetector cell, relative to the current tile.
                sIntervalArray mCellWorldAzimuthBounds{};   //!< Horizontal bounds of the photodetectorcell, in world coordinates.
                sIntervalArray mCellWorldElevationBounds{}; //!< Vertical bounds of the photodetectorcell, in world coordinates.
                sSizeArray mCellSizes{};                    //!< Size of the photodetector cell.
                sFloat32Array mBaseline{};                  //!< Baseline offset for the photodetector cell.
                sFloat32Array mGain{};                      //!< Gain got the corresponding photodetector cell.
                sFloat32Array mStaticNoiseShift{};          //!< Shift in the static noise caused by temperature.
                sTextureArray mStaticNoise{};               //!< Static noise generated by this photodetector cell.
                sTextureArray mElectronicCrosstalk{};       //!< Crosstalk influence.
            } mPhotoDetectorData;

        } mPulseSampling;

      protected:
        void AppendPhotoDetectorData( sPhotoDetector const &aPhotoDetector, math::vec2 aFlashPosition, math::vec2 aTilePosition );
        void AppendPhotoDetectorData( math::vec2 aFlashPosition, math::vec2 aTilePosition );
        void AppendCellPosition( math::vec4 const &aCellPosition, math::vec2 aFlashPosition, math::vec2 aTilePosition );
        void AppendLaserFlash( Entity const &aFlash, math::vec2 aTilePosition, float aTimestamp );

        /// @brief Reset internal data structures
        void ResetInternalStructures( uint32_t aFlashCount, uint32_t aTotalPhotodetectorCellCount, uint32_t qTotalElectronicCrosstalkMatrixSize );
    };

} // namespace LTSE::SensorModel