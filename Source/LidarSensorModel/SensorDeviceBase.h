/// @file   SensorDeviceBase.h
///
/// @brief  Base class for sensor device definitions
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2021 LeddarTech Inc. All rights reserved.

#pragma once

#include <filesystem>
#include <unordered_map>

#include "Core/EntityRegistry/Registry.h"

#include "TensorOps/Scope.h"

#include "AcquisitionContext/AcquisitionContext.h"
#include "EnvironmentSampler.h"
#include "SensorModelBase.h"

#include "Serialize/SensorAsset.h"
#include "Serialize/SensorDefinition.h"

/** @brief */
namespace LTSE::SensorModel
{

    /// @brief Represents a physical sensor
    ///
    /// Built on top of a @ref SensorModelBase, this class adds methods which encapsulate the behaviour of a physical sensor.
    class SensorDeviceBase
    {
      public:
        Ref<SensorModelBase> mSensorDefinition{};
        Ref<Scope> mComputationScope = nullptr;

      public:
        /// @brief Constructor
        SensorDeviceBase( uint32_t aMemorySize );

        /// @brief Default copy constructor
        SensorDeviceBase( const SensorDeviceBase & ) = default;

        /// @brief Default destructor
        ~SensorDeviceBase() = default;

        /// @brief Create sampling data using the given sampler configuration
        ///
        /// Returns a reference to an @ref EnvironmentSampler which will create
        /// sample azimuths and elevation data for the given configured tiles, at the given positions
        ///
        /// @tparam _TileType Should either be std::string, or std::vector<std::string>
        /// @tparam _PosType Should either be float, or std::vector<float>
        /// @tparam _TsType Should either be float, or std::vector<float>
        ///
        /// @param aSamplerCreateInfo Sampling parameters
        /// @param aAcqCreateInfo Acquisition parameters
        /// @param aTile Tile ID(s) to include in the sampling
        /// @param aPosition Position(s) for the tiles to include
        /// @param aTimestamp Timestamps(s) for the tiles to include
        ///
        template <typename _TileType, typename _PosType, typename _TsType>
        Ref<EnvironmentSampler> Sample( EnvironmentSampler::sCreateInfo const &aSamplerCreateInfo, AcquisitionSpecification const &aAcqCreateInfo, _TileType const &aTile,
                                        _PosType const &aPosition, _TsType const &aTimestamp )
        {
            auto lTile = ResolveTileID( aTile );
            return DoSample( aSamplerCreateInfo, aAcqCreateInfo, lTile, aPosition, aTimestamp );
        }

        /// @brief Resolve a tile ID into the tile entity
        ///
        /// @param aTileID The tile ID to be resolved into an entity
        ///
        /// @return The tile entity corresponding to the ID
        ///
        /// @throw runtime error if the requested tile ID does not exist
        ///
        Entity ResolveTileID( std::string const &aTileID );

        /// @brief Resolve a vector of tile IDs into a vector of tile entities
        ///
        /// @param aTileIDSequence A list of tile IDs to be resolved into entities
        ///
        /// @return A vector containing the resolved tiles IDs
        ///
        /// @throw runtime error if one of the requested tile IDs does not exist
        ///
        std::vector<Entity> ResolveTileID( std::vector<std::string> const &aTileIDSequence );

        void Clear();

        virtual void Process( Timestep const &aTs, Scope &aScope, AcquisitionContext const &aFlashList, OpNode const &aAzimuth, OpNode const &aElevation, OpNode const &aIntensity,
                              OpNode const &aDistance );

      protected:
        /// @brief Compute the sampling diven a list of laser flashes
        Ref<EnvironmentSampler> DoSample( EnvironmentSampler::sCreateInfo const &SamplerCreateInfo, Ref<Scope> &aComputationScope, AcquisitionContext const &aFlashList );

        /// @brief Compute the sampling based on a resolved tile entity
        ///
        /// @tparam _PosType Type for the positions, should be either math::vec2, or std::vector<math::vec2>
        /// @tparam _TsType Type for the timestamps, should be either float, or std::vector<float>
        ///
        /// @param aSamplerCreateInfo Sampling parameters
        /// @param aAcqCreateInfo Acquisition parameters
        /// @param aTile Tile to include in the sampling
        /// @param aPosition Position(s) for the tiles to include
        /// @param aTimestamp Timestamps(s) for the tiles to include
        ///
        template <typename _PosType, typename _TsType>
        Ref<EnvironmentSampler> DoSample( EnvironmentSampler::sCreateInfo const &aSamplerCreateInfo, AcquisitionSpecification const &aAcqCreateInfo, Entity const &aTile,
                                          _PosType const &aPosition, _TsType const &aTimestamp )
        {
            if( !aTile )
                return nullptr;

            AcquisitionContext lFlashList( aAcqCreateInfo, aTile, aPosition, aTimestamp );

            return DoSample( aSamplerCreateInfo, mComputationScope, lFlashList );
        }

        /// @brief Compute the sampling based on a vector of resolved tile entities
        ///
        /// @tparam _PosType Type for the positions, should be either math::vec2, or std::vector<math::vec2>
        /// @tparam _TsType Type for the timestamps, should be either float, or std::vector<float>
        ///
        /// @param aSamplerCreateInfo Sampling parameters
        /// @param aAcqCreateInfo Acquisition parameters
        /// @param aTile Tiles to include in the sampling
        /// @param aPosition Position(s) for the tiles to include
        /// @param aTimestamp Timestamps(s) for the tiles to include
        ///
        template <typename _PosType, typename _TsType>
        Ref<EnvironmentSampler> DoSample( EnvironmentSampler::sCreateInfo const &aSamplerCreateInfo, AcquisitionSpecification const &aAcqCreateInfo,
                                          std::vector<Entity> const &aTiles, _PosType const &aPosition, _TsType const &aTimestamp )
        {
            if( aTiles.size() == 0 )
                return nullptr;

            AcquisitionContext lFlashList( aAcqCreateInfo, aTiles, aPosition, aTimestamp );

            return DoSample( aSamplerCreateInfo, mComputationScope, lFlashList );
        }
    };
} // namespace LTSE::SensorModel