/// @file   SensorDeviceBase.cpp
///
/// @brief  Implementation file for sensor device definitions
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2021 LeddarTech Inc. All rights reserved.

#pragma once

#include <filesystem>
#include <unordered_map>

#include "Core/EntityRegistry/Registry.h"

#include "Core/Logging.h"
#include "SensorDeviceBase.h"
#include "SensorModelBase.h"

#include "Components.h"
#include "Core/Logging.h"
#include "SensorDeviceBase.h"
#include "SensorModelBase.h"

#include "Components.h"
#include "SensorDeviceBase.h"
#include "SensorModelBase.h"

/** @brief */
namespace LTSE::SensorModel
{
    SensorDeviceBase::SensorDeviceBase( uint32_t aMemorySize )
    {
        Clear();
        mComputationScope = New<Scope>( aMemorySize );
    }

    void SensorDeviceBase::Clear()
    {
        if( mSensorDefinition )
            mSensorDefinition->Clear();
    }

    Entity SensorDeviceBase::ResolveTileID( std::string const &aTileID )
    {
        auto lTile = mSensorDefinition->GetTileByID( aTileID );
        if( !lTile )
            throw std::runtime_error( "Non-existing tile ID" );
        return lTile;
    }

    std::vector<Entity> SensorDeviceBase::ResolveTileID( std::vector<std::string> const &aTileIDSequence )
    {
        std::vector<Entity> lTiles( aTileIDSequence.size() );

        for( uint32_t i = 0; i < aTileIDSequence.size(); i++ )
        {
            auto lTile = mSensorDefinition->GetTileByID( aTileIDSequence[i] );
            if( !lTile )
                throw std::runtime_error( "Non-existing tile ID in sequence" );
            lTiles[i] = lTile;
        }

        return lTiles;
    }

    Ref<EnvironmentSampler> SensorDeviceBase::DoSample( EnvironmentSampler::sCreateInfo const &aSamplerCreateInfo, Ref<Scope> &aComputationScope,
                                                        AcquisitionContext const &aFlashList )
    {
        aComputationScope->Reset();
        Ref<EnvironmentSampler> lSampler = New<EnvironmentSampler>( aSamplerCreateInfo, aComputationScope, aFlashList );
        lSampler->Run();

        return lSampler;
    }

    void SensorDeviceBase::Process( Timestep const &aTs, Scope &aScope, AcquisitionContext const &aFlashList, OpNode const &aAzimuth, OpNode const &aElevation,
                                    OpNode const &aIntensity, OpNode const &aDistance )
    {
        //
    }

} // namespace LTSE::SensorModel