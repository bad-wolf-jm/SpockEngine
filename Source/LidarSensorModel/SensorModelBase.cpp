/// @file   SensorModelBase.cpp
///
/// @brief  Implementation file for SensorModelBase class
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2021 LeddarTech Inc. All rights reserved.

#include "SensorModelBase.h"

#include <algorithm>

#include "Core/Logging.h"

#include "Components.h"
#include "Core/AssetFile.h"

using namespace LTSE::Cuda;
using namespace LTSE::Core;

namespace LTSE::SensorModel
{
    SensorModelBase::SensorModelBase() { Clear(); }

    void SensorModelBase::Clear()
    {
        mRegistry.Clear();
        mRootTile      = mRegistry.CreateEntityWithRelationship();
        mRootAsset     = mRegistry.CreateEntityWithRelationship();
        mRootLayout    = mRegistry.CreateEntityWithRelationship();
        mRootComponent = mRegistry.CreateEntityWithRelationship();
    }

    Entity SensorModelBase::CreateEntity( std::string const &aName, Entity const &aParent )
    {
        auto lNewEntity = mRegistry.CreateEntityWithRelationship( aName );

        if( aParent )
            mRegistry.SetParent( lNewEntity, aParent );

        return lNewEntity;
    }

    Entity SensorModelBase::CreateEntity( std::string const &aName )
    {
        auto lNewEntity = mRegistry.CreateEntityWithRelationship( aName );
        return lNewEntity;
    }

    Entity SensorModelBase::CreateEntity( Entity const &aParent )
    {
        auto lNewEntity = mRegistry.CreateEntityWithRelationship();

        if( aParent )
            mRegistry.SetParent( lNewEntity, aParent );

        return lNewEntity;
    }

    Entity SensorModelBase::CreateEntity()
    {
        auto lNewEntity = mRegistry.CreateEntityWithRelationship();

        return lNewEntity;
    }

    Entity SensorModelBase::CreateTile( std::string const &aTileID, math::vec2 const &aPosition )
    {
        if( mTileIDToTileLUT.find( aTileID ) != mTileIDToTileLUT.end() )
            throw std::runtime_error( "Duplicate tile ID" );

        sTileSpecificationComponent lTileSpecification{};
        lTileSpecification.mID       = aTileID;
        lTileSpecification.mPosition = aPosition;

        mTileIDToTileLUT[aTileID] = CreateEntity( mRootTile, lTileSpecification );

        return mTileIDToTileLUT[aTileID];
    }

    Entity SensorModelBase::CreateFlash( Entity aTile, std::string const &aFlashID, math::vec2 const &aRelativeAngle, math::vec2 const &aExtent )
    {
        sLaserFlashSpecificationComponent lFlashSpecification{};
        lFlashSpecification.mFlashID  = aFlashID;
        lFlashSpecification.mPosition = aRelativeAngle;
        lFlashSpecification.mExtent   = aExtent;

        return CreateEntity( aTile, lFlashSpecification );
    }

    Entity SensorModelBase::CreateTileLayout( std::string aID, std::string aName )
    {
        auto lNewLayout = CreateEntity( aName, mRootLayout );

        lNewLayout.Add<sTileLayoutComponent>();

        mLayoutByID[aID] = lNewLayout;

        return lNewLayout;
    }

    Entity SensorModelBase::CreateTileLayout( std::string aID, sTileLayoutData aData )
    {
        auto lNewLayout = CreateTileLayout( aID, aData.mName );

        auto &lTileLayoutData = lNewLayout.Get<sTileLayoutComponent>();

        for( auto &x : aData.Elements )
            lTileLayoutData.mLayout[x.first] = { x.second.mTileID, x.second.mPosition };

        return lNewLayout;
    }

    Entity SensorModelBase::CreateTileLayout( std::string aName, sTileLayoutComponent aData )
    {
        auto lNewLayout = CreateTileLayout( aData.mID, aName );

        lNewLayout.Replace<sTileLayoutComponent>( aData );

        return lNewLayout;
    }

    std::vector<Entity> SensorModelBase::GetAllTiles()
    {
        std::vector<Entity> lTiles = {};
        ForEach<sTileSpecificationComponent>( [&]( Entity lEntity, sTileSpecificationComponent &lComponent ) { lTiles.push_back( lEntity ); } );

        return lTiles;
    }

    Entity SensorModelBase::GetTileByID( std::string const &aTileID )
    {
        if( mTileIDToTileLUT.count( aTileID ) != 0 )
            return mTileIDToTileLUT[aTileID];

        throw std::runtime_error( "TileID does not exist" );
    }

    Entity SensorModelBase::GetAssetByID( std::string const &aAssetID )
    {
        if( mAssetsByID.count( aAssetID ) != 0 )
            return mAssetsByID[aAssetID];

        throw std::runtime_error( "AssetID does not exist" );
    }

    Entity SensorModelBase::GetLayoutByID( std::string const &aLayoutID )
    {
        if( mLayoutByID.count( aLayoutID ) != 0 )
            return mLayoutByID[aLayoutID];

        throw std::runtime_error( "AssetID does not exist" );
    }

    Entity SensorModelBase::GetComponentByID( std::string const &aComponentID )
    {
        if( mComponentsByID.count( aComponentID ) != 0 )
            return mComponentsByID[aComponentID];

        throw std::runtime_error( "ComponentID does not exist" );
    }

    Entity SensorModelBase::CreateAsset( std::string const &aID, std::string const &aAssetName )
    {
        if( mAssetsByID.find( aID ) != mAssetsByID.end() )
            throw std::runtime_error( "Duplicate asset ID" );

        sAssetMetadata lMetadata{};
        lMetadata.mID = aID;

        auto lNewAsset = CreateEntity( aAssetName, mRootAsset, lMetadata );

        mAssetsByID[aID] = lNewAsset;

        return lNewAsset;
    }

    Entity SensorModelBase::CreateAsset( std::string const &aID, std::string const &aAssetName, fs::path const &aAssetRoot, fs::path const &aAssetPath )
    {
        auto lNewAsset = CreateAsset( aID, aAssetName );

        lNewAsset.Add<sAssetLocation>( sAssetLocation{ aAssetRoot, aAssetPath } );

        return lNewAsset;
    }

    Entity SensorModelBase::LoadAndFillAssetDefinition( Entity aNewAsset, sPhotodetectorAssetData const &aAsset )
    {
        aNewAsset.Tag<sPhotodetectorAssetTag>();
        auto &lMetadata      = aNewAsset.Get<sAssetMetadata>();
        auto &lAssetFileData = aNewAsset.Get<sAssetLocation>();

        BinaryAsset lStaticNoiseData( lAssetFileData.mRoot / aAsset.mStaticNoiseData );
        BinaryAsset lElectronicCrosstalkData( lAssetFileData.mRoot / aAsset.mElectronicXtalkData );

        auto &lPhotoDetectorComponent                = aNewAsset.Add<sPhotoDetector>();
        lPhotoDetectorComponent.mCellPositions       = {};
        lPhotoDetectorComponent.mStaticNoise         = {};
        lPhotoDetectorComponent.mGain                = {};
        lPhotoDetectorComponent.mStaticNoiseShift    = {};
        lPhotoDetectorComponent.mBaseline            = {};
        lPhotoDetectorComponent.mElectronicCrosstalk = {};

        uint32_t lElectronicCrosstalkID = 0;
        for( auto &lAssetData : aAsset.mCells )
        {
            lPhotoDetectorComponent.mCellPositions.push_back( lAssetData.mPosition );
            lPhotoDetectorComponent.mGain.push_back( lAssetData.mGain );
            lPhotoDetectorComponent.mBaseline.push_back( lAssetData.mBaseline );
            lPhotoDetectorComponent.mStaticNoiseShift.push_back( lAssetData.mStaticNoiseShift );

            {
                auto [lTextureData, lTextureSampler] = lStaticNoiseData.Retrieve( lAssetData.mId );

                sStaticNoise lStaticNoiseTemplate{};
                lStaticNoiseTemplate.mInterpolator = LoadTexture( lTextureData, lTextureSampler );

                auto lStaticNoiseEntity = CreateEntity( "", aNewAsset, lStaticNoiseTemplate );
                lPhotoDetectorComponent.mStaticNoise.push_back( lStaticNoiseEntity );
            }

            for( auto &lAssetData : aAsset.mCells )
            {
                auto [lTextureData, lTextureSampler] = lElectronicCrosstalkData.Retrieve( lElectronicCrosstalkID );

                sElectronicCrosstalk lElectronicCrosstalkTemplate{};
                lElectronicCrosstalkTemplate.mInterpolator = LoadTexture( lTextureData, lTextureSampler );

                auto lElectronicCrosstalkEntity = CreateEntity( "", aNewAsset, lElectronicCrosstalkTemplate );
                lPhotoDetectorComponent.mElectronicCrosstalk.push_back( lElectronicCrosstalkEntity );

                lElectronicCrosstalkID++;
            }
        }

        auto &lComponentMetadata = aNewAsset.Add<sSensorComponent>( lMetadata.mID );

        mComponentsByID[lMetadata.mID] = aNewAsset;

        return aNewAsset;
    }

    Entity SensorModelBase::LoadAndFillAssetDefinition( Entity aNewAsset, sLaserAssetData const &aAsset )
    {
        aNewAsset.Tag<sLaserAssemblyAssetTag>();
        auto &lMetadata      = aNewAsset.Get<sAssetMetadata>();
        auto &lAssetFileData = aNewAsset.Get<sAssetLocation>();

        BinaryAsset lDiffuserData( lAssetFileData.mRoot / aAsset.mDiffuser );
        auto [lTextureData, lTextureSampler] = lDiffuserData.Retrieve( 0 );

        sDiffusionPattern lDiffusionPattern{};
        lDiffusionPattern.mInterpolator = LoadTexture( lTextureData, lTextureSampler );
        auto lDiffuserEntity            = CreateEntity( "", aNewAsset, lDiffusionPattern );

        BinaryAsset lWaveformData( lAssetFileData.mRoot / aAsset.mWaveformTemplate );
        auto [lTextureData1, lTextureSampler1] = lWaveformData.Retrieve( 0 );

        sPulseTemplate lWaveformTemplate{};
        lWaveformTemplate.mInterpolator = LoadTexture( lTextureData1, lTextureSampler1 );
        auto lWaveformTemplateEntity    = CreateEntity( "", aNewAsset, lWaveformTemplate );

        auto &lLaserComponent          = aNewAsset.Add<sLaserAssembly>();
        lLaserComponent.mWaveformData  = lWaveformTemplateEntity;
        lLaserComponent.mDiffuserData  = lDiffuserEntity;
        lLaserComponent.mTimebaseDelay = aAsset.mTimebaseDelay;
        lLaserComponent.mFlashTime     = aAsset.mFlashTime;

        aNewAsset.Add<sSensorComponent>( lMetadata.mID );

        mComponentsByID[lMetadata.mID] = aNewAsset;

        return aNewAsset;
    }

    Entity SensorModelBase::CreateAsset( std::string const &aID, sSensorAssetData const &aAssetData )
    {
        auto lNewAsset = CreateAsset( aID, aAssetData.mName, aAssetData.mRoot, aAssetData.mFilePath );

        switch( aAssetData.Type() )
        {
        case eAssetType::PHOTODETECTOR:
            return LoadAndFillAssetDefinition( lNewAsset, aAssetData.Get<sPhotodetectorAssetData>() );
        case eAssetType::LASER_ASSEMBLY:
            return LoadAndFillAssetDefinition( lNewAsset, aAssetData.Get<sLaserAssetData>() );
        default:
            throw std::runtime_error( "Unrecognized asset type" );
        }
    }

    Entity SensorModelBase::LoadAsset( std::string const &aID, std::string const &aAssetName, fs::path const &aAssetRoot, fs::path const &aAssetPath )
    {
        return CreateAsset( aID, ReadAsset( aAssetRoot, aAssetPath, aAssetName ) );
    }

    Ref<LTSE::Cuda::TextureSampler2D> SensorModelBase::LoadTexture( Core::TextureData2D &aTexture, Core::TextureSampler2D &aSampler )
    {
        TextureData2D::sCreateInfo lCreateInfo{};

        LTSE::Cuda::sTextureCreateInfo lTextureCreateInfo;
        lTextureCreateInfo.mFilterMode            = LTSE::Core::eSamplerFilter::LINEAR;
        lTextureCreateInfo.mWrappingMode          = LTSE::Core::eSamplerWrapping::CLAMP_TO_EDGE;
        lTextureCreateInfo.mNormalizedCoordinates = true;

        switch( aTexture.mSpec.mFormat )
        {
        case LTSE::Core::eColorFormat::R32_FLOAT:
            lTextureCreateInfo.mNormalizedValues = false;
            break;
        default:
            lTextureCreateInfo.mNormalizedValues = true;
            break;
        }

        Ref<LTSE::Cuda::Texture2D> lTexture = New<LTSE::Cuda::Texture2D>( lTextureCreateInfo, aTexture.GetImageData() );

        return New<LTSE::Cuda::TextureSampler2D>( lTexture, aSampler.mSamplingSpec );
    }

    Ref<LTSE::Cuda::TextureSampler2D> SensorModelBase::LoadTexture( fs::path const &aTexturePath, std::string const &aName )
    {
        TextureData2D::sCreateInfo lCreateInfo{};
        TextureData2D lTextureData( lCreateInfo, aTexturePath );

        LTSE::Cuda::sTextureCreateInfo lTextureCreateInfo;
        lTextureCreateInfo.mFilterMode            = LTSE::Core::eSamplerFilter::LINEAR;
        lTextureCreateInfo.mWrappingMode          = LTSE::Core::eSamplerWrapping::CLAMP_TO_EDGE;
        lTextureCreateInfo.mNormalizedCoordinates = true;

        switch( lTextureData.mSpec.mFormat )
        {
        case LTSE::Core::eColorFormat::R32_FLOAT:
            lTextureCreateInfo.mNormalizedValues = false;
            break;
        default:
            lTextureCreateInfo.mNormalizedValues = true;
            break;
        }

        Ref<LTSE::Cuda::Texture2D> lTexture = New<LTSE::Cuda::Texture2D>( lTextureCreateInfo, lTextureData.GetImageData() );

        sTextureSamplingInfo lSamplingInfo{};
        lSamplingInfo.mScaling       = std::array<float, 2>{ 1.0f, 1.0f };
        lSamplingInfo.mMinification  = LTSE::Core::eSamplerFilter::LINEAR;
        lSamplingInfo.mMagnification = LTSE::Core::eSamplerFilter::LINEAR;
        lSamplingInfo.mWrapping      = LTSE::Core::eSamplerWrapping::CLAMP_TO_EDGE;

        return New<LTSE::Cuda::TextureSampler2D>( lTexture, lSamplingInfo );
    }

    Entity SensorModelBase::CreateElement( std::string const &aID, std::string const &aComponentName )
    {
        auto lNewComponent = CreateEntity( aComponentName, mRootComponent );

        auto &lComponentMetadata = lNewComponent.Add<sSensorComponent>();
        lComponentMetadata.mID   = aID;

        mComponentsByID[aID] = lNewComponent;

        return lNewComponent;
    }

    Entity SensorModelBase::CreateElement( std::string const &aID, sSensorComponentData const &aComponent )
    {
        switch( aComponent.Type() )
        {
        case eComponentType::SAMPLER:
        {
            auto &lSamplerSpec = aComponent.Get<sSamplerComponentData>();

            sSampler lSamplerComponent{};
            lSamplerComponent.mLength    = lSamplerSpec.mLength;
            lSamplerComponent.mFrequency = lSamplerSpec.mFrequency;

            return CreateElement( aComponent.mName, aID, lSamplerComponent );
        }
        default:
            throw std::runtime_error( "Unrecognized component type (this should not happen)" );
        }
    }
} // namespace LTSE::SensorModel
