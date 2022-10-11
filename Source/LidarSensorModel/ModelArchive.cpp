/// @file   ModelArchive.cpp
///
/// @brief  Implementation file for model archiver
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2022 LeddarTech Inc. All rights reserved.

#include "ModelArchive.h"

namespace LTSE::SensorModel
{
    namespace
    {

        template <typename _Ty> static sAssetInternalReference GetAssetInternalReference( Entity const &aAssetEntity )
        {
            if( aAssetEntity.Has<sJoinComponent<_Ty>>() )
            {
                auto lFlashAttenuationEntity = aAssetEntity.Get<sJoinComponent<_Ty>>().mJoinEntity;

                sAssetInternalReference lAssetReference{};
                if( lFlashAttenuationEntity.Has<sInternalAssetReference>() )
                {
                    auto &lDiffuserAssetID = lFlashAttenuationEntity.Get<sInternalAssetReference>();

                    lAssetReference.mAssetID = lDiffuserAssetID.mParentID;
                    lAssetReference.mMapID   = lDiffuserAssetID.mID;
                }
                return lAssetReference;
            }
            else
            {
                return sAssetInternalReference{};
            }
        }

        template <typename _Ty> static std::string GetAssetReference( Entity const &aAssetEntity )
        {
            if( aAssetEntity.Has<sJoinComponent<_Ty>>() && aAssetEntity.Get<sJoinComponent<_Ty>>().mJoinEntity )
            {
                auto lParentAsset = aAssetEntity.Get<sJoinComponent<_Ty>>().mJoinEntity.Get<sRelationshipComponent>().mParent;

                return lParentAsset.Get<sAssetMetadata>().mID;
            }
            else
                return "";
        }

        template <typename _Ty> static std::string GetComponentReference( Entity const &aComponent )
        {
            if( aComponent.Has<sJoinComponent<_Ty>>() && aComponent.Get<sJoinComponent<_Ty>>().mJoinEntity )
                return aComponent.Get<sJoinComponent<_Ty>>().mJoinEntity.Get<sSensorComponent>().mID;
            else
                return "";
        }

        void ArchiveTiles( Ref<SensorModelBase> aSensorDefinition, sSensorDefinition &lNewDefinition )
        {
            std::vector<Entity> lTiles = aSensorDefinition->mRootTile.Get<sRelationshipComponent>().mChildren;

            for( auto &lTileEntity : lTiles )
            {
                sTileData lTileData{};

                auto &lTileSpecification = lTileEntity.Get<sTileSpecificationComponent>();

                lTileData.mID                = lTileSpecification.mID;
                lTileData.mPosition          = lTileSpecification.mPosition;
                lTileData.SamplerComponentID = GetComponentReference<sSampler>( lTileEntity );

                for( auto &lFlashEntity : lTileEntity.Get<sRelationshipComponent>().mChildren )
                {
                    sFlashData lFlashData{};

                    auto &lFlashSpecification = lFlashEntity.Get<sLaserFlashSpecificationComponent>();

                    lFlashData.Area = math::vec4{ lFlashSpecification.mPosition.x, lFlashSpecification.mPosition.y, lFlashSpecification.mExtent.x, lFlashSpecification.mExtent.y };
                    lFlashData.LaserDiodeComponentID = GetComponentReference<sLaserAssembly>( lFlashEntity );

                    lTileData.Flashes.push_back( lFlashData );
                }

                lNewDefinition.Tiles[lTileData.mID] = lTileData;
            }
        }

        static sSensorComponentData ArchiveComponent( sSampler &aComponent )
        {
            sSensorComponentData lComponentData{};

            sSamplerComponentData lComponentSpec{};
            lComponentSpec.mLength    = aComponent.mLength;
            lComponentSpec.mFrequency = aComponent.mFrequency;

            lComponentData.mValue = lComponentSpec;

            return lComponentData;
        }

        template <typename _ComponentType> static sSensorComponentData ArchiveComponent( Entity const &aComponentEntity )
        {
            if( !aComponentEntity.Has<_ComponentType>() )
                return sSensorComponentData{};

            auto lComponentData  = ArchiveComponent( aComponentEntity.Get<_ComponentType>() );
            lComponentData.mName = aComponentEntity.Get<sTag>().mValue;

            return lComponentData;
        }

        void ArchiveComponents( Ref<SensorModelBase> aSensorDefinition, sSensorDefinition &lNewDefinition )
        {
            aSensorDefinition->ForEach<sSensorComponent>(
                [&]( auto aComponentEntity, auto &aMetadata )
                {
                    sSensorComponentData lSampler = ArchiveComponent<sSampler>( aComponentEntity );

                    if( lSampler.Type() != eComponentType::UNKNOWN )
                        lNewDefinition.Components[aMetadata.mID] = lSampler;
                } );
        }

        void ArchiveAssets( Ref<SensorModelBase> aSensorDefinition, sSensorDefinition &lNewDefinition )
        {
            for( auto &lAssetEntity : aSensorDefinition->mRootAsset.Get<sRelationshipComponent>().mChildren )
            {
                sSensorAssetData lAssetData{};

                auto &lMetadata  = lAssetEntity.Get<sAssetMetadata>();
                lAssetData.mName = lAssetEntity.Get<sTag>().mValue;

                auto &lFilePath      = lAssetEntity.Get<sAssetLocation>();
                lAssetData.mRoot     = lFilePath.mRoot;
                lAssetData.mFilePath = lFilePath.mFilePath;

                lNewDefinition.Assets[lMetadata.mID] = lAssetData;
            }
        }

    } // namespace

    void Save( Ref<SensorModelBase> aSensorDefinition, fs::path const &aRoot, fs::path const &aModelFilePath )
    {
        sSensorDefinition lSensorArchiveDefinition{};
        lSensorArchiveDefinition.mName = aSensorDefinition->mName;

        ArchiveAssets( aSensorDefinition, lSensorArchiveDefinition );
        ArchiveComponents( aSensorDefinition, lSensorArchiveDefinition );
        ArchiveTiles( aSensorDefinition, lSensorArchiveDefinition );

        SaveSensorDefinition( lSensorArchiveDefinition, aRoot, aModelFilePath );
    }

} // namespace LTSE::SensorModel