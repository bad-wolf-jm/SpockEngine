#include <functional>

#include "SensorModel.h"

#include "Scripting/PrimitiveTypes.h"

#include "Core/Memory.h"

#include "LidarSensorModel/AcquisitionContext/AcquisitionContext.h"
#include "LidarSensorModel/EnvironmentSampler.h"
#include "LidarSensorModel/SensorModelBase.h"
// #include "ProcessingNodes/ElectronicCrosstalkWaveforms.h"
// #include "ProcessingNodes/WaveformGenerator.h"

namespace LTSE::Core
{
    using namespace sol;
    using namespace LTSE::SensorModel;
    using namespace entt::literals;

    AcquisitionSpecification ParseAcquisitionSpecification( table aSpecTable )
    {
        AcquisitionSpecification lCreateInfo{};
        lCreateInfo.mBasePoints       = aSpecTable["basepoints"].valid() ? aSpecTable["basepoints"] : 1;
        lCreateInfo.mOversampling     = aSpecTable["oversampling"].valid() ? aSpecTable["oversampling"] : 1;
        lCreateInfo.mAccumulation     = aSpecTable["accumulation"].valid() ? aSpecTable["accumulation"] : 1;
        lCreateInfo.mAPDBias          = aSpecTable["apd_bias"].valid() ? aSpecTable["apd_bias"] : 0.0f;
        lCreateInfo.mTIAGain          = aSpecTable["tia_gain"].valid() ? aSpecTable["tia_gain"] : 1.0f;
        lCreateInfo.mTemperature      = aSpecTable["system_temperature"].valid() ? aSpecTable["system_temperature"] : 0.0f;
        lCreateInfo.mAmbientNoise.mAC = aSpecTable["ac_noise"].valid() ? aSpecTable["ac_noise"] : 0.0f;
        lCreateInfo.mAmbientNoise.mDC = aSpecTable["dc_noise"].valid() ? aSpecTable["dc_noise"] : 0.0f;

        return lCreateInfo;
    }

    EnvironmentSampler::sCreateInfo ParseEnvironmentSamplerSpecification( table aSpecTable )
    {
        EnvironmentSampler::sCreateInfo lCreateInfo{};
        lCreateInfo.mLaserPower              = aSpecTable["laser_power"].valid() ? aSpecTable["laser_power"] : 1.0f;
        lCreateInfo.mSamplingResolution      = aSpecTable["sampling_resolution"].valid() ? aSpecTable["sampling_resolution"] : math::vec2{ 0.1f, 0.1f };
        lCreateInfo.mUseRegularMultiSampling = aSpecTable["use_regular_multisampling"].valid() ? aSpecTable["use_regular_multisampling"] : false;
        lCreateInfo.mMultiSamplingFactor     = aSpecTable["multisampling_factor"].valid() ? aSpecTable["multisampling_factor"] : 1;

        return lCreateInfo;
    }

    void OpenSensorModelLibrary( sol::table &aModule )
    {

        // aModule["ResolveWaveforms"] = SensorModel::ResolveWaveforms;
        // aModule["ResolveElectronicCrosstalkWaveforms"] = SensorModel::ResolveElectronicCrosstalkWaveforms;

        auto lTileSpecType = DeclarePrimitiveType<sTileSpecificationComponent>( aModule, "sTileSpecificationComponent" );

        auto lFlashSpecType = DeclarePrimitiveType<sLaserFlashSpecificationComponent>( aModule, "sLaserFlashSpecificationComponent" );
        lFlashSpecType["position"] = &sLaserFlashSpecificationComponent::mPosition;
        lFlashSpecType["timebase_delay"] = &sLaserFlashSpecificationComponent::mTimebaseDelay;
        lFlashSpecType["extent"] = &sLaserFlashSpecificationComponent::mExtent;
        lFlashSpecType["id"] = &sLaserFlashSpecificationComponent::mFlashID;

        DeclarePrimitiveType<sAssetMetadata>( aModule, "sAssetMetadata" );
        DeclarePrimitiveType<sAssetLocation>( aModule, "sAssetLocation" );
        DeclarePrimitiveType<sInternalAssetReference>( aModule, "sInternalAssetReference" );

        DeclarePrimitiveType<sDiffusionAssetTag>( aModule, "sDiffusionAssetTag" );
        DeclarePrimitiveType<sReductionMapAssetTag>( aModule, "sReductionMapAssetTag" );
        DeclarePrimitiveType<sPulseTemplateAssetTag>( aModule, "sPulseTemplateAssetTag" );
        DeclarePrimitiveType<sStaticNoiseAssetTag>( aModule, "sStaticNoiseAssetTag" );
        DeclarePrimitiveType<sPhotodetectorAssetTag>( aModule, "sPhotodetectorAssetTag" );
        DeclarePrimitiveType<sLaserAssemblyAssetTag>( aModule, "sLaserAssemblyAssetTag" );

        auto lDiffuserType = DeclarePrimitiveType<sDiffusionPattern>( aModule, "sDiffusionPattern" );
        lDiffuserType["interpolator"] = &sDiffusionPattern::mInterpolator;

        auto lPulseTemplateType = DeclarePrimitiveType<sPulseTemplate>( aModule, "sPulseTemplate" );
        lPulseTemplateType["interpolator"] = &sPulseTemplate::mInterpolator;

        auto lStaticNoiseTemplateType = DeclarePrimitiveType<sStaticNoise>( aModule, "sStaticNoise" );
        lStaticNoiseTemplateType["interpolator"] = &sStaticNoise::mInterpolator;

        auto lElectronicCrosstalkTemplateType = DeclarePrimitiveType<sElectronicCrosstalk>( aModule, "sElectronicCrosstalk" );
        lElectronicCrosstalkTemplateType["interpolator"] = &sElectronicCrosstalk::mInterpolator;

        DeclarePrimitiveType<sSensorComponent>( aModule, "sSensorComponent" );

        auto lPhotodetectorType = DeclarePrimitiveType<sPhotoDetector>( aModule, "sPhotoDetector" );
        lPhotodetectorType["cell_positions"] = &sPhotoDetector::mCellPositions;
        lPhotodetectorType["gain"] = &sPhotoDetector::mGain;
        lPhotodetectorType["baseline"] = &sPhotoDetector::mBaseline;
        lPhotodetectorType["static_noise_shift"] = &sPhotoDetector::mStaticNoiseShift;
        lPhotodetectorType["static_noise"] = &sPhotoDetector::mStaticNoise;
        lPhotodetectorType["extalk_templates"] = &sPhotoDetector::mElectronicCrosstalk;

        auto lLaserAssemblyType = DeclarePrimitiveType<sLaserAssembly>( aModule, "sLaserAssembly" );
        lLaserAssemblyType["waveform_data"] = &sLaserAssembly::mWaveformData;
        lLaserAssemblyType["diffuser_data"] = &sLaserAssembly::mDiffuserData;
        lLaserAssemblyType["timebase_delay"] = &sLaserAssembly::mTimebaseDelay;
        lLaserAssemblyType["flash_time"] = &sLaserAssembly::mFlashTime;



        DeclarePrimitiveType<sSampler>( aModule, "sSampler" );

        DeclarePrimitiveType<sTileLayoutComponent>( aModule, "sTileLayoutComponent" );

        auto lNewType = aModule.new_usertype<SensorModelBase>( "SensorModel" );

        lNewType[sol::call_constructor] = []() { return New<SensorModelBase>(); };

        lNewType["create_entity"] =
            overload( []( SensorModelBase &aSelf ) { return aSelf.CreateEntity(); }, []( SensorModelBase &aSelf, std::string const &aName ) { return aSelf.CreateEntity( aName ); },
                      []( SensorModelBase &aSelf, Entity aParent ) { return aSelf.CreateEntity( aParent ); },
                      []( SensorModelBase &aSelf, std::string const &aName, Entity aParent ) { return aSelf.CreateEntity( aName, aParent ); },
                      []( SensorModelBase &aSelf, std::string const &aName, const sol::table &aComponent, sol::this_state s ) -> sol::object
                      {
                          if( !aComponent.valid() )
                              return sol::lua_nil_t{};

                          const auto lMaybeAny = InvokeMetaFunction( GetTypeID( aComponent ), "CreateSensorEntity0"_hs, &aSelf, aName, aComponent, s );
                          return lMaybeAny ? lMaybeAny.cast<sol::reference>() : sol::lua_nil_t{};
                      },
                      []( SensorModelBase &aSelf, std::string const &aName, Entity const &aParent, const sol::table &aComponent, sol::this_state s ) -> sol::object
                      {
                          if( !aComponent.valid() )
                              return sol::lua_nil_t{};

                          const auto lMaybeAny = InvokeMetaFunction( GetTypeID( aComponent ), "CreateSensorEntity1"_hs, &aSelf, aName, aParent, aComponent, s );
                          return lMaybeAny ? lMaybeAny.cast<sol::reference>() : sol::lua_nil_t{};
                      },
                      []( SensorModelBase &aSelf, Entity const &aParent, const sol::table &aComponent, sol::this_state s ) -> sol::object
                      {
                          if( !aComponent.valid() )
                              return sol::lua_nil_t{};

                          const auto lMaybeAny = InvokeMetaFunction( GetTypeID( aComponent ), "CreateSensorEntity2"_hs, &aSelf, aParent, aComponent, s );
                          return lMaybeAny ? lMaybeAny.cast<sol::reference>() : sol::lua_nil_t{};
                      } );

        lNewType["create_tile"] = []( SensorModelBase &aSelf, std::string const &aTileID, math::vec2 const &aPosition ) { return aSelf.CreateTile( aTileID, aPosition ); };

        lNewType["create_flash"] = []( SensorModelBase &aSelf, Entity aTile, std::string const &aFlashID, math::vec2 const &aRelativeAngle, math::vec2 const &aExtent )
        { return aSelf.CreateFlash( aTile, aFlashID, aRelativeAngle, aExtent ); };

        lNewType["get_tile_by_id"]      = []( SensorModelBase &aSelf, std::string const &aTileID ) { return aSelf.GetTileByID( aTileID ); };
        lNewType["get_asset_by_id"]     = []( SensorModelBase &aSelf, std::string const &aAssetID ) { return aSelf.GetAssetByID( aAssetID ); };
        lNewType["get_layout_by_id"]    = []( SensorModelBase &aSelf, std::string const &aLayoutID ) { return aSelf.GetLayoutByID( aLayoutID ); };
        lNewType["get_component_by_id"] = []( SensorModelBase &aSelf, std::string const &aComponentID ) { return aSelf.GetComponentByID( aComponentID ); };

        lNewType["load_asset"] = []( SensorModelBase &aSelf, std::string const &aID, std::string const &aAssetName, std::string const &aAssetRoot, std::string const &aAssetPath )
        { return aSelf.LoadAsset( aID, aAssetName, aAssetRoot, aAssetPath ); };

        auto lAcquisitionContextType = aModule.new_usertype<AcquisitionContext>( "AcquisitionContext" );
        lAcquisitionContextType[sol::call_constructor] =
            factories( []( AcquisitionSpecification aSpec, Entity const &aTile, std::vector<math::vec2> const &aPosition, std::vector<float> aTimestamp )
                       { return AcquisitionContext( aSpec, aTile, aPosition, aTimestamp ); },
                       []( AcquisitionSpecification aSpec, std::vector<Entity> const &aTile, std::vector<math::vec2> const &aPosition, std::vector<float> aTimestamp )
                       { return AcquisitionContext( aSpec, aTile, aPosition, aTimestamp ); },
                       []( AcquisitionSpecification aSpec, Entity const &aTile, math::vec2 const &aPosition, float aTimestamp )
                       { return AcquisitionContext( aSpec, aTile, aPosition, aTimestamp ); },
                       []( table aSpecTable, Entity const &aTile, math::vec2 const &aPosition, float aTimestamp )
                       {
                           AcquisitionSpecification aSpec = ParseAcquisitionSpecification( aSpecTable );

                           return AcquisitionContext( aSpec, aTile, aPosition, aTimestamp );
                       },
                       []( table aSpecTable, Entity const &aTile, std::vector<math::vec2> const &aPositions, std::vector<float> aTimestamps )
                       {
                           AcquisitionSpecification aSpec = ParseAcquisitionSpecification( aSpecTable );

                           return AcquisitionContext( aSpec, aTile, aPositions, aTimestamps );
                       },

                       []( table aSpecTable, std::vector<Entity> const &aTile, std::vector<math::vec2> const &aPositions, std::vector<float> aTimestamps )
                       {
                           AcquisitionSpecification aSpec = ParseAcquisitionSpecification( aSpecTable );

                           return AcquisitionContext( aSpec, aTile, aPositions, aTimestamps );
                       } );

        lAcquisitionContextType["tile_ids"]         = []( AcquisitionContext &aSelf ) { return aSelf.mEnvironmentSampling.mTileID; };
        lAcquisitionContextType["flash_ids"]        = []( AcquisitionContext &aSelf ) { return aSelf.mEnvironmentSampling.mFlashID; };
        lAcquisitionContextType["tile_positions"]   = []( AcquisitionContext &aSelf ) { return aSelf.mEnvironmentSampling.mTilePosition; };
        lAcquisitionContextType["world_positions"]  = []( AcquisitionContext &aSelf ) { return aSelf.mEnvironmentSampling.mWorldPosition; };
        lAcquisitionContextType["local_positions"]  = []( AcquisitionContext &aSelf ) { return aSelf.mEnvironmentSampling.mLocalPosition; };
        lAcquisitionContextType["world_azimuths"]   = []( AcquisitionContext &aSelf ) { return aSelf.mEnvironmentSampling.mWorldAzimuth; };
        lAcquisitionContextType["world_elevations"] = []( AcquisitionContext &aSelf ) { return aSelf.mEnvironmentSampling.mWorldElevation; };
        lAcquisitionContextType["flash_sizes"]      = []( AcquisitionContext &aSelf ) { return aSelf.mEnvironmentSampling.mFlashSize; };
        lAcquisitionContextType["timestamps"]       = []( AcquisitionContext &aSelf ) { return aSelf.mEnvironmentSampling.mTimestamp; };
        lAcquisitionContextType["diffusion"]        = []( AcquisitionContext &aSelf ) { return aSelf.mEnvironmentSampling.mDiffusion; };

        lAcquisitionContextType["sampling_lengths"]     = []( AcquisitionContext &aSelf ) { return aSelf.mEnvironmentSampling.mSampling.mLength; };
        lAcquisitionContextType["sampling_intervals"]   = []( AcquisitionContext &aSelf ) { return aSelf.mEnvironmentSampling.mSampling.mInterval; };
        lAcquisitionContextType["sampling_frequencies"] = []( AcquisitionContext &aSelf ) { return aSelf.mEnvironmentSampling.mSampling.mFrequency; };

        lAcquisitionContextType["pulse_templates"] = []( AcquisitionContext &aSelf ) { return aSelf.mEnvironmentSampling.mLaser.mPulseTemplate; };
        lAcquisitionContextType["timebase_delays"] = []( AcquisitionContext &aSelf ) { return aSelf.mEnvironmentSampling.mLaser.mTimebaseDelay; };
        lAcquisitionContextType["flash_times"]     = []( AcquisitionContext &aSelf ) { return aSelf.mEnvironmentSampling.mLaser.mFlashTime; };

        lAcquisitionContextType["detector_cell_count"] = []( AcquisitionContext &aSelf ) { return aSelf.mPulseSampling.mPhotoDetectorCellCount; };

        lAcquisitionContextType["detector_cell_flash_indices"]       = []( AcquisitionContext &aSelf ) { return aSelf.mPulseSampling.mPhotoDetectorData.mFlashIndex; };
        lAcquisitionContextType["detector_cell_positions"]           = []( AcquisitionContext &aSelf ) { return aSelf.mPulseSampling.mPhotoDetectorData.mCellPositions; };
        lAcquisitionContextType["detector_cell_world_positions"]     = []( AcquisitionContext &aSelf ) { return aSelf.mPulseSampling.mPhotoDetectorData.mCellWorldPositions; };
        lAcquisitionContextType["detector_cell_tile_position"]       = []( AcquisitionContext &aSelf ) { return aSelf.mPulseSampling.mPhotoDetectorData.mCellTilePositions; };
        lAcquisitionContextType["detector_cell_world_azimuth_bound"] = []( AcquisitionContext &aSelf ) { return aSelf.mPulseSampling.mPhotoDetectorData.mCellWorldAzimuthBounds; };
        lAcquisitionContextType["detector_cell_world_elevation_bound"] = []( AcquisitionContext &aSelf )
        { return aSelf.mPulseSampling.mPhotoDetectorData.mCellWorldElevationBounds; };
        lAcquisitionContextType["detector_cell_sizes"]                = []( AcquisitionContext &aSelf ) { return aSelf.mPulseSampling.mPhotoDetectorData.mCellSizes; };
        lAcquisitionContextType["detector_cell_baseline"]             = []( AcquisitionContext &aSelf ) { return aSelf.mPulseSampling.mPhotoDetectorData.mBaseline; };
        lAcquisitionContextType["detector_cell_gain"]                 = []( AcquisitionContext &aSelf ) { return aSelf.mPulseSampling.mPhotoDetectorData.mGain; };
        lAcquisitionContextType["detector_cell_static_noise_shift"]   = []( AcquisitionContext &aSelf ) { return aSelf.mPulseSampling.mPhotoDetectorData.mStaticNoiseShift; };
        lAcquisitionContextType["detector_cell_static_noise"]         = []( AcquisitionContext &aSelf ) { return aSelf.mPulseSampling.mPhotoDetectorData.mStaticNoise; };
        lAcquisitionContextType["detector_cell_electronic_crosstalk"] = []( AcquisitionContext &aSelf ) { return aSelf.mPulseSampling.mPhotoDetectorData.mElectronicCrosstalk; };

        auto lEnvironmenSamplerType = aModule.new_usertype<EnvironmentSampler>( "EnvironmentSampler" );

        lEnvironmenSamplerType[sol::call_constructor] = factories(
            []( EnvironmentSampler::sCreateInfo aSpec, Ref<Scope> aScope, AcquisitionContext const &aFlashList )
            {
                return New<EnvironmentSampler>( aSpec, aScope, aFlashList );
            },
            []( table aSpecTable, Ref<Scope> aScope, AcquisitionContext const &aFlashList )
            {
                EnvironmentSampler::sCreateInfo aSpec = ParseEnvironmentSamplerSpecification( aSpecTable );

                return New<EnvironmentSampler>( aSpec, aScope, aFlashList );
            },
            []( table aSpecTable, uint32_t aPoolSize, AcquisitionContext const &aFlashList )
            {
                EnvironmentSampler::sCreateInfo aSpec = ParseEnvironmentSamplerSpecification( aSpecTable );

                return New<EnvironmentSampler>( aSpec, aPoolSize, aFlashList );
            } );

        lEnvironmenSamplerType["run"] = &EnvironmentSampler::Run;

        lEnvironmenSamplerType["get_acquisition_context"] = &EnvironmentSampler::GetScheduledFlashes;
        lEnvironmenSamplerType["get_flash_count"]         = &EnvironmentSampler::GetScheduledFlashCount;
        lEnvironmenSamplerType["azimuths"]                = readonly( &EnvironmentSampler::mSampledAzimuths );
        lEnvironmenSamplerType["elevation"]               = readonly( &EnvironmentSampler::mSampledElevations );
        lEnvironmenSamplerType["intensities"]             = readonly( &EnvironmentSampler::mSampleRayIntensities );
        lEnvironmenSamplerType["timestamps"]              = readonly( &EnvironmentSampler::mTimestamps );
        lEnvironmenSamplerType["relative_azimuths"]       = readonly( &EnvironmentSampler::mRelativeAzimuths );
        lEnvironmenSamplerType["relative_elevation"]      = readonly( &EnvironmentSampler::mRelativeElevations );
    }
} // namespace LTSE::Core