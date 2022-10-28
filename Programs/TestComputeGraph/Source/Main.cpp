#include "Core/Platform/EngineLoop.h"
#include "Graphics/API/UI/UIContext.h"

#include "UI/UI.h"
#include "UI/Widgets.h"

#include "Core/Cuda/CudaBuffer.h"
#include "Core/Cuda/ExternalMemory.h"

#include "Core/Logging.h"
#include "Core/Memory.h"

#include <filesystem>

#include "TensorOps/Implementation/KernelLaunchers.h"
#include "TensorOps/NodeComponents.h"
#include "TensorOps/Scope.h"

#include "Serialize/SensorAsset.h"

#include "ProcessingNodes/WaveformGenerator.h"

#include "LidarSensorModel/Components.h"
#include "LidarSensorModel/EnvironmentSampler.h"
#include "LidarSensorModel/SensorDeviceBase.h"
#include "LidarSensorModel/SensorModelBase.h"

// #include "WaveformGener/ator.h"

#include <chrono>

using namespace LTSE::Core;
using namespace LTSE::TensorOps;
using namespace LTSE::SensorModel;
namespace fs = std::filesystem;

EngineLoop g_EngineLoop;

Ref<SensorDeviceBase> g_SensorModel;

math::vec2 g_PhotodetectorCellSize = { 0.2f, 0.2f };
math::vec2 g_TileFOV;
size_t g_FlashCount  = 32;
float g_FlashSpacing = 0.205;

Entity g_Tile{};
Entity g_Attenuation{};
Entity g_Reduction{};
Entity g_WaveformTemplate{};

std::vector<float> g_Waveforms;

float g_EnvironmentSamplingResolution = 0.2f;
float g_TargetDistance                = 2.0f;
constexpr float SPEED_OF_LIGHT        = 0.299792458f;

uint32_t frameCounter = 0;
float fpsTimer        = 0.0f;
uint32_t lastFPS      = 0;

float ComputeFpsTimer        = 0.0f;
uint32_t ComputeLastFPS      = 0;
uint32_t ComputeFrameCounter = 0;

struct Profile
{
    std::string Name  = "P";
    int64_t StartTime = 0;

    Profile( std::string a_Name )
        : Name{ a_Name }
    {
        auto now    = std::chrono::system_clock::now();
        auto now_ns = std::chrono::time_point_cast<std::chrono::microseconds>( now );
        StartTime   = now_ns.time_since_epoch().count();
    }

    ~Profile()
    {
        auto now    = std::chrono::system_clock::now();
        auto now_ns = std::chrono::time_point_cast<std::chrono::microseconds>( now );
        LTSE::Logging::Info( "{} - {}", Name, static_cast<float>( now_ns.time_since_epoch().count() - StartTime ) / 1000.0f );
    }
};

size_t l_PoolSize = 1024 * 1024 * 1024 * 3;

void RunPipeline( Scope &g_ComputeScope )
{

    EnvironmentSampler::sCreateInfo lSamplerCreateInfo{};
    lSamplerCreateInfo.mUseRegularMultiSampling = false;
    lSamplerCreateInfo.mSamplingResolution      = { g_EnvironmentSamplingResolution, g_EnvironmentSamplingResolution };
    lSamplerCreateInfo.mMultiSamplingFactor     = 8;

    Ref<EnvironmentSampler> lSamples = g_SensorModel->Sample( lSamplerCreateInfo, "0", math::vec2{ 0.0f, 0.0f }, 0.0f );

    auto l_SampledAzimuths      = ( *lSamples )["Azimuth"];
    auto l_SampledElevations    = ( *lSamples )["Elevation"];
    auto l_SampleRayIntensities = ( *lSamples )["Intensity"];

    // Compute azimuth and elevation universal coordinates
    uint32_t l_ScheduledFlashCount     = lSamples->GetScheduledFlashCount();
    LaserFlashList &l_ScheduledFlashes = lSamples->GetScheduledFlashes();

    auto l_SampleRayIntensitiesFlattened = Flatten( g_ComputeScope, l_SampleRayIntensities );

    auto l_AzimuthOffsetNode           = ScalarVectorValue( g_ComputeScope, eScalarType::FLOAT32, l_ScheduledFlashes.mWorldPosition.mX );
    auto l_SampledAzimuthsNDC          = Subtract( g_ComputeScope, l_SampledAzimuths, l_AzimuthOffsetNode );
    auto l_SampledAzimuthsNDCFlattened = Flatten( g_ComputeScope, l_SampledAzimuthsNDC );

    auto l_ElevationOffsetNode           = ScalarVectorValue( g_ComputeScope, eScalarType::FLOAT32, l_ScheduledFlashes.mWorldPosition.mY );
    auto l_SampledElevationsNDC          = Subtract( g_ComputeScope, l_SampledElevations, l_ElevationOffsetNode );
    auto l_SampledElevationsNDCFlattened = Flatten( g_ComputeScope, l_SampledElevationsNDC );

    auto l_FlashAttenuationTexturesNode = VectorValue( g_ComputeScope, lSamples->GetScheduledFlashes().mDiffusion );
    auto l_DiffuserCoefficient          = Sample2D( g_ComputeScope, l_SampledAzimuthsNDCFlattened, l_SampledElevationsNDCFlattened, l_FlashAttenuationTexturesNode );

    // Create ray intensities
    auto l_RayIntensities = Multiply( g_ComputeScope, l_SampleRayIntensitiesFlattened, l_DiffuserCoefficient );

    // Create test distances
    sConstantValueInitializerComponent l_DistanceInitializer{};
    l_DistanceInitializer.mValue       = g_TargetDistance;
    auto l_SampleRayDistances          = MultiTensorValue( g_ComputeScope, l_DistanceInitializer, l_SampledAzimuths.Get<sMultiTensorComponent>().mValue.Shape() );
    auto l_SampleRayDistancesFlattened = Flatten( g_ComputeScope, l_SampleRayDistances );

    // Collect APD sizes
    auto l_FlashEntities = lSamples->GetScheduledFlashes();

    auto l_APDRepetitionsNode = VectorValue( g_ComputeScope, l_FlashEntities.mPhotoDetectors.mSize );

    auto l_SampleRayAzimuthBlowUp     = Tile( g_ComputeScope, l_SampledAzimuthsNDCFlattened, l_APDRepetitionsNode );
    auto l_SampleRayElevationBlowUp   = Tile( g_ComputeScope, l_SampledElevationsNDCFlattened, l_APDRepetitionsNode );
    auto l_SampleRayIntensitiesBlowUp = Tile( g_ComputeScope, l_RayIntensities, l_APDRepetitionsNode );
    auto l_SampleRayDistancesBlowUp   = Tile( g_ComputeScope, l_SampleRayDistancesFlattened, l_APDRepetitionsNode );

    std::vector<std::vector<uint32_t>> l_PDTensorShape( l_ScheduledFlashCount );

    std::vector<float> l_PDCellPositionsX = {};
    std::vector<float> l_PDCellPositionsY = {};
    std::vector<float> l_PDCellSizesW     = {};
    std::vector<float> l_PDCellSizesH     = {};
    for( uint32_t i = 0; i < l_ScheduledFlashCount; i++ )
    {
        std::vector<math::vec4> l_PDCells = l_ScheduledFlashes.mPhotoDetectors.mCells[i];

        l_PDTensorShape[i] = { static_cast<uint32_t>( l_PDCells.size() ) };

        for( uint32_t j = 0; j < l_PDCells.size(); j++ )
        {
            l_PDCellPositionsX.push_back( l_PDCells[i].x );
            l_PDCellPositionsY.push_back( l_PDCells[i].y );
            l_PDCellSizesW.push_back( l_PDCells[i].z );
            l_PDCellSizesH.push_back( l_PDCells[i].w );
        }
    }

    sDataInitializerComponent l_PDCellPositionXNodeInitializer( l_PDCellPositionsX );
    auto l_PDCellPositionXNode = MultiTensorValue( g_ComputeScope, l_PDCellPositionXNodeInitializer, sTensorShape( l_PDTensorShape, sizeof( float ) ) );

    sDataInitializerComponent l_PDCellPositionYNodeInitializer( l_PDCellPositionsY );
    auto l_PDCellPositionYNode = MultiTensorValue( g_ComputeScope, l_PDCellPositionYNodeInitializer, sTensorShape( l_PDTensorShape, sizeof( float ) ) );

    sDataInitializerComponent l_PDCellSizeWNodeInitializer( l_PDCellSizesW );
    auto l_PDCellSizeWNode = MultiTensorValue( g_ComputeScope, l_PDCellSizeWNodeInitializer, sTensorShape( l_PDTensorShape, sizeof( float ) ) );

    sDataInitializerComponent l_PDCellSizeHNodeInitializer( l_PDCellSizesH );
    auto l_PDCellSizeHNode = MultiTensorValue( g_ComputeScope, l_PDCellSizeHNodeInitializer, sTensorShape( l_PDTensorShape, sizeof( float ) ) );

    std::vector<uint32_t> l_PDCellRep( l_ScheduledFlashCount );
    for( uint32_t i = 0; i < l_ScheduledFlashCount; i++ )
    {
        l_PDCellRep[i] = l_SampledAzimuthsNDCFlattened.Get<sMultiTensorComponent>().mValue.Shape().mShape[i][0];
    }

    auto l_APDDualRepetitionsNode = VectorValue( g_ComputeScope, l_PDCellRep );
    auto l_PDCellPositionNodeXBU  = Repeat( g_ComputeScope, l_PDCellPositionXNode, l_APDDualRepetitionsNode );
    auto l_PDCellPositionNodeYBU  = Repeat( g_ComputeScope, l_PDCellPositionYNode, l_APDDualRepetitionsNode );

    auto x = l_SampleRayAzimuthBlowUp.Get<sMultiTensorComponent>().mValue.Shape();
    auto y = l_PDCellPositionNodeXBU.Get<sMultiTensorComponent>().mValue.Shape();

    auto l_SampledAzimuthWrtCell   = Subtract( g_ComputeScope, l_SampleRayAzimuthBlowUp, l_PDCellPositionNodeXBU );
    auto l_SampledElevationWrtCell = Subtract( g_ComputeScope, l_SampleRayElevationBlowUp, l_PDCellPositionNodeYBU );

    auto l_FlashReductionTexturesNode = VectorValue( g_ComputeScope, lSamples->GetScheduledFlashes().mReduction );
    auto l_ReductionCoefficient       = Sample2D( g_ComputeScope, l_SampledAzimuthWrtCell, l_SampledElevationWrtCell, l_FlashReductionTexturesNode );
    auto l_ReducedIntensities         = Multiply( g_ComputeScope, l_SampleRayIntensitiesBlowUp, l_ReductionCoefficient );

    auto l_Templates        = VectorValue( g_ComputeScope, l_ScheduledFlashes.mLaser.mPulseTemplate );
    auto l_SamplingLength   = VectorValue( g_ComputeScope, l_ScheduledFlashes.mSampling.mLength );
    auto l_SamplingInterval = VectorValue( g_ComputeScope, l_ScheduledFlashes.mSampling.mInterval );
    auto l_TimebaseDelay    = ScalarVectorValue( g_ComputeScope, eScalarType::FLOAT32, l_ScheduledFlashes.mLaser.mTimebaseDelay );

    auto l_SpeedOfLightCoefficient = ConstantScalarValue( g_ComputeScope, 2.0f / SPEED_OF_LIGHT );

    auto l_DetectionTimes  = AffineTransform( g_ComputeScope, l_SpeedOfLightCoefficient, l_SampleRayDistancesBlowUp, l_TimebaseDelay );
    auto l_WaveformBuffer0 = ResolveWaveforms( g_ComputeScope, l_DetectionTimes, l_ReducedIntensities, l_SamplingLength, l_SamplingInterval, l_Templates );

    auto l_WaveformScaling = ConstantScalarValue( g_ComputeScope, 2500.0f );
    auto l_WaveformBuffer  = Multiply( g_ComputeScope, l_WaveformBuffer0, l_WaveformScaling );

    /// Ambiant noise
    sRandomNormalInitializerComponent l_NoiseInitializer{};
    l_NoiseInitializer.mType   = eScalarType::FLOAT32;
    l_NoiseInitializer.mMean   = 0.1f;
    l_NoiseInitializer.mStd    = 0.01f;
    auto l_AmbientNoise        = MultiTensorValue( g_ComputeScope, l_NoiseInitializer, l_WaveformBuffer.Get<sMultiTensorComponent>().mValue.Shape() );
    auto l_NoisyWaveformBuffer = Add( g_ComputeScope, l_WaveformBuffer, l_AmbientNoise );

    auto l_ADCScaling = ConstantScalarValue( g_ComputeScope, 10000.0f );
    auto l_ADCOutput  = ToFixedPoint( g_ComputeScope, eScalarType::UINT32, l_NoisyWaveformBuffer, l_ADCScaling );

    auto now    = std::chrono::system_clock::now();
    auto now_ns = std::chrono::time_point_cast<std::chrono::microseconds>( now );
    auto valu0  = now_ns.time_since_epoch();

    g_ComputeScope.Run( l_NoisyWaveformBuffer );
    g_Waveforms = l_NoisyWaveformBuffer.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 0 );

    now              = std::chrono::system_clock::now();
    now_ns           = std::chrono::time_point_cast<std::chrono::microseconds>( now );
    auto ComputeTime = static_cast<float>( now_ns.time_since_epoch().count() - valu0.count() ) / 1000.0f;

    LTSE::Logging::Info( "{:.3f}", ComputeTime );

    ComputeFpsTimer += ComputeTime;
    ComputeFrameCounter++;
    if( ComputeFpsTimer > 1000.0f )
    {
        ComputeLastFPS      = static_cast<uint32_t>( (float)ComputeFrameCounter * ( 1000.0f / ComputeFpsTimer ) );
        ComputeFpsTimer     = 0.0f;
        ComputeFrameCounter = 0;
    }
}

void Update( Scope &a_Scope, Timestep ts )
{
    frameCounter++;
    fpsTimer += (float)ts;
    if( fpsTimer > 1000.0f )
    {
        lastFPS      = static_cast<uint32_t>( (float)frameCounter * ( 1000.0f / fpsTimer ) );
        fpsTimer     = 0.0f;
        frameCounter = 0;
    }

    a_Scope.Reset();
    RunPipeline( a_Scope );
}

bool RenderUI( ImGuiIO &io )
{
    auto l_WindowSize = UI::GetRootWindowSize();
    bool Quit         = false;
    UI::Pane( "##SCENE_COLLECTION", { 15.0f, 31.0f }, { 350.0f, 350.0f },
              [&]()
              {
                  Text( "{}", g_SensorModel->mSensorDefinition->mName );

                  UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( 0.0f, 10.0f ) );
                  if( lastFPS > 0 )
                      UI::Text( fmt::format( "Render: {} fps ({:.2f} ms)", lastFPS, ( 1000.0f / lastFPS ) ).c_str() );
                  else
                      UI::Text( "Render: 0 fps (0 ms)" );

                  if( lastFPS > 0 )
                      UI::Text( fmt::format( "Compute: {} tile/s ({:.2f} ms)", ComputeLastFPS, ( 1000.0f / ComputeLastFPS ) ).c_str() );
                  else
                      UI::Text( "Compute: 0 tile/s (0 ms)" );

                  UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( 0.0f, 10.0f ) );
                  Text( "Flash time: {:.3f} ms", 0.0f );
                  UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( 0.0f, 10.0f ) );
                  Text( "Target distance:" );
                  UI::SameLine();
                  UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( 00.0f, -5.0f ) );

                  UI::Slider( "##target_distance", "%f", 1.0f, 128.0f, &( g_TargetDistance ) );

                  UI::Button( "QUIT", { 100.0f, 30.0f }, [&]() { Quit = true; } );
              } );

    UI::Pane( "##Waveform", { 15.0f + 15 + 350.0f, 31.0f }, { 1350.0f, 450.0f },
              [&]()
              {
                  if( g_Waveforms.size() == 0 )
                      return;

                  constexpr uint32_t l_WaveformLength = 1024;
                  static float l_WF[l_WaveformLength] = { 0.0f };
                  std::memcpy( l_WF, g_Waveforms.data(), l_WaveformLength * sizeof( float ) );

                  auto l_WindowSize = LTSE::Core::UI::GetAvailableContentSpace();

                  LinePlotConfig l_PlotConfig = { "Waveforms", "Sample", "Amplitude", (float)l_WindowSize.x, (float)l_WindowSize.y, { -10, 1034 }, { -0.25, 1.25 } };
                  std::vector<LinePlotData> l_PlotData( 1 );

                  l_PlotData[0].x = std::vector<float>( l_WaveformLength );
                  l_PlotData[0].y = std::vector<float>( l_WaveformLength );

                  auto l_StartIndex = 0;

                  char x[32];
                  sprintf( x, "wf_%d(x)", 0 );
                  l_PlotData[0].Legend = x;
                  for( int i = 0; i < l_WaveformLength; i++ )
                  {
                      l_PlotData[0].x[i] = (float)i;
                      l_PlotData[0].y[i] = l_WF[i];
                  }
                  LinePlot( l_PlotConfig, l_PlotData );
              } );
    return Quit;
}

int main( int argc, char **argv )
{
    g_EngineLoop = EngineLoop();
    g_EngineLoop.PreInit( 0, nullptr );
    g_EngineLoop.Init();
    Scope g_ComputeScope( l_PoolSize );

    g_EngineLoop.UIDelegate.connect<RenderUI>();
    g_EngineLoop.UpdateDelegate.connect<Update>( g_ComputeScope );

    g_SensorModel                    = std::make_shared<SensorDeviceBase>( static_cast<uint32_t>( l_PoolSize ) );
    g_SensorModel->mSensorDefinition = std::make_shared<SensorModelBase>();

    g_SensorModel->mSensorDefinition->mName = "Pipeline Test";

    g_Tile = g_SensorModel->mSensorDefinition->CreateTile( "0", math::vec2{ 0.0f, 0.0f } );

    fs::path l_Root                    = "C:\\GitLab\\LTSimulationEngine\\Programs\\TestComputeGraph\\Data";
    auto l_AttenuationPatternAssetData = ReadAsset( l_Root, "diffusion/asset.yaml", "" );
    auto l_AttenuationPatternEntity    = g_SensorModel->mSensorDefinition->CreateAsset( "diffusion_0", l_AttenuationPatternAssetData );

    auto l_ReductionAssetData = ReadAsset( l_Root, "reduction/asset.yaml", "" );
    auto l_ReductionEntity    = g_SensorModel->mSensorDefinition->CreateAsset( "reduction_0", l_ReductionAssetData );

    auto l_PulseTemplateAssetData = ReadAsset( l_Root, "pulse_template/asset.yaml", "" );
    auto l_PulseTemplateEntity    = g_SensorModel->mSensorDefinition->CreateAsset( "pulse_template_0", l_PulseTemplateAssetData );

    g_Attenuation = l_AttenuationPatternEntity.Get<sAssetMetadata>().mChildEntities["diffusion_0"];
    g_Reduction   = l_ReductionEntity.Get<sAssetMetadata>().mChildEntities["reduction_0"];

    std::vector<math::vec4> l_CellPositions( 32 );
    std::vector<math::vec4> l_Baseline( 32 );
    std::vector<Entity> l_StaticNoise( 32 );
    for( uint32_t i = 0; i < l_CellPositions.size(); i++ )
    {
        l_CellPositions[i] = math::vec4{ 0.0f, i * 0.2 - ( 32.0f * 0.2f ) / 2.0f, 0.2f, 0.2f };
        l_Baseline[i]      = math::vec4{ 0.0f };
        l_StaticNoise[i]   = Entity{};
    }
    sPhotoDetector lPhotoDetectorComponent{};
    lPhotoDetectorComponent.mCellPositions = l_CellPositions;
    lPhotoDetectorComponent.mStaticNoise   = l_StaticNoise;
    lPhotoDetectorComponent.mBaseline      = l_Baseline;
    auto l_PhotoDetector                   = g_SensorModel->mSensorDefinition->CreateElement( "PD", "apd_0", lPhotoDetectorComponent );

    sLaserAssembly lLaserComponent{};
    lLaserComponent.mPulseTemplate = l_PulseTemplateEntity.Get<sAssetMetadata>().mChildEntities["~"];
    lLaserComponent.mTimebaseDelay = math::vec4{ 150.10384283916842f, 0.0f, 0.0f, 0.0f };
    lLaserComponent.mFlashTime     = math::vec4{ 0.1f, 0.0f, 0.0f, 0.0f };
    auto l_LaserComponent          = g_SensorModel->mSensorDefinition->CreateElement( "Laser", "laser_0", lLaserComponent );

    sSampler lSamplerComponent{};
    lSamplerComponent.mLength    = 1024;
    lSamplerComponent.mFrequency = 800000000.0f;
    auto l_Sampler               = g_SensorModel->mSensorDefinition->CreateElement( "Sampler", "sampler_0", lSamplerComponent );

    g_Tile.Adjoin<sSampler>( l_Sampler );

    math::vec2 l_FlashInnerExtent      = math::vec2{ 0.1, 4.0f };
    math::vec2 l_FlashRelativePosition = math::vec2{ 0.0f, 0.0f };

    for( uint32_t l_FlashID = 0; l_FlashID < 32; l_FlashID++ )
    {
        auto l_Flash = g_SensorModel->mSensorDefinition->CreateFlash( g_Tile, fmt::format( "{}", l_FlashID ), l_FlashRelativePosition, l_FlashInnerExtent );
        l_Flash.Adjoin<sDiffusionPattern>( g_Attenuation );
        l_Flash.Adjoin<sReductionPattern>( g_Reduction );
        l_Flash.Adjoin<sPhotoDetector>( l_PhotoDetector );
        l_Flash.Adjoin<sLaserAssembly>( l_LaserComponent );
    }

    while( g_EngineLoop.Tick() )
        ;

    return 0;
}
