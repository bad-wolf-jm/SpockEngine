
#include "Developer/GraphicContext/UI/UIContext.h"
#include "Developer/Platform/EngineLoop.h"

#include <fmt/core.h>
#include <fstream>

#include "Core/EntityRegistry/ScriptableEntity.h"
#include "Core/Logging.h"

#include "Developer/UI/UI.h"
#include "Developer/UI/Widgets.h"

#include "Developer/Editor/BaseEditorApplication.h"
#include "Developer/Editor/EditorWindow.h"

// #include "LidarSensorModel/ModelArchive.h"
// #include "LidarSensorModel/ModelBuilder.h"
// #include "LidarSensorModel/SensorModelBase.h"

// // clang-format off
// #include "Sensor/Cyclops.h"
// #include "Sensor/Controllers/BasicController.h"
// #include "Sensor/Controllers/LeddarEngine.h"
// #include "Sensor/WfShortMockData.h"
// // clang-format on

// #include "ProcessingNodes/HitRecordProcessor.h"
// #include "ProcessingNodes/WaveformGenerator.h"
// #include "ProcessingNodes/WaveformType.h"

// #include "Serialize/FileIO.h"
// #include "Serialize/SensorAsset.h"
// #include "Serialize/SensorDefinition.h"

using namespace LTSE::Core;
using namespace LTSE::Graphics;
using namespace LTSE::Editor;
using namespace LTSE::Core::UI;
using namespace LTSE::Core::UI;
using namespace LTSE::SensorModel;

// class TileLayoutCombo
// {
//   public:
//     Ref<SensorDeviceBase> SensorModel = nullptr;
//     std::string ID                    = "";
//     UI::ComboBox<Entity> Dropdown;

//   public:
//     TileLayoutCombo() = default;
//     TileLayoutCombo( std::string a_ID )
//         : ID{ a_ID }
//         , Dropdown{ UI::ComboBox<Entity>( a_ID ) } {};

//     ~TileLayoutCombo() = default;

//     Entity GetValue();

//     void Display( Entity &a_TargetEntity )
//     {
//         Dropdown.Labels = { "None" };
//         Dropdown.Values = { Entity{} };

//         uint32_t n = 1;
//         for( auto lLayout : SensorModel->mSensorDefinition->mRootLayout.Get<sRelationshipComponent>().mChildren )
//         {
//             Dropdown.Labels.push_back( lLayout.Get<sTag>().mValue );
//             Dropdown.Values.push_back( lLayout );

//             if( (uint32_t)lLayout == (uint32_t)a_TargetEntity )
//                 Dropdown.CurrentItem = n;
//             n++;
//         }

//         Dropdown.Display();

//         if( Dropdown.Changed )
//         {
//             a_TargetEntity = Dropdown.GetValue();
//         }
//     }
// };

// enum class eControllerID : uint8_t
// {
//     NONE,
//     BASIC,
//     LEDDAR_ENGINE
// };

// class ControllerChooser
// {
//   public:
//     Ref<SensorDeviceBase> SensorModel = nullptr;
//     std::string ID                    = "";
//     UI::ComboBox<eControllerID> Dropdown;

//   public:
//     ControllerChooser() = default;
//     ControllerChooser( std::string a_ID )
//         : ID{ a_ID }
//         , Dropdown{ UI::ComboBox<eControllerID>( a_ID ) } {};

//     ~ControllerChooser() = default;

//     Ref<SensorControllerBase> GetValue();

//     void Display( eControllerID &Current )
//     {
//         auto it = std::find( Dropdown.Values.begin(), Dropdown.Values.end(), Current );
//         if( it != Dropdown.Values.end() )
//             Dropdown.CurrentItem = std::distance( Dropdown.Values.begin(), it );

//         Dropdown.Display();
//     }
// };

// struct SensorControllerBehaviour : sBehaviourController
// {
//     Ref<SensorControllerBase> SensorController = nullptr;
//     Ref<Scope> m_ComputeScope                  = nullptr;
//     Ref<WorldSampler> m_WorldSampler           = nullptr;
//     Ref<EngineLoop> mEngineLoop                = nullptr;
//     Ref<Scene> m_World                         = nullptr;
//     Ref<SensorDeviceBase> m_Sensor             = nullptr;

//     PointCloudVisualizer m_PointCloudVisualizer{};

//     SensorControllerBehaviour( Ref<EngineLoop> aEngineLoop, Ref<Scene> aWorld, Ref<SensorControllerBase> aSensorController, Ref<SensorDeviceBase> aSensor )
//         : mEngineLoop{ aEngineLoop }
//         , m_World{ aWorld }
//         , SensorController{ aSensorController }
//         , m_Sensor{ aSensor }
//     {
//     }

//     void OnCreate()
//     {
//         m_ComputeScope = New<Scope>( 512 * 1024 * 1024 );
//         m_WorldSampler = New<WorldSampler>( m_World->GetRayTracingContext() );

//         if( SensorController )
//         {
//             SensorController->Connect( m_Sensor );
//             SensorController->PowerOn();
//         }
//     }

//     void OnDestroy()
//     {
//         if( SensorController )
//         {
//             SensorController->Shutdown();
//             SensorController->Disconnect();
//         }
//     }

//     void OnUpdate( Timestep ts )
//     {
//         if( !SensorController )
//             return;

//         Ref<EnvironmentSampler> lEnvSamples = SensorController->Emit( ts );
//         if( !lEnvSamples )
//             return;

//         m_ComputeScope->Reset();
//         MultiTensor &lAzimuths    = ( *lEnvSamples )["Azimuth"].Get<sMultiTensorComponent>().mValue;
//         MultiTensor &lElevations  = ( *lEnvSamples )["Elevation"].Get<sMultiTensorComponent>().mValue;
//         MultiTensor &lIntensities = ( *lEnvSamples )["Intensity"].Get<sMultiTensorComponent>().mValue;

//         sTensorShape lOutputShape( lIntensities.Shape().mShape, sizeof( HitRecord ) );
//         MultiTensor lHitRecords = MultiTensor( m_ComputeScope->mPool, lOutputShape );

//         if( Has<TransformMatrixComponent>() )
//         {
//             auto &lParticles = Get<ParticleSystemComponent>();

//             m_WorldSampler->Sample( Get<TransformMatrixComponent>().Matrix, m_World, lAzimuths, lElevations, lIntensities, lHitRecords );

//             if( !( lParticles.Particles ) || lParticles.ParticleCount != lAzimuths.SizeAs<float>() )
//             {
//                 lParticles.ParticleCount = lAzimuths.SizeAs<float>();
//                 lParticles.Particles =
//                     New<Buffer>( mEngineLoop->GetGraphicContext(), eBufferBindType::VERTEX_BUFFER, false, true, true, true, lParticles.ParticleCount * sizeof( Particle ) );
//             }

//             GPUExternalMemory l_PointCloudMappedBuffer( *( lParticles.Particles ), lParticles.ParticleCount * sizeof( Particle ) );
//             m_PointCloudVisualizer.InvertZAxis = false;
//             m_PointCloudVisualizer.Resolution  = 0.2;
//             m_PointCloudVisualizer.Visualize( Get<TransformMatrixComponent>().Matrix, lHitRecords, l_PointCloudMappedBuffer );
//             l_PointCloudMappedBuffer.Dispose();
//         }
//         auto &lDistancesNode   = RetrieveDistance( *( SensorController->ControlledSensor()->mComputationScope ), lHitRecords );
//         auto &lIntensitiesNode = RetrieveIntensities( *( SensorController->ControlledSensor()->mComputationScope ), lHitRecords );

//         SensorController->Receive( ts, *( SensorController->ControlledSensor()->mComputationScope ), lEnvSamples->GetScheduledFlashes(), ( *lEnvSamples )["Azimuth"],
//                                    ( *lEnvSamples )["Elevation"], lIntensitiesNode, lDistancesNode );
//     }
// };

class EchoDSMVPEditor : public BaseEditorApplication
{
  public:
    EchoDSMVPEditor()
        : BaseEditorApplication()
    {
        ApplicationName = "EchoDS_MVP";
        WindowSize      = { 2920, 1580 };
        // m_ComputeScope  = New<Scope>( 512 * 1024 * 1024 );
    };

    ~EchoDSMVPEditor() = default;

    void LoadSensorConfiguration() {}
    void SaveSensorConfiguration() {}

    std::vector<float> mWaveforms = {};

    float g_TargetDistance     = 2.0f;
    float g_Intensity          = 1.0f;
    math::vec4 g_TimebaseDelay = math::vec4{ 0.0f, 0.0f, 0.0f, 0.0f };
    uint32_t g_Basepoints      = 100;
    uint32_t g_Oversampling    = 1;
    uint32_t g_Accumulation    = 1;
    float g_SystemTemperature  = 25.0f;
    float g_StaticNoiseFactor  = .001f;
    // std::vector<sWaveformPacket> mShortWaveformBuffer;

    // sWaveformPacket mShortWaveformToDisplay{};

    // void Sparkline( const char *id, const float *values, int count, float min_v, float max_v, int offset, const ImVec4 &col, const ImVec2 &size )
    // {
    //     ImPlot::PushStyleVar( ImPlotStyleVar_PlotPadding, ImVec2( 0, 0 ) );
    //     if( ImPlot::BeginPlot( id, size, ImPlotFlags_CanvasOnly | ImPlotFlags_NoChild ) )
    //     {
    //         ImPlot::SetupAxes( 0, 0, ImPlotAxisFlags_NoDecorations, ImPlotAxisFlags_NoDecorations );
    //         ImPlot::SetupAxesLimits( 0, count - 1, min_v, max_v, ImGuiCond_Always );
    //         ImPlot::PushStyleColor( ImPlotCol_Line, col );
    //         ImPlot::PlotLine( id, values, count, 1, 0, offset );
    //         ImPlot::PushStyleVar( ImPlotStyleVar_FillAlpha, 0.25f );
    //         ImPlot::PlotShaded( id, values, count, 0, 1, 0, offset );
    //         ImPlot::PopStyleVar();
    //         ImPlot::PopStyleColor();
    //         ImPlot::EndPlot();
    //     }
    //     ImPlot::PopStyleVar();
    // }

    // std::vector<float> mSampledAzimuths;
    // sTensorShape mSampledAzimuthsShape;

    // std::vector<float> mSampledElevations;
    // sTensorShape mSampledElevationsShape;

    // std::vector<float> mSampledIntensities;
    // sTensorShape mSampledIntensitiesShape;

    // std::vector<std::vector<float>> mAveragedSampledIntensities;
    // std::vector<std::vector<uint32_t>> mSampledIntensitiesDisplayShape;

    // void OnBeginScenario()
    // {
    //     switch( CurrentControllerID )
    //     {
    //     case eControllerID::BASIC:
    //         CurrentController = New<OptixSensorLidar::BasicSensorController>();
    //         break;
    //     case eControllerID::LEDDAR_ENGINE:
    //         CurrentController = New<OptixSensorLidar::LESensorController>( "192.168.204.130", 27100 );
    //         break;
    //     case eControllerID::NONE:
    //     default:
    //         break;
    //     }

    //     mEditorWindow.Sensor.Get<sBehaviourComponent>().Bind<SensorControllerBehaviour>( mEngineLoop, m_World, CurrentController, m_Sensor );
    // }

    void OnUI()
    {
#if 0
        static bool p_open0 = true;
        if( ImGui::Begin( "ECHODS_MVP LE CONNECTION", &p_open0, ImGuiWindowFlags_None ) )
        {
        }
        ImGui::End();
#endif
        // static bool p_open2 = true;
        // if( ImGui::Begin( "SHORT WAVEWFORMS", &p_open2, ImGuiWindowFlags_None ) )
        // {

        //     ImVec2 l_PopupSize = ImGui::GetWindowSize();

        //     ImGuiTableFlags flags = ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY | ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV |
        //                             ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable;
        //     ImGui::PushStyleVar( ImGuiStyleVar_CellPadding, ImVec2( 5, 5 ) );

        //     if( UI::Button( "Generate", { 150.0f, 50.0f } ) )
        //     {
        //         EnvironmentSampler::sCreateInfo lEnvironmentSamplingParameter{};
        //         AcquisitionSpecification lAcqCreateInfo{};
        //         lAcqCreateInfo.mBasePoints   = 100;
        //         lAcqCreateInfo.mOversampling = 1;

        //         auto lSamples = m_Sensor->Sample( lEnvironmentSamplingParameter, lAcqCreateInfo, std::string( "tile_1" ), math::vec2{ 0.0f, 0.0f }, 0.0f );

        //         auto lDistance = MultiTensorValue( *m_Sensor->mComputationScope, sConstantValueInitializerComponent( 3.0f ),
        //                                            ( *lSamples )["Azimuth"].Get<sMultiTensorComponent>().mValue.Shape() );

        //         m_Sensor->Process( 0.0f, *m_Sensor->mComputationScope, lSamples->GetScheduledFlashes(), ( *lSamples )["Azimuth"], ( *lSamples )["Elevation"],
        //                            ( *lSamples )["Intensity"], lDistance );

        //         m_Sensor->mComputationScope->Run( ( *m_Sensor->mComputationScope )["FPGAOutput"] );
        //         mShortWaveformBuffer = ( *m_Sensor->mComputationScope )["FPGAOutput"].Get<sMultiTensorComponent>().mValue.FetchFlattened<sWaveformPacket>();
        //     }

        //     ImGui::Columns( 2 );

        //     if( mShortWaveformBuffer.size() == 1024 )
        //     {
        //         static const char *xlabels[] = { "1",  "2",  "3",  "4",  "5",  "6",  "7",  "8",  "9",  "10", "11", "12", "13", "14", "15", "16",
        //                                          "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32" };
        //         static const char *ylabels[] = { "1",  "2",  "3",  "4",  "5",  "6",  "7",  "8",  "9",  "10", "11", "12", "13", "14", "15", "16",
        //                                          "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32" };

        //         std::array<float, 32 * 32> lShortWaveformAmplitudes;
        //         uint32_t lSwfIdx          = 0;
        //         float lGlobalMaxAmplitude = 0;
        //         float lGlobalMinAmplitude = std::numeric_limits<float>::max();
        //         for( uint32_t lSwfIdx = 0; lSwfIdx < 32 * 32; lSwfIdx++ )
        //         {
        //             auto lSwf = mShortWaveformBuffer[lSwfIdx];
        //             uint32_t aID, aVersion, aDetectionCount, aMaxDetectionCount, aSampleCount;
        //             lSwf.mHeader.UnpackHeader( aID, aVersion, aDetectionCount, aMaxDetectionCount, aSampleCount );

        //             float lMaxAmplitude = 0;
        //             float lMinAmplitude = std::numeric_limits<float>::max();
        //             for( uint32_t i = 0; i < aDetectionCount; i++ )
        //             {
        //                 lMaxAmplitude = std::max( lMaxAmplitude, static_cast<float>( lSwf.mWaveform[i].mPulse.mAmplitude ) );
        //                 lMinAmplitude = std::min( lMinAmplitude, static_cast<float>( lSwf.mWaveform[i].mPulse.mAmplitude ) );
        //             }

        //             auto lRow    = lSwfIdx % 32;
        //             auto lColumn = lSwfIdx / 32;

        //             lShortWaveformAmplitudes[( 31 - lRow ) * 32 + lColumn] = lMaxAmplitude;

        //             lGlobalMaxAmplitude = std::max( lGlobalMaxAmplitude, lMaxAmplitude );
        //             lGlobalMinAmplitude = std::min( lGlobalMinAmplitude, lMinAmplitude );
        //         }

        //         ImPlot::PushColormap( ImPlotColormap_Jet );

        //         if( ImPlot::BeginPlot( "##SWF_AMPLITUDES", ImVec2( 512, 512 ), ImPlotFlags_Crosshairs | ImPlotFlags_NoFrame | ImPlotFlags_NoChild ) )
        //         {
        //             ImPlot::SetupAxes( NULL, NULL, ImPlotAxisFlags_Lock, ImPlotAxisFlags_Lock );
        //             ImPlot::SetupAxisTicks( ImAxis_X1, 0.0f + 1.0f / 64.0f, 1.0f - 1.0f / 64.0f, 32, xlabels );
        //             ImPlot::SetupAxisTicks( ImAxis_Y1, 0.0f + 1.0f / 64.0f, 1.0f - 1.0f / 64.0f, 32, ylabels );
        //             ImPlot::PlotHeatmap( "##SWF_AMPLITUDES_T", lShortWaveformAmplitudes.data(), 32, 32, lGlobalMinAmplitude, lGlobalMaxAmplitude, NULL );

        //             if( ImPlot::IsPlotHovered() && ImGui::IsMouseClicked( 0 ) && ImGui::GetIO().KeyCtrl )
        //             {
        //                 ImPlotPoint pt   = ImPlot::GetPlotMousePos();
        //                 uint32_t lSeg    = static_cast<uint32_t>( pt.y * 32 );
        //                 uint32_t lScan   = static_cast<uint32_t>( pt.x * 32 );
        //                 uint32_t lSwfIdx = lScan * 32 + lSeg;

        //                 mShortWaveformToDisplay = mShortWaveformBuffer[lSwfIdx];
        //             }
        //             ImGui::SameLine();
        //             ImPlot::ColormapScale( "##HeatScale", lGlobalMinAmplitude, lGlobalMaxAmplitude, ImVec2( 60, 500 ) );

        //             ImPlot::EndPlot();
        //         }
        //     }

        //     ImGui::NextColumn();

        //     float lMaxLabelSize       = 0;
        //     float lTextHeight         = 0;
        //     float lLabelSizes[8]      = { 0.0f };
        //     const char *lFieldNames[] = {
        //         "Interpolated distance:", "Amplitude:", "Baseline:", "Max index:", "Last unsaturated sample:", "Baseline after saturation:", "Short trace offset:", "Samples:" };

        //     for( uint32_t i = 0; i < 8; i++ )
        //     {
        //         auto lTextSize = ImGui::CalcTextSize( lFieldNames[i] );
        //         lLabelSizes[i] = lTextSize.x;
        //         lMaxLabelSize  = std::max( lMaxLabelSize, lTextSize.x );
        //         lTextHeight    = std::max( lTextHeight, lTextSize.y );
        //     }

        //     for( uint32_t i = 0; i < 5; i++ )
        //     {
        //         auto lPosition0 = UI::GetCurrentCursorPosition();
        //         Text( "Interpolated distance:" );
        //         UI::SameLine();
        //         UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2{ ( lMaxLabelSize - lLabelSizes[0] + 10.0f ), 0.0f } );
        //         UI::Text( "{}", mShortWaveformToDisplay.mWaveform[i].mPulse.mInterpolatedDistance );

        //         Text( "Amplitude:" );
        //         UI::SameLine();
        //         UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2{ ( lMaxLabelSize - lLabelSizes[1] + 10.0f ), 0.0f } );
        //         UI::Text( "{}", mShortWaveformToDisplay.mWaveform[i].mPulse.mAmplitude );

        //         Text( "Baseline:" );
        //         UI::SameLine();
        //         UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2{ ( lMaxLabelSize - lLabelSizes[2] + 10.0f ), 0.0f } );
        //         UI::Text( "{}", mShortWaveformToDisplay.mWaveform[i].mPulse.mPulseBaseLevel );

        //         Text( "Max index:" );
        //         UI::SameLine();
        //         UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2{ ( lMaxLabelSize - lLabelSizes[3] + 10.0f ), 0.0f } );
        //         UI::Text( "{}", mShortWaveformToDisplay.mWaveform[i].mPulse.mMaxIndex );

        //         Text( "Last unsaturated sample:" );
        //         UI::SameLine();
        //         UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2{ ( lMaxLabelSize - lLabelSizes[4] + 10.0f ), 0.0f } );
        //         UI::Text( "{}", mShortWaveformToDisplay.mWaveform[i].mPulse.mLastUnsaturatedSample );

        //         Text( "Baseline after saturation:" );
        //         UI::SameLine();
        //         UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2{ ( lMaxLabelSize - lLabelSizes[5] + 10.0f ), 0.0f } );
        //         UI::Text( "{}", mShortWaveformToDisplay.mWaveform[i].mPulse.mBaselineAfterSaturation );

        //         Text( "Short trace offset:" );
        //         UI::SameLine();
        //         UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2{ ( lMaxLabelSize - lLabelSizes[6] + 10.0f ), 0.0f } );
        //         UI::Text( "{}", mShortWaveformToDisplay.mWaveform[i].mPulse.mOffset );

        //         auto lPosition1  = UI::GetCurrentCursorPosition();
        //         float lTagHeight = lPosition1.y - lPosition0.y;

        //         UI::SetCursorPosition( lPosition0 + math::vec2{ lMaxLabelSize + 75.0f, 0.0f } );
        //         {
        //             float lValues0[11];
        //             float lValues1[11];
        //             float lMinValue = 0;
        //             float lMaxValue = 100;
        //             for( uint32_t j = 0; j < 11; j++ )
        //             {
        //                 lValues0[j] = static_cast<float>( mShortWaveformToDisplay.mWaveform[i].mRawTrace[j] );
        //                 lMinValue   = std::min( lMinValue, lValues0[j] );
        //                 lMaxValue   = std::max( lMaxValue, lValues0[j] );

        //                 lValues1[j] = static_cast<float>( mShortWaveformToDisplay.mWaveform[i].mProcessedTrace[j] );
        //                 lMinValue   = std::min( lMinValue, lValues1[j] );
        //                 lMaxValue   = std::max( lMaxValue, lValues1[j] );
        //             }
        //             lMaxValue += ( lMaxValue - lMinValue ) * 0.05f;
        //             lMinValue -= ( lMaxValue - lMinValue ) * 0.05f;

        //             ImGui::PushID( i );

        //             ImPlot::PushStyleVar( ImPlotStyleVar_PlotPadding, ImVec2( 0, 0 ) );
        //             if( ImPlot::BeginPlot( "##spark_0", ImVec2( lTagHeight * 2, lTagHeight ), ImPlotFlags_Crosshairs | ImPlotFlags_NoChild ) )
        //             {
        //                 ImPlot::SetupAxes( 0, 0, ImPlotAxisFlags_NoDecorations, ImPlotAxisFlags_None );
        //                 ImPlot::SetupAxesLimits( 0, 11 - 1, lMinValue, lMaxValue, ImGuiCond_Always );
        //                 ImPlot::PushStyleColor( ImPlotCol_Line, ImPlot::GetColormapColor( 0 ) );
        //                 ImPlot::PlotLine( "##spark_0", lValues0, 11, 1, 0, 0 );
        //                 ImPlot::PushStyleVar( ImPlotStyleVar_FillAlpha, 0.25f );
        //                 ImPlot::PlotShaded( "##spark_0", lValues0, 11, 0, 1, 0, 0 );
        //                 ImPlot::PopStyleVar();
        //                 ImPlot::PopStyleColor();

        //                 ImPlot::PushStyleColor( ImPlotCol_Line, ImPlot::GetColormapColor( 1 ) );
        //                 ImPlot::PlotLine( "##spark_1", lValues1, 11, 1, 0, 0 );
        //                 ImPlot::PushStyleVar( ImPlotStyleVar_FillAlpha, 0.25f );
        //                 ImPlot::PlotShaded( "##spark_1", lValues1, 11, 0, 1, 0, 0 );
        //                 ImPlot::PopStyleVar();
        //                 ImPlot::PopStyleColor();

        //                 ImPlot::EndPlot();
        //             }
        //             ImPlot::PopStyleVar();
        //             ImGui::PopID();
        //         }

        //         if( i != 4 )
        //             ImGui::SetCursorPos( ImGui::GetCursorPos() + ImVec2{ 0.0f, 10.0f } );
        //     }
        //     ImGui::PopStyleVar();
        //     ImGui::Columns();
        // }
        // ImGui::End();

#if 0
        static bool p_open4 = true;
        if( ImGui::Begin( "STATIC NOISE", &p_open4, ImGuiWindowFlags_None ) )
        {
            ImDrawList *draw_list  = ImGui::GetWindowDrawList();
            ImVec2 lWindowPosition = ImGui::GetCursorScreenPos();
            ImVec2 lSize{ 100.0f, 50.0f };

            // Draw APD
            for( uint32_t i = 0; i < 32; i++ )
            {
                ImVec2 lTopLeft{ 0.0f, i * ( lSize.y + 5.0f ) };
                ImVec2 lBottomRight = lTopLeft + lSize;

                UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( 15.0f, 0.0f ) );
                Text( "ID: {}", 0 );
                UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( 15.0f, 0.0f ) );
                Text( "X: {:.3f}", 0.f );
                UI::SameLine();
                Text( "Y: {:.3f}", 0.f );
                UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( 0.0f, 10.0f ) );
            }
        }
        ImGui::End();
#endif
        // static bool p_open3 = true;
        // if( ImGui::Begin( "ELECTRONIC XTALK", &p_open3, ImGuiWindowFlags_None ) )
        // {
        //     static uint32_t lAggressorPosition = 3;

        //     if( UI::Button( "Generate", { 150.0f, 50.0f } ) )
        //     {
        //         EnvironmentSampler::sCreateInfo lEnvironmentSamplingParameter{};
        //         AcquisitionSpecification lAcqCreateInfo{};
        //         lAcqCreateInfo.mBasePoints   = g_Basepoints;
        //         lAcqCreateInfo.mOversampling = g_Oversampling;

        //         auto lSamples = m_Sensor->Sample( lEnvironmentSamplingParameter, lAcqCreateInfo, std::string( "tile_1" ), math::vec2{ 0.0f, 0.0f }, 0.0f );

        //         // PhotoDetector 15 gets an aggressor
        //         auto lPhotodetectorData = lSamples->GetScheduledFlashes().mPulseSampling.mPhotoDetectorData;

        //         auto lPhotoDetectorCellWorldElevationMin = ConstantScalarValue( *m_Sensor->mComputationScope,
        //         lPhotodetectorData.mCellWorldElevationBounds.mMin[lAggressorPosition] ); auto lPhotoDetectorCellWorldElevationMax = ConstantScalarValue(
        //         *m_Sensor->mComputationScope, lPhotodetectorData.mCellWorldElevationBounds.mMax[lAggressorPosition] ); auto lHorizontallyAlignedDetections =
        //             InInterval( *m_Sensor->mComputationScope, ( *lSamples )["Elevation"], lPhotoDetectorCellWorldElevationMin, lPhotoDetectorCellWorldElevationMax, false, false
        //             );
        //         auto lZero = ConstantScalarValue( *m_Sensor->mComputationScope, 0.0f );
        //         auto lOne  = ConstantScalarValue( *m_Sensor->mComputationScope, 1.0f );

        //         auto lIntensities = Where( *m_Sensor->mComputationScope, lHorizontallyAlignedDetections, lOne, lZero );

        //         // Put a target 25 meters from sensor
        //         auto lDistance = MultiTensorValue( *m_Sensor->mComputationScope, sConstantValueInitializerComponent( 25.0f ),
        //                                            ( *lSamples )["Azimuth"].Get<sMultiTensorComponent>().mValue.Shape() );

        //         m_Sensor->Process( 0.0f, *m_Sensor->mComputationScope, lSamples->GetScheduledFlashes(), ( *lSamples )["Azimuth"], ( *lSamples )["Elevation"], lIntensities,
        //                            lDistance );

        //         m_Sensor->mComputationScope->Run( ( *m_Sensor->mComputationScope )["FPGAOutput"] );
        //         mWaveforms = ( *m_Sensor->mComputationScope )["SaturatedWaveforms"].Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
        //     }

        //     if( mWaveforms.size() != 0 )
        //     {
        //         uint32_t l_WaveformLength = g_Basepoints * g_Oversampling * 32;
        //         std::vector<float> l_X( l_WaveformLength );
        //         for( int i = 0; i < l_WaveformLength; i++ )
        //             l_X[i] = (float)i;
        //         std::vector<float> l_PDMarkers( 32 );
        //         for( int i = 0; i < 32; i++ )
        //             l_PDMarkers[i] = (float)i * g_Basepoints * g_Oversampling;

        //         auto l_WindowSize = LTSE::Core::UI::GetAvailableContentSpace();
        //         static double vals[] = {0.25, 0.5, 0.75};

        //         static ImPlotAxisFlags flags = ImPlotAxisFlags_NoTickLabels;
        //         ImPlot::SetNextAxisLimits( ImAxis_X1, -1, l_WaveformLength + 1, ImPlotCond_Once );
        //         ImPlot::SetNextAxisLimits( ImAxis_Y1, -1.0001f, 1.009f, ImPlotCond_Once );
        //         if( ImPlot::BeginPlot( "Waveforms", "Sample", "Amplitude", ImVec2{ (float)l_WindowSize.x, 512.0f }, ImPlotFlags_NoFrame | ImPlotFlags_NoChild ) )
        //         {
        //             ImPlot::SetupAxes( NULL, NULL, flags, flags );
        //             ImPlot::PlotLine( "##Line", l_X.data(), mWaveforms.data(), l_WaveformLength );
        //             ImPlot::PlotVLines("PD Boundaries",l_PDMarkers.data(),l_PDMarkers.size());
        //             ImPlot::EndPlot();
        //         }
        //     }
        // }
        // ImGui::End();

        // static bool p_open7 = true;
        // if( ImGui::Begin( "ENVIRONMENT SAMPLING DATA", &p_open7, ImGuiWindowFlags_None ) )
        // {
        //     if( UI::Button( "Generate", { 150.0f, 50.0f } ) )
        //     {
        //         EnvironmentSampler::sCreateInfo lEnvironmentSamplingParameter{};
        //         lEnvironmentSamplingParameter.mMultiSamplingFactor = 5;
        //         AcquisitionSpecification lAcqCreateInfo{};
        //         lAcqCreateInfo.mBasePoints   = 100;
        //         lAcqCreateInfo.mOversampling = 1;

        //         auto lSamples = m_Sensor->Sample( lEnvironmentSamplingParameter, lAcqCreateInfo, std::string( "tile_1" ), math::vec2{ 0.0f, 0.0f }, 0.0f );

        //         auto lDistance = MultiTensorValue( *m_Sensor->mComputationScope, sConstantValueInitializerComponent( 3.0f ),
        //                                            ( *lSamples )["Azimuth"].Get<sMultiTensorComponent>().mValue.Shape() );

        //         m_Sensor->Process( 0.0f, *m_Sensor->mComputationScope, lSamples->GetScheduledFlashes(), ( *lSamples )["Azimuth"], ( *lSamples )["Elevation"],
        //                            ( *lSamples )["Intensity"], lDistance );

        //         m_Sensor->mComputationScope->Run( ( *m_Sensor->mComputationScope )["FPGAOutput"] );

        //         mSampledAzimuths    = ( *m_Sensor->mComputationScope )["Azimuth"].Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
        //         mSampledElevations  = ( *m_Sensor->mComputationScope )["Elevation"].Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
        //         mSampledIntensities = ( *m_Sensor->mComputationScope )["Intensity"].Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();

        //         mAveragedSampledIntensities.resize( 0 );
        //         mSampledIntensitiesDisplayShape.resize( 0 );
        //         for( uint32_t lLayerID = 0; lLayerID < lSamples->mSamplingShape.CountLayers(); lLayerID++ )
        //         {
        //             auto &lLayerDimension = lSamples->mSamplingShape.mShape[lLayerID];
        //             mSampledIntensitiesDisplayShape.push_back( { lLayerDimension[0], lLayerDimension[1] } );
        //             std::vector<float> lAveragedValues( lLayerDimension[0] * lLayerDimension[1] );

        //             auto lLayerInfo    = lSamples->mSamplingShape.GetBufferSize( lLayerID );
        //             float *lDataBuffer = mSampledIntensities.data() + ( lLayerInfo.mOffset / sizeof( float ) );

        //             if( lSamples->mSamplingShape.mRank == 2 )
        //             {
        //                 memcpy( lAveragedValues.data(), lDataBuffer, lLayerInfo.mSize );
        //             }
        //             else
        //             {

        //                 for( uint32_t i = 0; i < lLayerDimension[0]; i++ )
        //                 {
        //                     for( uint32_t j = 0; j < lLayerDimension[1]; j++ )
        //                     {
        //                         float lAverage = 0.0f;
        //                         for( uint32_t k = 0; k < lLayerDimension[2]; k++ )
        //                         {
        //                             lAverage += lDataBuffer[i * lLayerDimension[1] * lLayerDimension[2] + j * lLayerDimension[2] + k];
        //                         }
        //                         lAveragedValues[i * lLayerDimension[1] + j] = lAverage / static_cast<float>( lLayerDimension[2] );
        //                     }
        //                 }
        //             }
        //             mAveragedSampledIntensities.push_back( lAveragedValues );
        //         }
        //     }

        //     for( uint32_t i = 0; i < mSampledIntensitiesDisplayShape.size(); i++ )
        //     {
        //         ImGui::PushID( i );
        //         ImPlot::PushColormap( ImPlotColormap_Jet );
        //         if( ImPlot::BeginPlot( "##SCATTER", ImVec2( 50, 450 ), ImPlotFlags_CanvasOnly | ImPlotFlags_NoInputs | ImPlotFlags_NoFrame | ImPlotFlags_NoChild ) )
        //         {
        //             ImPlot::SetupAxes( NULL, NULL, ImPlotAxisFlags_NoDecorations, ImPlotAxisFlags_NoDecorations );
        //             ImPlot::PlotHeatmap( "##T", mAveragedSampledIntensities[i].data(), mSampledIntensitiesDisplayShape[i][0], mSampledIntensitiesDisplayShape[i][1], 0.0f,
        //                                  0.000001f, NULL );
        //             ImPlot::EndPlot();
        //         }
        //         ImGui::PopID();
        //         if( i != mSampledIntensitiesDisplayShape.size() - 1 )
        //             ImGui::SameLine();
        //     }
        // }
        // ImGui::End();
        static bool p_open1 = true;

        if( ImGui::Begin( "IMPLOT DEMO", &p_open1, ImGuiWindowFlags_None ) )
        {
            ImPlot::ShowDemoWindow();
        }
        ImGui::End();

        // if( ImGui::Begin( "WAVEFORMS", &p_open1, ImGuiWindowFlags_None ) )
        // {
        //     ImDrawList *draw_list = ImGui::GetWindowDrawList();
        //     auto l_WindowSize     = UI::GetAvailableContentSpace();

        //     float lMaxLabelSize       = 0;
        //     float lTextHeight         = 0;
        //     float lLabelSizes[8]      = { 0.0f };
        //     const char *lFieldNames[] = { "Target distance:", "Intensity:", "Base points:", "Oversampling:", "Accumulation:", "System temperature:" };

        //     for( uint32_t i = 0; i < 6; i++ )
        //     {
        //         auto lTextSize = ImGui::CalcTextSize( lFieldNames[i] );
        //         lLabelSizes[i] = lTextSize.x;
        //         lMaxLabelSize  = std::max( lMaxLabelSize, lTextSize.x );
        //         lTextHeight    = std::max( lTextHeight, lTextSize.y );
        //     }

        //     ImGui::AlignTextToFramePadding();
        //     Text( "Target distance:" );
        //     UI::SameLine();
        //     UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( lMaxLabelSize - lLabelSizes[0], 0.0f ) );
        //     UI::Slider( "##target_distance", "%f", 1.0f, 128.0f, &( g_TargetDistance ) );
        //     UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( 0.0f, 5.0f ) );

        //     ImGui::AlignTextToFramePadding();
        //     Text( "Intensity:" );
        //     UI::SameLine();
        //     UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( lMaxLabelSize - lLabelSizes[1], 0.0f ) );
        //     UI::Slider( "##intensity", "%f", 0.0f, 1.0f, &( g_Intensity ) );
        //     UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( 0.0f, 5.0f ) );

        //     ImGui::AlignTextToFramePadding();
        //     Text( "Base points:" );
        //     UI::SameLine();
        //     UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( lMaxLabelSize - lLabelSizes[2], 0.0f ) );
        //     UI::Slider( "##Basepoints", "%d", 100, 200, &( g_Basepoints ) );
        //     UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( 0.0f, 5.0f ) );

        //     ImGui::AlignTextToFramePadding();
        //     Text( "Oversampling:" );
        //     UI::SameLine();
        //     UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( lMaxLabelSize - lLabelSizes[3], 0.0f ) );
        //     UI::Slider( "##oversampling", "%d", 1, 1, &( g_Oversampling ) );
        //     UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( 0.0f, 5.0f ) );

        //     ImGui::AlignTextToFramePadding();
        //     Text( "Accumulation:" );
        //     UI::SameLine();
        //     UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( lMaxLabelSize - lLabelSizes[4], 0.0f ) );
        //     UI::Slider( "##accumulation", "%d", 1, 64, &( g_Accumulation ) );
        //     UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( 0.0f, 5.0f ) );

        //     ImGui::AlignTextToFramePadding();
        //     Text( "System temperature:" );
        //     UI::SameLine();
        //     UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( lMaxLabelSize - lLabelSizes[5], 0.0f ) );
        //     UI::Slider( "##temperature", "%f", -10.0f, 100.0f, &( g_SystemTemperature ) );
        //     UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( 0.0f, 5.0f ) );

        //     if( UI::Button( "Generate", { 150.0f, 50.0f } ) )
        //     {
        //         EnvironmentSampler::sCreateInfo lEnvironmentSamplingParameter{};
        //         AcquisitionSpecification lAcqCreateInfo{};
        //         lAcqCreateInfo.mBasePoints   = g_Basepoints;
        //         lAcqCreateInfo.mOversampling = g_Oversampling;

        //         auto lSamples = m_Sensor->Sample( lEnvironmentSamplingParameter, lAcqCreateInfo, std::string( "tile_1" ), math::vec2{ 0.0f, 0.0f }, 0.0f );

        //         auto lDistance = MultiTensorValue( *m_Sensor->mComputationScope, sConstantValueInitializerComponent( g_TargetDistance ),
        //                                            ( *lSamples )["Azimuth"].Get<sMultiTensorComponent>().mValue.Shape() );

        //         m_Sensor->Process( 0.0f, *m_Sensor->mComputationScope, lSamples->GetScheduledFlashes(), ( *lSamples )["Azimuth"], ( *lSamples )["Elevation"],
        //                            ( *lSamples )["Intensity"], lDistance );

        //         m_Sensor->mComputationScope->Run( ( *m_Sensor->mComputationScope )["FPGAOutput"] );
        //         mWaveforms = ( *m_Sensor->mComputationScope )["SaturatedWaveforms"].Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
        //     }

        //     if( mWaveforms.size() != 0 )
        //     {
        //         uint32_t l_WaveformLength = g_Basepoints * g_Oversampling;
        //         std::vector<float> l_X( l_WaveformLength );
        //         for( int i = 0; i < l_WaveformLength; i++ )
        //         {
        //             l_X[i] = (float)i;
        //         }

        //         auto l_WindowSize = LTSE::Core::UI::GetAvailableContentSpace();

        //         for( uint32_t i = 0; i < 32; i++ )
        //         {
        //             ImPlot::PushColormap( ImPlotColormap_Jet );
        //             ImGui::PushID( i );
        //             if( ImPlot::BeginPlot( "##Perlin", ImVec2( 200.0f, 75.0f ), ImPlotFlags_CanvasOnly | ImPlotFlags_NoInputs | ImPlotFlags_NoFrame | ImPlotFlags_NoChild ) )
        //             {
        //                 ImGui::PushID( i );
        //                 ImPlot::SetupAxes( NULL, NULL, ImPlotAxisFlags_NoDecorations, ImPlotAxisFlags_NoDecorations );
        //                 ImPlot::PlotHeatmap( "##T", mWaveforms.data() + i * 32 * l_WaveformLength, 32, l_WaveformLength, 0.0f, 1.0005f, NULL );
        //                 ImGui::PopID();
        //                 ImPlot::EndPlot();
        //             }
        //             ImGui::PopID();
        //             ImPlot::PopColormap();
        //             if( ( i + 1 ) % 8 != 0 )
        //                 ImGui::SameLine();
        //         }

        //         ImPlot::SetNextAxisLimits( ImAxis_X1, -1, 101, ImPlotCond_Once );
        //         ImPlot::SetNextAxisLimits( ImAxis_Y1, -1.0001f, 2.009f, ImPlotCond_Once );
        //         if( ImPlot::BeginPlot( "Waveforms", "Sample", "Amplitude", ImVec2{ (float)l_WindowSize.x, (float)l_WindowSize.y } ) )
        //         {
        //             for( uint32_t i = 0; i < 1024; i++ )
        //             {
        //                 ImGui::PushID( i );
        //                 ImPlot::PlotLine( "##Line", l_X.data(), mWaveforms.data() + ( i * l_WaveformLength ), l_WaveformLength );
        //                 ImGui::PopID();
        //             }
        //             ImPlot::EndPlot();
        //         }
        //     }
        // }
        // ImGui::End();

        // static bool p_open = true;
        // if( ImGui::Begin( "ECHODS_MVP SIMULATION CONFIGURATION", &p_open, ImGuiWindowFlags_None ) )
        // {
        //     auto l_DrawList   = ImGui::GetWindowDrawList();
        //     auto l_WindowSize = UI::GetAvailableContentSpace();

        //     auto l_TextSize0   = ImGui::CalcTextSize( "Sampling resolution:" );
        //     auto l_SliderSize0 = static_cast<float>( l_WindowSize.x ) - l_TextSize0.x - 35.0f;

        //     UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 10.0f ) );
        //     auto l_TextSize5 = ImGui::CalcTextSize( "Tile layout:" );
        //     Text( "Tile layout:" );
        //     ImGui::SameLine();
        //     TileLayoutCombo lLayoutChooser( "##FOO" );
        //     ImGui::SetNextItemWidth( l_SliderSize0 );
        //     UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( ( l_TextSize0.x - l_TextSize5.x ) + 10.0f, -5.0f ) );
        //     lLayoutChooser.SensorModel = m_Sensor;
        //     lLayoutChooser.Display( CurrentTileLayout );

        //     UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 10.0f ) );
        //     auto l_TextSize6 = ImGui::CalcTextSize( "Sensor controller:" );
        //     Text( "Sensor controller:" );
        //     ImGui::SameLine();
        //     ControllerChooser lControllerChooser( "##FOO_2" );
        //     ImGui::SetNextItemWidth( l_SliderSize0 );
        //     UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( ( l_TextSize0.x - l_TextSize6.x ) + 10.0f, -5.0f ) );
        //     lControllerChooser.Dropdown.Labels = { "None", "Basic sensor controller", "LeddarEngine connection" };
        //     lControllerChooser.Dropdown.Values = { eControllerID::NONE, eControllerID::BASIC, eControllerID::LEDDAR_ENGINE };
        //     lControllerChooser.Display( CurrentControllerID );
        //     if( lControllerChooser.Dropdown.Changed )
        //     {
        //         CurrentControllerID = lControllerChooser.Dropdown.GetValue();
        //     }

        //     switch( CurrentControllerID )
        //     {
        //     case eControllerID::BASIC:
        //     {
        //         Text( "Basic sensor controller" );
        //         UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( 10.0f, 0.0f ) );
        //         break;
        //     }
        //     case eControllerID::LEDDAR_ENGINE:
        //     {
        //         auto l_TextSize0 = ImGui::CalcTextSize( "Sampling resolution:" );
        //         auto l_TextSize1 = ImGui::CalcTextSize( "." );

        //         ImGui::AlignTextToFramePadding();
        //         auto l_TextSize2 = ImGui::CalcTextSize( "LeddarEngine IP:" );
        //         UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 0.0f ) );
        //         Text( "LeddarEngine IP:" );
        //         UI::SameLine();

        //         UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( l_TextSize0.x - l_TextSize2.x + 10.0f, 0.0f ) );
        //         float lIPInputWidth = ( ( l_WindowSize.x - l_TextSize0.x - 40.0f ) - ( 3.0f * l_TextSize1.x ) ) / 4.0f;

        //         char buf0[128] = { 0 };
        //         std::strncpy( buf0, "192", 3 );
        //         ImGui::SetNextItemWidth( lIPInputWidth );
        //         if( ImGui::InputText( "##LE_IP_0", buf0, ARRAYSIZE( buf0 ), ImGuiInputTextFlags_EnterReturnsTrue ) )
        //         {
        //         }
        //         UI::SameLine();
        //         Text( "." );
        //         UI::SameLine();
        //         char buf1[128] = { 0 };
        //         std::strncpy( buf1, "168", 3 );
        //         ImGui::SetNextItemWidth( lIPInputWidth );
        //         if( ImGui::InputText( "##LE_IP_1", buf1, ARRAYSIZE( buf1 ), ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_CharsDecimal ) )
        //         {
        //         }
        //         UI::SameLine();
        //         Text( "." );
        //         UI::SameLine();
        //         char buf2[128] = { 0 };
        //         std::strncpy( buf2, "204", 3 );
        //         ImGui::SetNextItemWidth( lIPInputWidth );
        //         if( ImGui::InputText( "##LE_IP_2", buf2, ARRAYSIZE( buf2 ), ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_CharsDecimal ) )
        //         {
        //         }
        //         UI::SameLine();
        //         Text( "." );
        //         UI::SameLine();
        //         char buf3[128] = { 0 };
        //         std::strncpy( buf3, "130", 3 );
        //         ImGui::SetNextItemWidth( lIPInputWidth );
        //         if( ImGui::InputText( "##LE_IP_3", buf3, ARRAYSIZE( buf3 ), ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_CharsDecimal ) )
        //         {
        //         }

        //         ImGui::AlignTextToFramePadding();
        //         UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 0.0f ) );
        //         auto l_TextSize3 = ImGui::CalcTextSize( "LeddarEngine IP:" );
        //         Text( "LeddarEngine IP:" );
        //         UI::SameLine();
        //         char buf4[128] = { 0 };
        //         std::strncpy( buf4, "27100", 5 );
        //         UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( l_TextSize0.x - l_TextSize3.x + 10.0f, 0.0f ) );
        //         ImGui::SetNextItemWidth( 150 );
        //         if( ImGui::InputText( "##LE_PORT", buf4, ARRAYSIZE( buf4 ), ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_CharsDecimal ) )
        //         {
        //         }

        //         UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( 10.0f, 0.0f ) );
        //         break;
        //     }
        //     case eControllerID::NONE:
        //     default:
        //         break;
        //     }
        // }
        // ImGui::End();
    }

    void Init()
    {
        BaseEditorApplication::Init();
        // m_Sensor = a_SensorToControl;
        // mEditorWindow.OnBeginScenario.connect<&EchoDSMVPEditor::OnBeginScenario>( *this );
    }

    void Update( Timestep ts )
    {
        BaseEditorApplication::Update( ts );

        // if( !m_WorldSampler )
        //     m_WorldSampler = New<WorldSampler>( m_World->GetRayTracingContext() );

        // Ref<EnvironmentSampler> lEnvSamples = nullptr;
        // if( CurrentTileLayout )
        // {
        //     auto &lTileLayout                      = CurrentTileLayout.Get<sTileLayoutComponent>().mLayout;
        //     std::vector<std::string> lTileIds      = {};
        //     std::vector<math::vec2> lTilePositions = {};
        //     std::vector<float> lTileTimes          = {};
        //     for( auto &x : lTileLayout )
        //     {
        //         lTileIds.push_back( x.second.first );
        //         lTilePositions.push_back( x.second.second );
        //         lTileTimes.push_back( 0.0f );
        //     }

        //     lEnvSamples = m_Sensor->Sample( mEditorWindow.ActiveSensor.Get<EnvironmentSampler::sCreateInfo>(), mEditorWindow.ActiveSensor.Get<AcquisitionSpecification>(),
        //     lTileIds,
        //                                     lTilePositions, lTileTimes );
        // }
        // else
        // {
        //     return;
        // }
    }

    void DisplayPointCloud() {}

  private:
    // PointCloudVisualizer m_PointCloudVisualizer{};
    // Ref<FLMSensorDevice> m_Sensor    = nullptr;
    // Ref<WorldSampler> m_WorldSampler = nullptr;
    // Ref<Scope> m_ComputeScope        = nullptr;
    // float m_PointSize                = 0.0;

    // Entity Sensor{};
    // Entity CurrentTileLayout{};

    // bool RunSensorSimulation                    = false;
    // eControllerID CurrentControllerID           = eControllerID::NONE;
    // Ref<SensorControllerBase> CurrentController = nullptr;
};

int main( int argc, char **argv )
{
    EchoDSMVPEditor g_EditorWindow{};

    // std::shared_ptr<FLMSensorDevice> lSensorDevice = New<FLMSensorDevice>();

    // Ref<SensorModelBase> lSensorModel = Build<SensorModelBase>( "C:\\GitLab\\EchoDS_FLM\\SensorData", "TestSensor0.yaml" );
    // lSensorDevice->mSensorDefinition  = lSensorModel;

    g_EditorWindow.Init();

    return g_EditorWindow.Run();
}
