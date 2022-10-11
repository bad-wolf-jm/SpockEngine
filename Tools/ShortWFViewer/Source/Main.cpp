#include "Developer/Platform/EngineLoop.h"
#include "Developer/GraphicContext/UI/UIContext.h"

#include "Developer/UI/UI.h"
#include "Developer/UI/Widgets.h"

#include "Cuda/CudaBuffer.h"
#include "Developer/Core/Cuda/ExternalMemory.h"

#include "Core/Logging.h"
#include "Core/Memory.h"

#include <filesystem>
#include <fstream>

#include "TensorOps/Implementation/KernelLaunchers.h"
#include "TensorOps/NodeComponents.h"
#include "TensorOps/Scope.h"

#include "Serialize/SensorAsset.h"

#include "ProcessingNodes/WaveformGenerator.h"
#include "ProcessingNodes/WaveformType.h"

#include "LidarSensorModel/Components.h"
#include "LidarSensorModel/EnvironmentSampler.h"
#include "LidarSensorModel/SensorDeviceBase.h"
#include "LidarSensorModel/SensorModelBase.h"

#include <chrono>

using namespace LTSE::Core;
using namespace LTSE::TensorOps;
using namespace LTSE::SensorModel;
namespace fs = std::filesystem;

EngineLoop g_EngineLoop;

uint32_t frameCounter = 0;
float fpsTimer        = 0.0f;
uint32_t lastFPS      = 0;

float ComputeFpsTimer        = 0.0f;
uint32_t ComputeLastFPS      = 0;
uint32_t ComputeFrameCounter = 0;

size_t l_PoolSize = 1024 * 1024 * 1024 * 3;

std::vector<sWaveformPacket> buffer;

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

    // a_Scope.Reset();
    // RunPipeline( a_Scope );
}

bool RenderUI( ImGuiIO &io )
{
    auto l_WindowSize = UI::GetRootWindowSize();
    bool Quit         = false;

    static bool p_open = true;

    ImGui::Begin( "Short waveforms", &p_open, ImGuiWindowFlags_None );
    {
        ImVec2 l_PopupSize = ImGui::GetWindowSize();

        ImGuiTableFlags flags = ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY | ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV |
                                ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable;
        ImGui::PushStyleVar( ImGuiStyleVar_CellPadding, ImVec2( 5, 5 ) );

        std::array<std::string, 19> lColumns = { "SC", "PD", "ID",     "VER", "MAX_D", "SPL_C",  "DET",         "FRM_NUM",          "BASE", "NOISE", "DIST", "LAST_UNSAT", "PULSE_BL",
                                                 "SAT_BL", "AMP", "MAX_I", "OFFSET", "RAW_SAMPLES", "PROCESSED_SAMPLES" };

        if( ImGui::BeginTable( "##TABLE", lColumns.size(), flags, ImVec2{ l_PopupSize.x - 20.0f, l_PopupSize.y - 150.0f } ) )
        {
            ImGui::TableSetupScrollFreeze( 1, 1 );
            for( auto &a_ColumnName : lColumns )
                ImGui::TableSetupColumn( a_ColumnName.c_str() );
            ImGui::TableHeadersRow();

            uint32_t IDX = 0;
            for( auto &lSWF : buffer )
            {
                ImGui::TableNextRow();

                uint32_t aID, aVersion, aDetectionCount, aMaxDetectionCount, aSampleCount;
                lSWF.mHeader.UnpackHeader( aID, aVersion, aDetectionCount, aMaxDetectionCount, aSampleCount );

                ImGui::TableSetColumnIndex( 0 );
                UI::Text( "{}", IDX / 32 );

                ImGui::TableSetColumnIndex( 1 );
                UI::Text( "{}", IDX % 32 );

                ImGui::TableSetColumnIndex( 2 );
                UI::Text( "{}", aID );

                ImGui::TableSetColumnIndex( 3 );
                UI::Text( "{}", aVersion );

                ImGui::TableSetColumnIndex( 4 );
                UI::Text( "{}", aMaxDetectionCount );

                ImGui::TableSetColumnIndex( 5 );
                UI::Text( "{}", aSampleCount );

                ImGui::TableSetColumnIndex( 6 );
                UI::Text( "{}", aDetectionCount );

                ImGui::TableSetColumnIndex( 7 );
                UI::Text( "{}", lSWF.mHeader.mFrameNumber );

                ImGui::TableSetColumnIndex( 8 );
                UI::Text( "{}", lSWF.mHeader.mTraceBaseLevel );

                ImGui::TableSetColumnIndex( 9 );
                UI::Text( "{}", lSWF.mHeader.mTraceNoiseLevel );

                ImGui::TableSetColumnIndex( 10 );
                UI::Text( "{}", lSWF.mWaveform[0].mPulse.mInterpolatedDistance );

                ImGui::TableSetColumnIndex( 11 );
                UI::Text( "{}", lSWF.mWaveform[0].mPulse.mLastUnsaturatedSample );

                ImGui::TableSetColumnIndex( 12 );
                UI::Text( "{}", lSWF.mWaveform[0].mPulse.mPulseBaseLevel );

                ImGui::TableSetColumnIndex( 13 );
                UI::Text( "{}", lSWF.mWaveform[0].mPulse.mBaselineAfterSaturation );

                ImGui::TableSetColumnIndex( 14 );
                UI::Text( "{}", lSWF.mWaveform[0].mPulse.mAmplitude );

                ImGui::TableSetColumnIndex( 15 );
                UI::Text( "{}", lSWF.mWaveform[0].mPulse.mMaxIndex );

                ImGui::TableSetColumnIndex( 16 );
                UI::Text( "{}", lSWF.mWaveform[0].mPulse.mOffset );

                ImGui::TableSetColumnIndex( 17 );
                {
                    float lValues[11];
                    for( uint32_t i = 0; i < 11; i++ )
                        lValues[i] = static_cast<float>( lSWF.mWaveform[0].mRawTrace[i] );
                    ImGui::PlotLines( "##", lValues, 11 );
                }

                ImGui::TableSetColumnIndex( 18 );
                {
                    float lValues[11];
                    for( uint32_t i = 0; i < 11; i++ )
                        lValues[i] = static_cast<float>( lSWF.mWaveform[0].mProcessedTrace[i] );
                    ImGui::PlotLines( "##", lValues, 11 );
                }
                IDX ++;
            }

            ImGui::EndTable();
        }
        ImGui::PopStyleVar();
    }
    ImGui::End();

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

    std::ifstream wf( "C:\\GitLab\\EchoDS_FLM\\Tile.bin", std::ios::in | std::ios::binary | std::ios::ate );
    std::streamsize size = wf.tellg();
    wf.seekg( 0, std::ios::beg );

    buffer = std::vector<sWaveformPacket>( size / sizeof( sWaveformPacket ) );
    // buffer.resize( size );
    wf.read( reinterpret_cast<char *>( buffer.data() ), size );

    while( g_EngineLoop.Tick() )
        ;

    return 0;
}
