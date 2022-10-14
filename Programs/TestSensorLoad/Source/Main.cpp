#include "Core/Platform/EngineLoop.h"
#include "Core/GraphicContext//UI/UIContext.h"

#include "UI/UI.h"
#include "UI/Widgets.h"

#include "AssetManager/SensorModelLoader.h"

#include "SensorModelDev/Base/Components.h"
#include "SensorModelDev/Base/KernelComponents.h"
#include "SensorModelDev/Editor/EditorComponents.h"
#include "SensorModelDev/Editor/SensorModelEditor.h"

#include "Core/Cuda/CudaBuffer.h"
#include "Core/Cuda/ExternalMemory.h"

#include "Core/Logging.h"

#include <filesystem>

#include "TensorOps/Implementation/KernelLaunchers.h"
#include "TensorOps/NodeComponents.h"
#include "TensorOps/Scope.h"

// #include "WaveformGenerator.h"

#include <chrono>

using namespace LTSE::Core;
using namespace LTSE::TensorOps;
using namespace LTSE::SensorModel::Dev;
namespace fs = std::filesystem;

EngineLoop g_EngineLoop;
std::shared_ptr<SensorModelEditor> g_SensorModel;
math::vec2 g_PhotodetectorCellSize = { 0.2f, 0.2f };
math::vec2 g_TileFOV;
size_t g_FlashCount  = 32;
float g_FlashSpacing = 0.205;
Entity g_Tile{};
Entity g_Attenuation{};
Entity g_Reduction{};
Entity g_WaveformTemplate{};

std::vector<float> g_Waveforms;

float g_EnvironmentSamplingResolution = 0.1f;
float g_TargetDistance                = 2.0f;
constexpr float SPEED_OF_LIGHT        = 299792458.0f;

uint32_t frameCounter = 0;
float fpsTimer        = 0.0f;
uint32_t lastFPS      = 0;

float ComputeFpsTimer        = 0.0f;
uint32_t ComputeLastFPS      = 0;
uint32_t ComputeFrameCounter = 0;


int main( int argc, char **argv )
{
    g_EngineLoop = EngineLoop();
    g_EngineLoop.PreInit( 0, nullptr );
    g_EngineLoop.Init();

    g_SensorModel = LoadModel( g_EngineLoop.GetDevice(), g_EngineLoop.UIContext(), "C:\\GitLab\\LTSimulationEngine\\Programs\\TestSensorLoad\\Data", "TestSensor0.yaml" );
    SaveModelDefinition(g_SensorModel, "C:\\GitLab\\LTSimulationEngine\\Programs\\TestSensorLoad\\Data\\", "TestSensor0_saved.yaml");

    return 0;
}
