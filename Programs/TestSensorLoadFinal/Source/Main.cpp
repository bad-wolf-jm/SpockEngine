#include "Developer/Platform/EngineLoop.h"
#include "Developer/GraphicContext/UI/UIContext.h"

#include "Developer/UI/UI.h"
#include "Developer/UI/Widgets.h"

// #include "AssetManager/SensorModelLoader.h"

// #include "SensorModelDev/Base/Components.h"
// #include "SensorModelDev/Base/KernelComponents.h"
// #include "SensorModelDev/Developer/Editor/EditorComponents.h"
// #include "SensorModelDev/Developer/Editor/SensorModelEditor.h"

#include "LidarSensorModel/ModelBuilder.h"
#include "LidarSensorModel/ModelArchive.h"
#include "LidarSensorModel/SensorDeviceBase.h"


#include "Cuda/CudaBuffer.h"
#include "Developer/Core/Cuda/ExternalMemory.h"

#include "Core/Logging.h"
#include "Core/Memory.h"

#include <filesystem>

#include "TensorOps/Implementation/KernelLaunchers.h"
#include "TensorOps/NodeComponents.h"
#include "TensorOps/Scope.h"

#include "Serialize/FileIO.h"
#include "Serialize/SensorAsset.h"
#include "Serialize/SensorDefinition.h"

// #include "WaveformGenerator.h"

#include <chrono>

using namespace LTSE::Core;
using namespace LTSE::TensorOps;
using namespace LTSE::SensorModel;
namespace fs = std::filesystem;

class TestSensorDevice : public SensorDeviceBase
{
  public:
    TestSensorDevice()
        : SensorDeviceBase( 64 * 1024 * 1024 ){};

    Ref<EnvironmentSampler> Sample( EnvironmentSampler::sCreateInfo &aSamplerCreateInfo ) { return nullptr; }
};

int main( int argc, char **argv )
{
    // std::shared_ptr<TestSensorDevice> g_SensorModel;
    // sSensorDefinition lSensorDefinition = ReadSensorDefinition( "C:\\GitLab\\LTSimulationEngine\\Programs\\TestSensorLoadFinal\\Data", "TestSensor0.yaml" );
    // g_SensorModel = New<TestSensorDevice>();
    std::shared_ptr<SensorModelBase> g_SensorModel = Build<SensorModelBase>( "C:\\GitLab\\LTSimulationEngine\\Programs\\TestSensorLoadFinal\\Data", "TestSensor0.yaml" );
    // std::shared_ptr<TestSensorDevice> g_SensorModel = Build<TestSensorDevice>( "C:\\GitLab\\LTSimulationEngine\\Programs\\TestSensorLoadFinal\\Data", "TestSensor0.yaml" );
    // g_SensorModel->Load(lSensorDefinition);
    Save( g_SensorModel, "C:\\GitLab\\LTSimulationEngine\\Programs\\TestSensorLoadFinal\\Data", "TestSensor0_saved.yaml" );

    return 0;
}
