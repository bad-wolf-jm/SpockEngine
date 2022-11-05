
#include "Core/GraphicContext//UI/UIContext.h"
#include "Engine/Engine.h"

#include <fmt/core.h>
#include <fstream>

#include "Core/EntityRegistry/ScriptableEntity.h"
#include "Core/Logging.h"

#include "UI/UI.h"
#include "UI/Widgets.h"

#include "Editor/BaseEditorApplication.h"
#include "Editor/EditorWindow.h"

#include "Mono/Manager.h"

#include "Scene/EnvironmentSampler/EnvironmentSampler.h"

using namespace LTSE::Core;
using namespace LTSE::Graphics;
using namespace LTSE::Editor;
using namespace LTSE::Core::UI;
using namespace LTSE::SensorModel;
using namespace LTSE::SensorModel::Dev;


struct SensorControllerBehaviour : sBehaviourController
{
    Ref<Scope>        m_ComputeScope = nullptr;
    Ref<WorldSampler> m_WorldSampler = nullptr;
    Ref<Engine>   mEngineLoop    = nullptr;
    Ref<Scene>        m_World        = nullptr;

    sPointCloudVisualizer m_PointCloudVisualizer{};

    SensorControllerBehaviour( Ref<Engine> aEngineLoop, Ref<Scene> aWorld )
        : mEngineLoop{ aEngineLoop }
        , m_World{ aWorld }
    {
    }

    void OnCreate()
    {
        m_ComputeScope = New<Scope>( 512 * 1024 * 1024 );
        m_WorldSampler = New<WorldSampler>( m_World->GetRayTracingContext() );
    }

    void OnDestroy() {}

    void OnUpdate( Timestep ts )
    {
        sRandomUniformInitializerComponent lInitializer{};
        lInitializer.mType = eScalarType::FLOAT32;

        std::vector<uint32_t> lDim1{ 2500, 2000 };

        auto lAzimuths    = MultiTensorValue( *m_ComputeScope, lInitializer, sTensorShape( { lDim1 }, sizeof( float ) ) );
        auto lElevations  = MultiTensorValue( *m_ComputeScope, lInitializer, sTensorShape( { lDim1 }, sizeof( float ) ) );
        auto lIntensities = MultiTensorValue( *m_ComputeScope, lInitializer, sTensorShape( { lDim1 }, sizeof( float ) ) );

        auto lRange = ConstantScalarValue( *m_ComputeScope, 25.0f );

        lAzimuths   = Multiply( *m_ComputeScope, lAzimuths, lRange );
        lElevations = Multiply( *m_ComputeScope, lElevations, lRange );
        m_ComputeScope->Run( { lAzimuths, lElevations, lIntensities } );

        sTensorShape lOutputShape( lIntensities.Get<sMultiTensorComponent>().mValue.Shape().mShape, sizeof( sHitRecord ) );
        MultiTensor  lHitRecords = MultiTensor( m_ComputeScope->mPool, lOutputShape );
        if( !Has<sTransformMatrixComponent>() ) return;

        auto &lParticles = Get<sParticleSystemComponent>();

        m_WorldSampler->Sample( Get<sTransformMatrixComponent>().Matrix, m_World, lAzimuths.Get<sMultiTensorComponent>().mValue,
            lElevations.Get<sMultiTensorComponent>().mValue, lIntensities.Get<sMultiTensorComponent>().mValue, lHitRecords );

        if( !( lParticles.Particles ) || lParticles.ParticleCount != lAzimuths.Get<sMultiTensorComponent>().mValue.SizeAs<float>() )
        {
            lParticles.ParticleCount = lAzimuths.Get<sMultiTensorComponent>().mValue.SizeAs<float>();
            lParticles.Particles = New<Buffer>( mEngineLoop->GetGraphicContext(), eBufferBindType::VERTEX_BUFFER, false, true, true,
                true, lParticles.ParticleCount * sizeof( Particle ) );
        }

        GPUExternalMemory lPointCloudMappedBuffer( *( lParticles.Particles ), lParticles.ParticleCount * sizeof( Particle ) );
        m_PointCloudVisualizer.mInvertZAxis = false;
        m_PointCloudVisualizer.mResolution  = 0.2;
        m_PointCloudVisualizer.Visualize( Get<sTransformMatrixComponent>().Matrix, lHitRecords, lPointCloudMappedBuffer );
        lPointCloudMappedBuffer.Dispose();
    }
};

class EchoDSMVPEditor : public BaseEditorApplication
{
  public:
    EchoDSMVPEditor()
        : BaseEditorApplication()
    {
        ApplicationName = "EchoDS_MVP";
        WindowSize      = { 2920, 1580 };
    };

    ~EchoDSMVPEditor() = default;

    void LoadSensorConfiguration() {}
    void SaveSensorConfiguration() {}

    std::vector<float> mWaveforms = {};

    float      g_TargetDistance    = 2.0f;
    float      g_Intensity         = 1.0f;
    math::vec4 g_TimebaseDelay     = math::vec4{ 0.0f, 0.0f, 0.0f, 0.0f };
    uint32_t   g_Basepoints        = 100;
    uint32_t   g_Oversampling      = 1;
    uint32_t   g_Accumulation      = 1;
    float      g_SystemTemperature = 25.0f;
    float      g_StaticNoiseFactor = .001f;

    void OnBeginScenario()
    {
        mEditorWindow.Sensor.Get<sBehaviourComponent>().Bind<SensorControllerBehaviour>( mEngineLoop, mWorld );
    }

    void OnUI()
    {

        static bool p_open1 = true;

        if( ImGui::Begin( "IMPLOT DEMO", &p_open1, ImGuiWindowFlags_None ) )
        {
            ImPlot::ShowDemoWindow();
        }
        ImGui::End();

    }

    void Init()
    {
        BaseEditorApplication::Init();
        mEditorWindow.OnBeginScenario.connect<&EchoDSMVPEditor::OnBeginScenario>( *this );

        ScriptManager::SetAppAssemblyPath("C:/GitLab/SpockEngine/Programs/Editor/Script/Build/Debug/Test.dll");

        mWorld->AttachScript(mEditorWindow.Sensor, "Test", "TestActorComponent");
    }

    void Update( Timestep ts )
    {
        BaseEditorApplication::Update( ts );

    }

    void DisplayPointCloud() {}

  private:
};

int main( int argc, char **argv )
{
    EchoDSMVPEditor g_EditorWindow{};

    g_EditorWindow.Init();

    return 0;
    // return g_EditorWindow.Run();
}
