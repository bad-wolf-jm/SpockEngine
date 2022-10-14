
#include "Core/TextureData.h"

// #include "Core/GraphicContext//OffscreenRenderTarget.h"

#include "Core/Platform/EngineLoop.h"
#include "Core/GraphicContext//UI/UIContext.h"

// #include "SensorModelDev/Base/Components.h"
// #include "SensorModelDev/Base/KernelComponents.h"
// #include "SensorModelDev/Editor/EditorComponents.h"
// #include "SensorModelDev/Editor/PointCloudVisualizer.h"
// #include "SensorModelDev/Editor/SensorModelEditor.h"
// #include "SensorModelDev/EnvironmentSampler/EnvironmentSampler.h"

#include "Scene/Animation.h"
// #include "Scene/Assetloader.h"
#include "Scene/Primitives/Primitives.h"
#include "Scene/Scene.h"

#include "Scene/Visualizers/Wall2D.h"

#include "Scene/Renderer/ParticleSystemRenderer.h"
#include "Scene/Renderer/SceneRenderer.h"
#include "Scene/Renderer/VisualHelperLineRenderer.h"

#include "Core/GraphicContext//Buffer.h"
#include "Scene/ParticleData.h"

#include "UI/UI.h"
#include "UI/Widgets.h"

#include "Core/Cuda/CudaBuffer.h"
#include "Core/Cuda/ExternalMemory.h"

#include <fmt/core.h>

#include "Core/Logging.h"
#include "Core/Memory.h"

#include "TensorOps/Scope.h"

#include "ImGuizmo.h"

// #include "Device.h"
// #include "TestSensorModel.h"

#include <fstream>
#include <iostream>

using namespace LTSE::Core;
using namespace LTSE::Cuda;
using namespace LTSE::Graphics;
// using namespace LTSE::SensorModel::Dev;
// using namespace LTSE::SensorModel::Test;
using namespace LTSE::Core::Primitives;
using namespace math::literals;
using namespace LTSE::Core::UI;
using namespace LTSE::TensorOps;

LTSE::Core::EngineLoop *g_EngineLoop;
// std::shared_ptr<LTSE::SensorModel::Dev::SensorModelEditor> g_SensorModel;

// Ref<TestSensorDevice> g_SensorModel = nullptr;

std::shared_ptr<LTSE::Core::Scene> g_World;
std::shared_ptr<LTSE::Core::SceneRenderer> g_WorldRenderer;

uint32_t frameCounter = 0;
float fpsTimer        = 0.0f;
uint32_t lastFPS      = 0;

VertexBufferData g_Sphere;

int32_t m_DisplayFlashID         = 1;
float m_PointSize                = 0.0;
bool m_UseOrthographicProjection = false;
bool m_DisplayFlashGrid          = false;

math::vec2 g_TileFOV;
math::vec2 g_PhotodetectorCellSize = { 0.2f, 0.2f };
Entity g_Tile;
Entity g_Attenuation;
size_t g_FlashCount  = 32;
float g_FlashSpacing = 0.205;

uint32_t m_TotalRayCount;
std::vector<Entity> m_CurrentFlashes                                                 = {};
std::vector<std::shared_ptr<Buffer>> m_FlashDisplayPointClouds = {};
std::vector<size_t> m_FlashDisplayPointCloudParticleCounts                           = {};

std::shared_ptr<Buffer> g_PointCloud;

std::shared_ptr<ParticleSystemRenderer> m_ParticleSystemRenderer = nullptr;
std::shared_ptr<VisualHelperLineRenderer> m_GizmoRenderer        = nullptr;

std::vector<Wall2D> g_SubdivisionVisualizers                         = {};
std::vector<std::shared_ptr<Buffer>> g_SubdivisionVisualizerVertices = {};
std::vector<std::shared_ptr<Buffer>> g_SubdivisionVisualizerIndices  = {};
std::vector<uint32_t> g_SubdivisionVisualizerSizes                   = {};

// LTSE::SensorModel::Dev::PointCloudVisualizer l_PointCloudVisualizer;

math::mat4 g_SensorTransform;

// std::shared_ptr<WorldSampler> g_WorldSampler = nullptr;

math::mat4 g_SphereTransforms[3];
std::vector<Scene::Element> g_Material = {};

void Table( std::string a_Name, std::vector<std::string> a_Columns, math::vec2 a_Size, std::function<void()> a_RenderContents )
{
    static ImGuiTableFlags flags = ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY | ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV |
                                   ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable;
    ImGui::PushStyleVar( ImGuiStyleVar_CellPadding, ImVec2( 5, 5 ) );

    if( ImGui::BeginTable( a_Name.c_str(), a_Columns.size(), flags, ImVec2{ a_Size.x, a_Size.y } ) )
    {
        ImGui::TableSetupScrollFreeze( 1, 1 );
        for( auto &a_ColumnName : a_Columns )
            ImGui::TableSetupColumn( a_ColumnName.c_str() );
        ImGui::TableHeadersRow();
        try
        {
            a_RenderContents();
        }
        catch( const std::exception &e )
        {
            LTSE::Logging::Info( "ERROR - {}", e.what() );
        }
        ImGui::EndTable();
    }
    ImGui::PopStyleVar();
}

void RenderScene()
{
    g_World->Update( 0.0f );
    g_WorldRenderer->Render( g_EngineLoop->GetRenderContext() );
    math::ivec2 l_ViewportSize = g_EngineLoop->GetViewportSize();

    ParticleSystemRenderer::ParticleData l_ParticleData{};
    l_ParticleData.Model         = math::mat4( 1.0f );
    l_ParticleData.ParticleCount = m_TotalRayCount;
    l_ParticleData.ParticleSize  = m_PointSize;
    l_ParticleData.Particles     = g_PointCloud;
    m_ParticleSystemRenderer->Render( g_WorldRenderer->View.Projection, g_WorldRenderer->View.View, g_EngineLoop->GetRenderContext(), l_ParticleData );
}

float g_TotalTime;
std::vector<float> gAzimuths;
std::vector<float> gElevations;
// std::vector<HitRecord> gResults;

void Update( Scope &a_Scope, Timestep ts )
{
    frameCounter++;
    fpsTimer += (float)ts;

    // LTSE::Logging::Info( "Frame time: {}", (float)ts );

    if( fpsTimer > 1000.0f )
    {
        lastFPS      = static_cast<uint32_t>( (float)frameCounter * ( 1000.0f / fpsTimer ) );
        fpsTimer     = 0.0f;
        frameCounter = 0;
    }

    g_World->Update( ts );

    // Generate all samples for a single frame
    // auto lEnvSamples = g_SensorModel->Sample();

    // if( !lEnvSamples )
    //     return;

    // MultiTensor &lAzimuths    = ( *lEnvSamples )["Azimuth"].Get<sMultiTensorComponent>().mValue;
    // MultiTensor &lElevations  = ( *lEnvSamples )["Elevation"].Get<sMultiTensorComponent>().mValue;
    // MultiTensor &lIntensities = ( *lEnvSamples )["Intensity"].Get<sMultiTensorComponent>().mValue;

    // sTensorShape lOutputShape( lIntensities.Shape().mShape, sizeof( HitRecord ) );

    // a_Scope.Reset();
    // MultiTensor lHitRecords = MultiTensor( a_Scope.mPool, lOutputShape );
    // g_WorldSampler->Sample( g_SensorTransform, g_World, lAzimuths, lElevations, lIntensities, lHitRecords );

    // gResults    = lHitRecords.FetchFlattened<HitRecord>();
    // gAzimuths   = lAzimuths.FetchFlattened<float>();
    // gElevations = lElevations.FetchFlattened<float>();

    // if( !( g_PointCloud ) || m_TotalRayCount != lAzimuths.SizeAs<float>() )
    // {
    //     m_TotalRayCount = lAzimuths.SizeAs<float>();

    //     g_PointCloud = g_EngineLoop->GetGraphicContext().New<VertexBufferObject<Particle>>( m_TotalRayCount );
    // }

    // GPUExternalMemory l_PointCloudMappedBuffer( *( g_PointCloud->GetBuffer() ), m_TotalRayCount * sizeof( Particle ) );
    // l_PointCloudVisualizer.InvertZAxis = true;
    // l_PointCloudVisualizer.Resolution = 0.9;
    // l_PointCloudVisualizer.Visualize( g_SensorTransform, lHitRecords, l_PointCloudMappedBuffer );

    // l_PointCloudMappedBuffer.Dispose();
}

bool RenderUI( ImGuiIO &io )
{
    static bool l_RenderBackground;
    static bool l_RenderGrid;
    static float l_Exposure;
    static float l_Gamma;
    static float l_IBLScale;
    static float l_ModelScale = 0.5f;
    static float l_CameraX    = 0.0f;
    static float l_CameraY    = 1.0f;
    static float l_CameraZ    = 7.5f;

    ViewManipulate( g_WorldRenderer->View.CameraPosition, g_WorldRenderer->View.View, { 400.0f, 15.0f } );
    g_WorldRenderer->View.CameraPosition = math::vec3( math::Inverse( g_WorldRenderer->View.View )[3] );

    UI::Pane( "##SCENE_DISPLAY", { 15.0f, 15.0f }, { 350.0f, 350.0f },
              [&]()
              {
                  if( lastFPS > 0 )
                      UI::Text( fmt::format( "Render: {} fps ({:.2f} ms)", lastFPS, ( 1000.0f / lastFPS ) ).c_str() );
                  else
                      UI::Text( "Render: 0 fps (0 ms)" );

                  if( ImGui::CollapsingHeader( "Scene" ) )
                  {
                      static float asset_scale = 1.0f;
                      ImGui::SliderFloat( "Scale", &asset_scale, 0.001f, 1.0f );
                  }

                  if( ImGui::CollapsingHeader( "Rendering" ) )
                  {
                      ImGui::Checkbox( "Render background", &l_RenderBackground );
                      ImGui::Checkbox( "Render grid", &g_WorldRenderer->RenderCoordinateGrid );
                  }

                  if( ImGui::CollapsingHeader( "Camera" ) )
                  {
                      ImGui::SliderFloat( "Exposure", &g_WorldRenderer->Settings.Exposure, 0.1f, 10.0f );
                      ImGui::SliderFloat( "Gamma", &g_WorldRenderer->Settings.Gamma, 0.1f, 4.0f );
                  }
              } );

    auto l_WindowSize = UI::GetRootWindowSize();
    UI::Pane( "##SCENE_COLLECTION", { 15.0f, 330.0f }, { 350.0f, 350.0f },
              [&]()
              {
                  UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( 0.0f, 10.0f ) );
                  if( lastFPS > 0 )
                      UI::Text( fmt::format( "Render: {} fps ({:.2f} ms)", lastFPS, ( 1000.0f / lastFPS ) ).c_str() );
                  else
                      UI::Text( "Render: 0 fps (0 ms)" );
              } );

    ManipulationConfig l_Manipulator{};
    math::ivec2 l_ViewportSize     = g_EngineLoop->GetViewportSize();
    l_Manipulator.Type             = ManipulationType::ROTATION;
    l_Manipulator.Projection       = g_WorldRenderer->View.Projection;
    l_Manipulator.WorldTransform   = g_WorldRenderer->View.View;
    l_Manipulator.ViewportPosition = math::vec2{ 0.0f };
    l_Manipulator.ViewportSize     = { static_cast<float>( l_ViewportSize.x ), static_cast<float>( l_ViewportSize.y ) };

    Manipulate( l_Manipulator, g_SensorTransform );

    return false;
}

int main( int argc, char **argv )
{
    g_EngineLoop = new LTSE::Core::EngineLoop();
    g_EngineLoop->PreInit( 0, nullptr );
    g_EngineLoop->Init();
    Scope g_ComputeScope( 1024 * 1024 );

    g_EngineLoop->RenderDelegate.connect<RenderScene>();
    g_EngineLoop->UIDelegate.connect<RenderUI>();
    g_EngineLoop->UpdateDelegate.connect<Update>( g_ComputeScope );

    // g_SensorModel     = New<TestSensorDevice>();
    g_SensorTransform = math::Translate( math::mat4( 1.0f ), math::vec3( 0.0f, 1.0f, 0.0f ) );

    ParticleRendererCreateInfo l_ParticleRendererCreateInfo{};
    l_ParticleRendererCreateInfo.VertexShader   = "Shaders\\ParticleSystem.vert.spv";
    l_ParticleRendererCreateInfo.FragmentShader = "Shaders\\ParticleSystem.frag.spv";
    l_ParticleRendererCreateInfo.RenderPass     = g_EngineLoop->GetRenderContext().GetRenderPass();

    m_ParticleSystemRenderer = std::make_shared<ParticleSystemRenderer>( g_EngineLoop->GetGraphicContext(), g_EngineLoop->GetRenderContext(), l_ParticleRendererCreateInfo );

    g_World = std::make_shared<LTSE::Core::Scene>( g_EngineLoop->GetGraphicContext(), g_EngineLoop->UIContext() );

    g_WorldRenderer                  = std::make_shared<LTSE::Core::SceneRenderer>( g_World, g_EngineLoop->GetRenderContext(), g_EngineLoop->GetRenderContext().GetRenderPass() );
    math::ivec2 l_ViewportSize       = g_EngineLoop->GetViewportSize();
    g_WorldRenderer->View.Projection = math::Perspective( 90.0_degf, static_cast<float>( l_ViewportSize.x ) / static_cast<float>( l_ViewportSize.y ), 0.01f, 100000.0f );
    g_WorldRenderer->View.Projection[1][1] *= -1.0f;
    g_WorldRenderer->View.CameraPosition = math::vec3( 0.0f, 1.0f, 7.5f );
    g_WorldRenderer->View.ModelFraming   = math::mat4( 0.5f );
    g_WorldRenderer->View.View           = math::Inverse( math::Translate( math::mat4( 1.0f ), g_WorldRenderer->View.CameraPosition ) );

    g_Sphere              = CreateSphere( 64, 64 );
    g_SphereTransforms[0] = math::Scale( math::Translate( math::mat4( 1.0f ), math::vec3( -1.5f, 0.0f, -1.5f ) ), math::vec3( 0.5 ) );
    g_SphereTransforms[1] = math::Translate( math::mat4( 1.0f ), math::vec3( 0.0f, 0.0f, -2.5f ) );
    g_SphereTransforms[2] = math::Scale( math::Translate( math::mat4( 1.0f ), math::vec3( 1.5f, 0.0f, -3.5f ) ), math::vec3( 0.75 ) );

    // {
    //     auto l_MaterialEntity = g_World->LoadAsset( "C:\\work\\materials\\Modern_Metal_Wall\\material.yaml", "Modern_Metal_Wall" );
    //     g_Material.push_back( l_MaterialEntity );
    // }
    // {
    //     auto l_MaterialEntity = g_World->LoadAsset( "C:\\work\\materials\\Layered_Rock\\material.yaml", "Layered_Rock" );
    //     g_Material.push_back( l_MaterialEntity );
    // }

    // auto l_Floor           = g_World->Create( "Floor", g_World->Root );
    // auto &l_FloorMesh      = l_Floor.Add<PrimitiveMeshComponent>();
    // auto &l_FloorTransform = l_Floor.Add<TransformComponent>();
    // l_FloorTransform.T->SetTransformMatrix( { 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f }, { 5.0f, 0.25f, 5.0f } );

    // auto &l_FloorRenderer = l_Floor.Add<RendererComponent>( g_Material[0] );
    // l_FloorMesh.Type      = PrimitiveMeshType::CUBE;
    // l_FloorMesh.Dirty     = true;
    // l_FloorMesh.UpdateBuffers( g_EngineLoop->GetGraphicContext() );

    // auto l_Sphere                                  = g_World->Create( "Sphere", g_World->Root );
    // auto &l_SphereMesh                             = l_Sphere.Add<PrimitiveMeshComponent>();
    // l_SphereMesh.Configuration.SphereConfiguration = { 32, 32 };
    // auto &l_SphereTransform                        = l_Sphere.Add<TransformComponent>();
    // l_SphereTransform.T->SetTransformMatrix( { 0.0f, 1.0f, -2.0f }, { 0.0f, 0.0f, 0.0f }, { .15f, 0.15f, .15f } );
    // auto &l_SphereRenderer = l_Sphere.Add<RendererComponent>( g_Material[1] );
    // l_SphereMesh.Type      = PrimitiveMeshType::SPHERE;
    // l_SphereMesh.Dirty     = true;
    // l_SphereMesh.UpdateBuffers( g_EngineLoop->GetGraphicContext() );
    // g_World->MarkAsRayTracingTarget( l_Sphere );
    LTSE::Logging::Info("--------------------------------------------");
    auto foo = AsyncModelLoader( "C:\\work\\asset_import_test\\Sponza\\asset.yaml" );
    LTSE::Logging::Info("--------------------------------------------");

    math::mat4 lTransform = math::Scale( math::Rotation( 90.0_degf, math::y_axis() ), math::vec3{ 0.01f, 0.01f, 0.01f } );
    g_World->LoadModel(  "C:\\work\\asset_import_test\\Sponza\\asset.yaml", "Sponza", lTransform );

    // math::mat4 lTransform = math::Scale( math::Rotation( 90.0_degf, math::y_axis() ), math::vec3{ 0.01f, 0.01f, 0.01f } );
    // g_World->LoadModel(  "C:\\GitLab\\LTSimulationEngine\\Saved\\City\\asset.yaml", "City", lTransform);

    // g_World->LoadModel("C:\\work\\asset_import_test\\DefaultScene\\asset.yaml", "Default_Scene", math::Scale(math::mat4(1.0f), {1.0f, 1.0f, 1.0f}));
    g_World->ForEach<StaticMeshComponent>( [&]( auto aEntity, auto &aComponent ) { g_World->MarkAsRayTracingTarget( aEntity ); } );
    // g_World->LoadModel("C:\\work\\asset_import_test\\BrainStem\\asset.yaml", "Sponza");
    // g_World->LoadModel("C:\\work\\asset_import_test\\Damaged_Helmet\\asset.yaml", "Sponza");
    // g_World->LoadModel("C:\\work\\asset_import_test\\Traffic_Cone\\asset.yaml", "Sponza", lTransform);
    // g_World->LoadModel("C:\\work\\asset_import_test\\Wooden_Cable_Spinner\\asset.yaml", "Sponza", lTransform);
    // g_World->LoadModel("C:\\work\\asset_import_test\\Garbage_Bags\\asset.yaml", "Sponza", lTransform);
    // g_World->LoadModel("C:\\work\\asset_import_test\\Feudal_Japanese_wall\\asset.yaml", "Sponza", lTransform);

    // Update( g_ComputeScope, 0 );

    while( g_EngineLoop->Tick() )
    {
        if( !foo.IsReady() )
            LTSE::Logging::Info( "Remaining: {}", foo.Progress() );
    }

    return 0;
}
