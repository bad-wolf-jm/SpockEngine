#include "Core/EntityRegistry/Registry.h"

#include "Core/Platform/EngineLoop.h"
#include "Graphics/API/UI/UIContext.h"

#include "Editor/MaterialEditor.h"
#include "Scene/Primitives/Primitives.h"
#include "Scene/Scene.h"

#include "Scene/Renderer/MeshRenderer.h"
#include "Scene/Renderer/SceneRenderer.h"

#include "Scene/Importer/glTFImporter.h"

#include "UI/UI.h"
#include "UI/Widgets.h"

#include <fmt/core.h>

#include "Core/Logging.h"

#include "ImGuizmo.h"

#include "FileTree.h"

using namespace LTSE::Core;
using namespace LTSE;
using namespace LTSE::Core::Primitives;
using namespace math::literals;

LTSE::Core::EngineLoop *g_EngineLoop;

std::shared_ptr<LTSE::Core::Scene> g_World;
std::shared_ptr<LTSE::Core::SceneRenderer> g_WorldRenderer;

uint32_t frameCounter = 0;
float fpsTimer        = 0.0f;
uint32_t lastFPS      = 0;

bool g_Animate = false;
Scene::Element g_Animation;
Scene::Element g_MainAsset;

std::shared_ptr<LTSE::Core::Scene> g_OffScreenWorld;
std::shared_ptr<LTSE::Core::SceneRenderer> g_OffScreenWorldRenderer;
std::shared_ptr<OffscreenRenderTarget> g_OffscreenRenderTarget;

std::shared_ptr<Texture2D> g_TextDisplayTexture;

ImageHandle g_OffscreenViewport;
ImageHandle g_TextDisplayTextureHandle;

VertexBufferData g_Sphere;
math::mat4 g_SphereTransforms[3];

Scene::Element g_Elements[3];

Scene::Element g_SelectedElement;

Scene::Element g_Material;
std::vector<Scene::Element> lAnimations;

float g_TotalTime;
float atime = 0.0f;

void Update( Timestep ts )
{
    frameCounter++;
    fpsTimer += (float)ts;
    if( fpsTimer > 1000.0f )
    {
        lastFPS      = static_cast<uint32_t>( (float)frameCounter * ( 1000.0f / fpsTimer ) );
        fpsTimer     = 0.0f;
        frameCounter = 0;
    }

    if( g_Animate )
    {
        auto &l_AnimationComponent = g_Animation.Get<sAnimationComponent>();

        atime += ( (float)ts / 1000.0f );

        for( auto channel : l_AnimationComponent.mChannels )
        {
            sImportedAnimationSampler &sampler = channel.mInterpolation;

            Scene::Element lChannelTargetNode = channel.mTargetNode;
            if( !lChannelTargetNode.Has<sAnimatedTransformComponent>() )
                lChannelTargetNode.Add<sAnimatedTransformComponent>();

            auto &aT = lChannelTargetNode.Get<sAnimatedTransformComponent>();

            if( sampler.mInputs.size() > sampler.mOutputsVec4.size() )
                continue;

            for( size_t i = 0; i < sampler.mInputs.size() - 1; i++ )
            {
                if( ( atime >= sampler.mInputs[i] ) && ( atime <= sampler.mInputs[i + 1] ) )
                {
                    float u = std::max( 0.0f, atime - sampler.mInputs[i] ) / ( sampler.mInputs[i + 1] - sampler.mInputs[i] );
                    if( u <= 1.0f )
                    {
                        math::vec4 lRotation{};

                        switch( channel.mChannelID )
                        {
                        case sImportedAnimationChannel::Channel::TRANSLATION:
                        {
                            glm::vec4 trans = glm::mix( sampler.mOutputsVec4[i], sampler.mOutputsVec4[i + 1], u );

                            aT.Translation = glm::vec3( trans );
                            break;
                        }
                        case sImportedAnimationChannel::Channel::SCALE:
                        {
                            glm::vec4 trans = glm::mix( sampler.mOutputsVec4[i], sampler.mOutputsVec4[i + 1], u );

                            aT.Scaling = glm::vec3( trans );
                            break;
                        }
                        case sImportedAnimationChannel::Channel::ROTATION:
                        {
                            glm::quat q1;
                            q1.x = sampler.mOutputsVec4[i].x;
                            q1.y = sampler.mOutputsVec4[i].y;
                            q1.z = sampler.mOutputsVec4[i].z;
                            q1.w = sampler.mOutputsVec4[i].w;

                            glm::quat q2;
                            q2.x = sampler.mOutputsVec4[i + 1].x;
                            q2.y = sampler.mOutputsVec4[i + 1].y;
                            q2.z = sampler.mOutputsVec4[i + 1].z;
                            q2.w = sampler.mOutputsVec4[i + 1].w;

                            aT.Rotation = glm::normalize( glm::slerp( q1, q2, u ) );
                            break;
                        }
                        }
                    }
                }
            }
        }
    }

    g_World->Update( 0.0f );
}

static bool EditButton( Scene::Element a_Node, math::vec2 a_Size )
{
    char l_OnLabel[128];
    sprintf( l_OnLabel, "%s##%d", ICON_FA_PENCIL_SQUARE_O, (uint32_t)a_Node );

    ImGui::PushStyleColor( ImGuiCol_Button, ImVec4{ 0.0, 0.0, 0.0, 0.0 } );
    ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ImVec4{ 1.0, 1.0, 1.0, 0.10 } );
    ImGui::PushStyleColor( ImGuiCol_ButtonActive, ImVec4{ 1.0, 1.0, 1.0, 0.20 } );

    bool l_IsVisible;
    bool l_DoEdit = UI::Button( l_OnLabel, a_Size );

    ImGui::PopStyleColor();
    ImGui::PopStyleColor();
    ImGui::PopStyleColor();
    return l_DoEdit;
}

static bool AddChildButton( Scene::Element a_Node, math::vec2 a_Size )
{
    char l_OnLabel[128];
    sprintf( l_OnLabel, "%s##%d", ICON_FA_PLUS, (uint32_t)a_Node );

    ImGui::PushStyleColor( ImGuiCol_Button, ImVec4{ 0.0, 0.0, 0.0, 0.0 } );
    ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ImVec4{ 1.0, 1.0, 1.0, 0.10 } );
    ImGui::PushStyleColor( ImGuiCol_ButtonActive, ImVec4{ 1.0, 1.0, 1.0, 0.20 } );

    bool l_IsVisible;
    bool l_DoEdit = UI::Button( l_OnLabel, a_Size );

    ImGui::PopStyleColor();
    ImGui::PopStyleColor();
    ImGui::PopStyleColor();
    return l_DoEdit;
}

static void DisplayNode( Scene::Element a_Node, float a_Width )
{
    ImGuiTreeNodeFlags l_Flags =
        ImGuiTreeNodeFlags_SpanFullWidth | ImGuiTreeNodeFlags_FramePadding | ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_AllowItemOverlap;
    ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, ImVec2( 3, 3 ) );
    if( a_Node.Has<sRelationshipComponent>() && ( a_Node.Get<sRelationshipComponent>().mChildren.size() != 0 ) )
    {
        auto l_Pos          = UI::GetCurrentCursorPosition();
        std::string l_Label = fmt::format( "##node_foo_{}", (uint32_t)a_Node );
        bool l_NodeIsOpen   = ImGui::TreeNodeEx( l_Label.c_str(), l_Flags );

        UI::SetCursorPosition( l_Pos + math::vec2( 20.0f, 3.0f ) );

        ImGui::PushStyleColor( ImGuiCol_Text, ImVec4{ 0.5f, 0.2f, 0.7f, 1.0f } );
        Text( "{}", ICON_FA_CIRCLE );
        UI::SameLine();
        ImGui::PopStyleColor();

        Text( a_Node.Get<sTag>().mValue );
        UI::SameLine();
        UI::SetCursorPosition( math::vec2( a_Width - 50.0f, UI::GetCurrentCursorPosition().y - 3.0f ) );
        if( EditButton( a_Node, math::vec2{ 20.0, 22.0 } ) )
            g_SelectedElement = a_Node;
        UI::SameLine();
        if( AddChildButton( a_Node, math::vec2{ 20.0, 22.0 } ) )
            g_World->Create( "NEW_ELEMENT", a_Node );

        if( l_NodeIsOpen )
        {
            a_Node.IfExists<sRelationshipComponent>(
                [&]( auto &a_Component )
                {
                    for( auto l_Child : a_Component.mChildren )
                        DisplayNode( l_Child, a_Width );
                } );
            ImGui::TreePop();
        }
    }
    else
    {
        auto l_Pos          = UI::GetCurrentCursorPosition();
        std::string l_Label = fmt::format( "##leaf_foo_{}", (uint32_t)a_Node );
        if( ImGui::Selectable( l_Label.c_str(), false, ImGuiSelectableFlags_AllowItemOverlap, ImVec2{ 0.0f, 0.0f } ) )
        {
            g_SelectedElement = a_Node;
        }
        UI::SetCursorPosition( l_Pos );

        ImGui::PushStyleColor( ImGuiCol_Text, ImVec4{ 0.5f, 0.2f, 0.7f, 1.0f } );
        Text( "{}", ICON_FA_CIRCLE_O );
        UI::SameLine();
        ImGui::PopStyleColor();

        Text( a_Node.Get<sTag>().mValue );
        UI::SameLine();
        UI::SetCursorPosition( math::vec2( a_Width - 50.0f, UI::GetCurrentCursorPosition().y - 3.0f ) );
        if( EditButton( a_Node, math::vec2{ 20.0, 22.0 } ) )
            g_SelectedElement = a_Node;
        UI::SameLine();
        if( AddChildButton( a_Node, math::vec2{ 20.0, 22.0 } ) )
            g_World->Create( "NEW_ELEMENT", a_Node );
    }
    ImGui::PopStyleVar();
}

class MaterialCombo
{
  public:
    std::string ID      = "";
    float ThumbnailSize = 50.0f;
    UI::ComboBox<Entity> Dropdown;

  public:
    MaterialCombo() = default;
    MaterialCombo( std::string a_ID )
        : ID{ a_ID }
        , Dropdown{ UI::ComboBox<Entity>( a_ID ) } {};

    ~MaterialCombo() = default;

    Entity GetValue()
    {
        if( Dropdown.Values.size() > 0 )
            return Dropdown.Values[Dropdown.CurrentItem];
        return Entity{};
    }

    void Display( Entity &a_TextureEntity )
    {
        Dropdown.Labels = { "None" };
        Dropdown.Values = { Entity{} };

        uint32_t n = 1;
        g_World->ForEach<sMaterialShaderComponent>(
            [&]( auto a_Entity, auto &a_Component )
            {
                Dropdown.Labels.push_back( a_Entity.Get<sTag>().mValue );
                Dropdown.Values.push_back( a_Entity );

                if( (uint32_t)a_Entity == (uint32_t)a_TextureEntity )
                    Dropdown.CurrentItem = n;
                n++;
            } );
        Dropdown.Display();

        if( Dropdown.Changed )
        {
            a_TextureEntity = Dropdown.GetValue();
        }
    }
};

template <typename _SliderType> class Slider
{
  public:
    std::string ID = "";

    _SliderType MinValue{};
    _SliderType MaxValue{};
    _SliderType CurrentValue{};

    std::string Format = "";

    bool Changed = false;

    Slider() = default;
    Slider( std::string a_ID )
        : ID{ a_ID }
    {
    }
    ~Slider() = default;

    void Display( _SliderType *mValue ) { Changed = UI::Slider( ID, Format.c_str(), MinValue, MaxValue, mValue ); }
};

template <typename _ValueChooserType> class PropertyEditor
{
  public:
    std::string Label;
    float LabelWidth;
    _ValueChooserType ValueChooser;

    PropertyEditor( std::string ID ) { ValueChooser = _ValueChooserType( ID ); }

    template <typename... _ArgTypes> void Display( _ArgTypes... a_ArgList )
    {
        float l_Width = UI::GetAvailableContentSpace().x;
        ImGui::AlignTextToFramePadding();
        Text( Label );
        UI::SameLine();
        ImGui::SetNextItemWidth( l_Width - LabelWidth );
        UI::SetCursorPosition( math::vec2( LabelWidth, UI::GetCurrentCursorPosition().y ) );
        ValueChooser.Display( std::forward<_ArgTypes>( a_ArgList )... );
    }
};

uint32_t m_ViewportHeight = 1;
uint32_t m_ViewportWidth  = 1;
RenderContext m_ViewportRenderContext{};
Ref<OffscreenRenderTarget> m_OffscreenRenderTarget      = nullptr;
Ref<Graphics::Texture2D> m_OffscreenRenderTargetTexture = nullptr;
ImageHandle m_OffscreenRenderTargetDisplayHandle{};

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
    static float l_CameraZ    = 3.5f;

    ImGuizmo::SetOrthographic( false );
    static bool p_open                           = true;
    constexpr ImGuiDockNodeFlags lDockSpaceFlags = ImGuiDockNodeFlags_PassthruCentralNode;
    constexpr ImGuiWindowFlags lMainwindowFlags  = ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse |
                                                  ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;

    float StatusBarHeight   = 0.0f;
    ImGuiViewport *viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos( viewport->WorkPos );
    ImGui::SetNextWindowSize( viewport->WorkSize - ImVec2( 0.0f, StatusBarHeight ) );
    ImGui::SetNextWindowViewport( viewport->ID );

    ImGui::PushStyleVar( ImGuiStyleVar_WindowRounding, 0.0f );
    ImGui::PushStyleVar( ImGuiStyleVar_WindowBorderSize, 0.0f );
    ImGui::PushStyleVar( ImGuiStyleVar_WindowPadding, ImVec2( 0.0f, 0.0f ) );
    ImGui::Begin( "DockSpace Demo", &p_open, lMainwindowFlags );
    ImGui::PopStyleVar( 3 );

    bool o_RequestQuit = false;
    ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, ImVec2( 0, 8 ) );
    if( ImGui::BeginMainMenuBar() )
    {
        ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, ImVec2( 15, 14 ) );
        // o_RequestQuit = RenderMainMenu();
        ImGui::PopStyleVar();
        ImGui::EndMainMenuBar();
    }
    ImGui::PopStyleVar();

    ImGuiID dockspace_id = ImGui::GetID( "FOO" );
    ImGui::DockSpace( dockspace_id, ImVec2( 0.0f, 0.0f ), lDockSpaceFlags );

    ImGui::Begin( "3D VIEW", &p_open, ImGuiWindowFlags_None );
    {
        if( m_OffscreenRenderTargetDisplayHandle.Handle )
        {
            auto m_WorkspaceAreaSize = UI::GetAvailableContentSpace();

            UI::Image( m_OffscreenRenderTargetDisplayHandle, math::ivec2{ m_WorkspaceAreaSize.x, m_WorkspaceAreaSize.y } );
        }
    }
    ImGui::End();

    ImGui::End();

    ImGui::Begin( "SCENE DISPLAY", &p_open, ImGuiWindowFlags_None );
    {
        math::ivec2 l_WindowSize   = UI::GetAvailableContentSpace();
        ImGuiTreeNodeFlags l_Flags = ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_AllowItemOverlap;
        float l_LabelSize          = 175.0f;

        if( lastFPS > 0 )
            UI::Text( fmt::format( "Render: {} fps ({:.2f} ms)", lastFPS, ( 1000.0f / lastFPS ) ).c_str() );
        else
            UI::Text( "Render: 0 fps (0 ms)" );

        if( ImGui::CollapsingHeader( "Environment", l_Flags ) )
        {
            if( g_World->Environment.Has<sAmbientLightingComponent>() )
            {
                auto &l_AmbientLightComponent = g_World->Environment.Get<sAmbientLightingComponent>();
                UI::ColorChooser( "Ambient light color:", 175, l_AmbientLightComponent.Color );

                Text( "Ambient intensity:" );
                UI::SameLine();
                ImVec2 l_CursorPosition = ImGui::GetCursorPos();
                UI::SetCursorPosition( ImVec2{ l_LabelSize, l_CursorPosition.y } + ImVec2( 0.0f, -5.0f ) );
                ImGui::SetNextItemWidth( l_WindowSize.x - l_LabelSize );
                UI::Slider( "##ambient_intensity", "%.5f", 0.0f, 0.01f, &l_AmbientLightComponent.Intensity );
            }

            if( g_World->Environment.Has<sBackgroundComponent>() )
            {
                auto &l_BackgroundComponent = g_World->Environment.Get<sBackgroundComponent>();
                UI::ColorChooser( "Background color:", 175, l_BackgroundComponent.Color );
            }
        }

        if( ImGui::CollapsingHeader( "Rendering", l_Flags ) )
        {
            ImGui::Checkbox( "Render coordinate grid", &g_WorldRenderer->RenderCoordinateGrid );
        }

        if( ImGui::CollapsingHeader( "Camera", l_Flags ) )
        {
            ImGui::SliderFloat( "Exposure", &g_WorldRenderer->Settings.Exposure, 0.1f, 10.0f );
            ImGui::SliderFloat( "Gamma", &g_WorldRenderer->Settings.Gamma, 0.1f, 4.0f );
        }
    }
    ImGui::End();

    ImGui::Begin( "SCENE NODES", &p_open, ImGuiWindowFlags_None );
    {
        DisplayNode( g_World->Root, UI::GetAvailableContentSpace().x );
    }
    ImGui::End();

    ImGui::Begin( "ITEM PROPERTIES", &p_open, ImGuiWindowFlags_None );
    {
        ImGuiTreeNodeFlags l_Flags = ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_AllowItemOverlap;
        if( g_SelectedElement )
        {
            char buf[128] = { 0 };
            std::strncpy( buf, g_SelectedElement.Get<sTag>().mValue.c_str(), std::min( g_SelectedElement.Get<sTag>().mValue.size(), std::size_t( 128 ) ) );
            if( ImGui::InputText( "##TAG_INPUT", buf, ARRAYSIZE( buf ), ImGuiInputTextFlags_EnterReturnsTrue ) )
            {
                g_SelectedElement.Get<sTag>().mValue = std::string( buf );
            }
            UI::SameLine();
            if( UI::Button( fmt::format( "{} Add component", ICON_FA_PLUS ).c_str(), math::vec2{ 150.0f, 30.0f } ) )
            {
                ImGui::OpenPopup( "##add_component" );
            }
            if( ImGui::BeginPopup( "##add_component" ) )
            {
                if( ImGui::MenuItem( "Transform component", NULL, false, !g_SelectedElement.Has<TransformComponent>() ) )
                {
                    g_SelectedElement.Add<TransformComponent>();
                }
                ImGui::Separator();
                if( ImGui::MenuItem( "Primitive mesh component", NULL, false, !g_SelectedElement.Has<PrimitiveMeshComponent>() ) )
                {
                    auto &l_NewPrimitiveComponent = g_SelectedElement.Add<PrimitiveMeshComponent>();
                    l_NewPrimitiveComponent.Dirty = true;
                    l_NewPrimitiveComponent.UpdateBuffers( g_EngineLoop->GetGraphicContext() );
                }
                if( ImGui::MenuItem( "Static mesh component", NULL, false, !g_SelectedElement.Has<sStaticMeshComponent>() ) )
                {
                    g_SelectedElement.Add<sStaticMeshComponent>();
                }
                ImGui::Separator();
                if( ImGui::MenuItem( "Directional light component", NULL, false, !g_SelectedElement.Has<sDirectionalLightComponent>() ) )
                {
                    g_SelectedElement.Add<sDirectionalLightComponent>();
                }
                if( ImGui::MenuItem( "Point light component", NULL, false, !g_SelectedElement.Has<sPointLightComponent>() ) )
                {
                    g_SelectedElement.Add<sPointLightComponent>();
                }
                if( ImGui::MenuItem( "spotlight component", NULL, false, !g_SelectedElement.Has<sSpotlightComponent>() ) )
                {
                    g_SelectedElement.Add<sSpotlightComponent>();
                }
                ImGui::Separator();
                if( ImGui::MenuItem( "Particle system", NULL, false, !g_SelectedElement.Has<sParticleSystemComponent>() ) )
                {
                    g_SelectedElement.Add<sParticleSystemComponent>();

                    auto &l_ParticleShaderConfiguration     = g_SelectedElement.Add<sParticleShaderComponent>();
                    l_ParticleShaderConfiguration.LineWidth = 1.0f;

                    ParticleRendererCreateInfo l_RendererCreateInfo{};
                    l_RendererCreateInfo.LineWidth = l_ParticleShaderConfiguration.LineWidth;
                }
                ImGui::EndPopup();
            }

            if( ImGui::CollapsingHeader( "Ray tracing", l_Flags ) )
            {
                bool l_RT = g_SelectedElement.Has<sRayTracingTargetComponent>();
                ImGui::Checkbox( "Mark as ray tracing target", &l_RT );
                if( l_RT )
                    g_World->MarkAsRayTracingTarget( g_SelectedElement );
                else
                    g_SelectedElement.TryRemove<sRayTracingTargetComponent>();
            }

            if( ImGui::CollapsingHeader( "Hierarchy", l_Flags ) )
            {
                g_SelectedElement.IfExists<sRelationshipComponent>(
                    [&]( auto &l_Component ) {

                    } );
            }

            if( ImGui::CollapsingHeader( "Transform", l_Flags ) )
            {
                g_SelectedElement.IfExists<TransformComponent>(
                    [&]( auto &l_Component )
                    {
                        math::vec3 l_Position = l_Component.T->GetTranslation();
                        UI::VectorComponentEditor( "Position:", l_Position, 0.0, 100 );

                        math::vec3 l_Rotation = l_Component.T->GetEulerRotation();
                        UI::VectorComponentEditor( "Rotation:", l_Rotation, 0.0, 100 );

                        math::vec3 l_Scale = l_Component.T->GetScale();
                        UI::VectorComponentEditor( "Scale:", l_Scale, 1.0, 100 );

                        l_Component.T->SetTransformMatrix( l_Position, l_Rotation, l_Scale );
                    } );
            }

            if( ImGui::CollapsingHeader( "Mesh", l_Flags ) )
            {
                float l_LabelSize = 175.0f;

                static UI::ComboBox<PrimitiveMeshType> l_PrimitiveChooser( "##combo_primitive_chooser" );
                l_PrimitiveChooser.Labels = { "Cube", "Plane", "Sphere", "Cylinder", "Cone", "Disk" };
                l_PrimitiveChooser.Values = { PrimitiveMeshType::CUBE,     PrimitiveMeshType::PLANE, PrimitiveMeshType::SPHERE,
                                              PrimitiveMeshType::CYLINDER, PrimitiveMeshType::CONE,  PrimitiveMeshType::DISK };

                static PropertyEditor<::Slider<int32_t>> l_SubdivisionX( "##subdivision_x" );
                l_SubdivisionX.Label                 = "X Subdivisions:";
                l_SubdivisionX.LabelWidth            = l_LabelSize;
                l_SubdivisionX.ValueChooser.MinValue = 3;
                l_SubdivisionX.ValueChooser.MaxValue = 128;
                l_SubdivisionX.ValueChooser.Format   = "%d";

                static PropertyEditor<::Slider<int32_t>> l_SubdivisionY( "##subdivision_y" );
                l_SubdivisionY.Label                 = "Y Subdivisions:";
                l_SubdivisionY.LabelWidth            = l_LabelSize;
                l_SubdivisionY.ValueChooser.MinValue = 3;
                l_SubdivisionY.ValueChooser.MaxValue = 128;
                l_SubdivisionY.ValueChooser.Format   = "%d";

                static PropertyEditor<::Slider<uint32_t>> l_Rings( "##rings" );
                l_Rings.Label                 = "Rings:";
                l_Rings.LabelWidth            = l_LabelSize;
                l_Rings.ValueChooser.MinValue = 3;
                l_Rings.ValueChooser.MaxValue = 128;
                l_Rings.ValueChooser.Format   = "%d";

                static PropertyEditor<::Slider<uint32_t>> l_Segments( "##segments" );
                l_Segments.Label                 = "Segments:";
                l_Segments.LabelWidth            = l_LabelSize;
                l_Segments.ValueChooser.MinValue = 3;
                l_Segments.ValueChooser.MaxValue = 128;
                l_Segments.ValueChooser.Format   = "%d";

                g_SelectedElement.IfExists<sStaticMeshComponent>(
                    [&]( auto &l_Component )
                    {
                        Text( "Static mesh" );
                        Text( "{}", l_Component.Name );
                    } );

                if( g_SelectedElement.Has<PrimitiveMeshComponent>() )
                {
                    bool l_MeshConfigChanged = false;
                    auto &l_Component        = g_SelectedElement.Get<PrimitiveMeshComponent>();
                    Text( "Primitive mesh" );
                    l_PrimitiveChooser.Display();

                    switch( l_PrimitiveChooser.GetValue() )
                    {
                    case PrimitiveMeshType::CUBE:
                        if( l_Component.Type != PrimitiveMeshType::CUBE )
                            l_MeshConfigChanged = true;
                        l_Component.Type = PrimitiveMeshType::CUBE;
                        break;
                    case PrimitiveMeshType::PLANE:
                    {
                        if( l_Component.Type != PrimitiveMeshType::PLANE )
                            l_MeshConfigChanged = true;
                        l_Component.Type         = PrimitiveMeshType::PLANE;
                        float l_LabelSize        = 175.0f;
                        math::ivec2 l_WindowSize = UI::GetAvailableContentSpace();
                        ImVec2 l_CursorPosition  = ImGui::GetCursorPos();

                        if( l_MeshConfigChanged )
                            l_Component.Configuration.PlaneConfiguration.Subdivisions = math::ivec2{ 3, 3 };

                        l_SubdivisionX.Display( &l_Component.Configuration.PlaneConfiguration.Subdivisions.x );
                        if( l_SubdivisionX.ValueChooser.Changed )
                            l_MeshConfigChanged = true;

                        l_SubdivisionY.Display( &l_Component.Configuration.PlaneConfiguration.Subdivisions.y );
                        if( l_SubdivisionY.ValueChooser.Changed )
                            l_MeshConfigChanged = true;
                    }
                    break;
                    case PrimitiveMeshType::SPHERE:
                    {
                        if( l_Component.Type != PrimitiveMeshType::SPHERE )
                            l_MeshConfigChanged = true;
                        l_Component.Type         = PrimitiveMeshType::SPHERE;
                        float l_LabelSize        = 175.0f;
                        math::ivec2 l_WindowSize = UI::GetAvailableContentSpace();
                        ImVec2 l_CursorPosition  = ImGui::GetCursorPos();
                        if( l_MeshConfigChanged )
                        {
                            l_Component.Configuration.SphereConfiguration.Rings    = 32;
                            l_Component.Configuration.SphereConfiguration.Segments = 32;
                        }

                        l_Rings.Display( &l_Component.Configuration.SphereConfiguration.Rings );
                        if( l_Rings.ValueChooser.Changed )
                            l_MeshConfigChanged = true;

                        l_Segments.Display( &l_Component.Configuration.SphereConfiguration.Segments );
                        if( l_Segments.ValueChooser.Changed )
                            l_MeshConfigChanged = true;
                    }
                    break;
                    case PrimitiveMeshType::CYLINDER:
                    {
                        if( l_Component.Type != PrimitiveMeshType::CYLINDER )
                            l_MeshConfigChanged = true;

                        l_Component.Type         = PrimitiveMeshType::CYLINDER;
                        float l_LabelSize        = 175.0f;
                        math::ivec2 l_WindowSize = UI::GetAvailableContentSpace();
                        ImVec2 l_CursorPosition  = ImGui::GetCursorPos();
                        if( l_MeshConfigChanged )
                        {
                            l_Component.Configuration.CylinderConfiguration.Rings    = 32;
                            l_Component.Configuration.CylinderConfiguration.Segments = 32;
                        }

                        l_Rings.Display( &l_Component.Configuration.CylinderConfiguration.Rings );
                        if( l_Rings.ValueChooser.Changed )
                            l_MeshConfigChanged = true;

                        l_Segments.Display( &l_Component.Configuration.CylinderConfiguration.Segments );
                        if( l_Segments.ValueChooser.Changed )
                            l_MeshConfigChanged = true;
                    }
                    break;
                    case PrimitiveMeshType::CONE:
                    {
                        if( l_Component.Type != PrimitiveMeshType::CONE )
                            l_MeshConfigChanged = true;
                        l_Component.Type         = PrimitiveMeshType::CONE;
                        float l_LabelSize        = 175.0f;
                        math::ivec2 l_WindowSize = UI::GetAvailableContentSpace();
                        if( l_MeshConfigChanged )
                        {
                            l_Component.Configuration.ConeConfiguration.Segments = 32;
                        }

                        l_Segments.Display( &l_Component.Configuration.ConeConfiguration.Segments );
                        if( l_Segments.ValueChooser.Changed )
                            l_MeshConfigChanged = true;
                    }
                    break;
                    case PrimitiveMeshType::DISK:
                    {
                        if( l_Component.Type != PrimitiveMeshType::DISK )
                            l_MeshConfigChanged = true;
                        l_Component.Type         = PrimitiveMeshType::DISK;
                        float l_LabelSize        = 175.0f;
                        math::ivec2 l_WindowSize = UI::GetAvailableContentSpace();
                        if( l_MeshConfigChanged )
                        {
                            l_Component.Configuration.DiskConfiguration.Segments = 32;
                        }

                        l_Segments.Display( &l_Component.Configuration.DiskConfiguration.Segments );
                        if( l_Segments.ValueChooser.Changed )
                            l_MeshConfigChanged = true;
                    }
                    break;
                    default:
                        break;
                    }
                    l_Component.Dirty = l_MeshConfigChanged;
                    g_SelectedElement.Replace<PrimitiveMeshComponent>( l_Component );
                }
            }

            if( ImGui::CollapsingHeader( "Material", l_Flags ) )
            {
                static MaterialCombo l_MaterialChooser( "##material_chooser" );
                if( g_SelectedElement.Has<RendererComponent>() )
                {
                    l_MaterialChooser.Display( g_SelectedElement.Get<RendererComponent>().Material );
                }
                else
                {
                    auto l_NewRenderer = Entity{};
                    l_MaterialChooser.Display( l_NewRenderer );
                    if( l_NewRenderer )
                        g_SelectedElement.Add<RendererComponent>( l_NewRenderer );
                }
            }

            if( ImGui::CollapsingHeader( "Light", l_Flags ) )
            {
                float l_LabelSize = 175.0f;

                math::ivec2 l_WindowSize = UI::GetAvailableContentSpace();

                static PropertyEditor<::Slider<float>> l_AzimuthEditor( "##azimuth" );
                l_AzimuthEditor.Label                 = "Azimuth:";
                l_AzimuthEditor.LabelWidth            = l_LabelSize;
                l_AzimuthEditor.ValueChooser.MinValue = 0.0f;
                l_AzimuthEditor.ValueChooser.MaxValue = 360.0f;
                l_AzimuthEditor.ValueChooser.Format   = "%.2f";

                static PropertyEditor<::Slider<float>> l_ElevationEditor( "##elevation" );
                l_ElevationEditor.Label               = "Elevation:";
                l_ElevationEditor.LabelWidth          = l_LabelSize;
                l_ElevationEditor.ValueChooser.Format = "%.2f";

                static PropertyEditor<::Slider<float>> l_ConeWidthEditor( "##cone_width" );
                l_ConeWidthEditor.Label                 = "Cone width:";
                l_ConeWidthEditor.ValueChooser.MinValue = 0.0f;
                l_ConeWidthEditor.ValueChooser.MaxValue = 180.0f;
                l_ConeWidthEditor.LabelWidth            = l_LabelSize;
                l_ConeWidthEditor.ValueChooser.Format   = "%.2f";

                static PropertyEditor<::Slider<float>> l_IntensityEditor( "##intensity" );
                l_IntensityEditor.Label                 = "Intensity:";
                l_IntensityEditor.LabelWidth            = l_LabelSize;
                l_IntensityEditor.ValueChooser.MinValue = 0.0f;
                l_IntensityEditor.ValueChooser.MaxValue = 50.0f;
                l_IntensityEditor.ValueChooser.Format   = "%.2f";

                if( g_SelectedElement.Has<sDirectionalLightComponent>() )
                {
                    auto &l_LightComponent = g_SelectedElement.Get<sDirectionalLightComponent>();

                    l_AzimuthEditor.Display( &l_LightComponent.Azimuth );

                    l_ElevationEditor.ValueChooser.MinValue = -90.0f;
                    l_ElevationEditor.ValueChooser.MaxValue = 90.0f;
                    l_ElevationEditor.Display( &l_LightComponent.Elevation );

                    UI::ColorChooser( "Color:", 125, l_LightComponent.Color );

                    l_IntensityEditor.Display( &l_LightComponent.Intensity );
                }

                if( g_SelectedElement.Has<sPointLightComponent>() )
                {
                    auto &l_LightComponent = g_SelectedElement.Get<sPointLightComponent>();
                    UI::VectorComponentEditor( "Position:", l_LightComponent.Position, 0.0, 100 );
                    UI::ColorChooser( "Color:", 125, l_LightComponent.Color );

                    l_IntensityEditor.Display( &l_LightComponent.Intensity );
                }

                if( g_SelectedElement.Has<sSpotlightComponent>() )
                {
                    auto &l_LightComponent = g_SelectedElement.Get<sSpotlightComponent>();

                    UI::VectorComponentEditor( "Position:", l_LightComponent.Position, 0.0, 100 );

                    l_AzimuthEditor.Display( &l_LightComponent.Azimuth );

                    l_ElevationEditor.ValueChooser.MinValue = 0.0f;
                    l_ElevationEditor.ValueChooser.MaxValue = 360.0f;
                    l_ElevationEditor.Display( &l_LightComponent.Elevation );

                    UI::ColorChooser( "Color:", 125, l_LightComponent.Color );

                    l_IntensityEditor.Display( &l_LightComponent.Intensity );
                    l_ConeWidthEditor.Display( &l_LightComponent.Cone );
                }
            }

            if( ImGui::CollapsingHeader( "Skeleton", l_Flags ) )
            {
                g_SelectedElement.IfExists<sSkeletonComponent>(
                    [&]( auto &l_Component ) {

                    } );
            }

            if( ImGui::CollapsingHeader( "Animation", l_Flags ) )
            {
                g_SelectedElement.IfExists<sAnimationComponent>( [&]( auto &l_Component ) { Text( "sAnimationComponent" ); } );

                g_SelectedElement.IfExists<AnimationChooser>( [&]( auto &l_Component ) { Text( "AnimationChooser" ); } );
            }
        }
    }
    ImGui::End();

    ImGui::Begin( "TEXTURES LIST", &p_open, ImGuiWindowFlags_None );
    {
    }
    ImGui::End();

    ImGui::Begin( "MATERIALS_LIST", &p_open, ImGuiWindowFlags_None );
    {
        Text( "Materials" );
        static bool IsSelected = false;
        g_World->ForEach<sMaterialShaderComponent>(
            [&]( auto a_Entity, auto &a_Component )
            {
                if( ImGui ::Selectable( a_Entity.Get<sTag>().mValue.c_str() ) )
                {
                    g_Material = a_Entity;
                };
            } );
    }
    ImGui::End();

    ImGui::Begin( "MATERIAL_EDITOR", &p_open, ImGuiWindowFlags_None );
    {
        auto l_WindowPropertiesSize = UI::GetAvailableContentSpace();

        Editor::MaterialEditor lME{};
        lME.ElementToEdit = g_Material;
        lME.World         = g_World;
        lME.Display( l_WindowPropertiesSize.x, l_WindowPropertiesSize.y );
    }
    ImGui::End();

    // ManipulationConfig l_Manipulator{};
    // math::vec2 l_ViewportSize      = g_EngineLoop->GetViewportSize();
    // l_Manipulator.Type             = ManipulationType::TRANSLATION;
    // l_Manipulator.Projection       = g_WorldRenderer->View.Projection;
    // l_Manipulator.WorldTransform   = g_WorldRenderer->View.View;
    // l_Manipulator.ViewportPosition = math::vec2{ 0.0f };
    // l_Manipulator.ViewportSize     = g_EngineLoop->GetViewportSize();

    // auto l_ElementToMove = Entity{};
    // if( l_ElementToMove.Has<TransformMatrixComponent>() )
    // {
    //     math::mat4 l_ParentTransform = math::mat4( 1.0f );
    //     auto l_Transform             = g_Elements[1].Get<TransformMatrixComponent>().Matrix;
    //     if( l_ElementToMove.Has<sRelationshipComponent>() )
    //     {
    //         if( l_ElementToMove.Get<sRelationshipComponent>().mParent.Has<TransformMatrixComponent>() )
    //         {
    //             l_ParentTransform = l_ElementToMove.Get<sRelationshipComponent>().mParent.Get<TransformMatrixComponent>().Matrix;
    //         }
    //     }

    //     Manipulate( l_Manipulator, l_Transform );
    //     l_ElementToMove.Get<TransformComponent>().T->SetTransformMatrix( math::Inverse( l_ParentTransform ) * l_Transform );
    // }

    return false;
}

void RenderScene()
{
    m_ViewportRenderContext.BeginRender();

    if( m_ViewportRenderContext )
        g_WorldRenderer->Render( m_ViewportRenderContext );

    m_ViewportRenderContext.EndRender();

    // g_World->Update( 0.0f );
    // g_WorldRenderer->Render( g_EngineLoop->GetSwapchainRenderer()->GetCurrentCommandBuffer() );
}

void RebuildOutputFramebuffer()
{
    if( m_ViewportWidth == 0 || m_ViewportHeight == 0 )
        return;

    if( !m_OffscreenRenderTarget )
    {
        OffscreenRenderTargetDescription l_RenderTargetCI{};
        l_RenderTargetCI.OutputSize  = { m_ViewportWidth, m_ViewportHeight };
        l_RenderTargetCI.SampleCount = 4;
        l_RenderTargetCI.Sampled     = true;
        m_OffscreenRenderTarget      = New<OffscreenRenderTarget>( g_EngineLoop->GetGraphicContext(), l_RenderTargetCI );
        m_ViewportRenderContext      = LTSE::Graphics::RenderContext( g_EngineLoop->GetGraphicContext(), m_OffscreenRenderTarget );
    }
    else
    {
        m_OffscreenRenderTarget->Resize( m_ViewportWidth, m_ViewportHeight );
    }

    m_OffscreenRenderTargetTexture = New<Graphics::Texture2D>( g_EngineLoop->GetGraphicContext(), TextureDescription{}, m_OffscreenRenderTarget->GetOutputImage() );

    if( !m_OffscreenRenderTargetDisplayHandle.Handle )
    {
        m_OffscreenRenderTargetDisplayHandle = g_EngineLoop->UIContext()->CreateTextureHandle( m_OffscreenRenderTargetTexture );
    }
    else
    {
        m_OffscreenRenderTargetDisplayHandle.Handle->Write( m_OffscreenRenderTargetTexture, 0 );
    }

    if( g_WorldRenderer )
    {
        math::ivec2 l_ViewportSize       = { m_ViewportWidth, m_ViewportHeight };
        g_WorldRenderer->View.Projection = math::Perspective( 90.0_degf, static_cast<float>( l_ViewportSize.x ) / static_cast<float>( l_ViewportSize.y ), 0.01f, 100000.0f );
        g_WorldRenderer->View.Projection[1][1] *= -1.0f;
        g_WorldRenderer->View.View = math::Inverse( math::Translate( math::mat4( 1.0f ), g_WorldRenderer->View.CameraPosition ) );
    }
}

int main( int argc, char **argv )
{
    g_EngineLoop = new LTSE::Core::EngineLoop();
    g_EngineLoop->PreInit( 0, nullptr );
    g_EngineLoop->Init();

    g_EngineLoop->RenderDelegate.connect<RenderScene>();
    g_EngineLoop->UIDelegate.connect<RenderUI>();
    g_EngineLoop->UpdateDelegate.connect<Update>();

    m_ViewportWidth  = 1400.0f;
    m_ViewportHeight = 900.0f;

    RebuildOutputFramebuffer();
    g_World         = New<Scene>( g_EngineLoop->GetGraphicContext(), g_EngineLoop->UIContext() );
    g_WorldRenderer = New<SceneRenderer>( g_World, m_ViewportRenderContext, m_OffscreenRenderTarget->GetRenderPass() );

    math::vec2 l_ViewportSize = g_EngineLoop->GetViewportSize();

    sCameraComponent l_CameraConfiguration;
    l_CameraConfiguration.Position    = math::vec3( 0.0f, 1.5f, 3.5f );
    l_CameraConfiguration.Near        = 0.01f;
    l_CameraConfiguration.Far         = 1000.0f;
    l_CameraConfiguration.AspectRatio = m_ViewportWidth / m_ViewportHeight;
    l_CameraConfiguration.FieldOfView = 90.0f;

    auto l_Camera = g_World->CreateEntity( "Camera" );
    l_Camera.Add<sCameraComponent>( l_CameraConfiguration );
    g_World->CurrentCamera = l_Camera;

    g_WorldRenderer->View.Projection     = g_World->GetProjection();
    g_WorldRenderer->View.CameraPosition = g_World->GetCameraPosition();
    g_WorldRenderer->View.ModelFraming   = math::mat4( 0.5f );
    g_WorldRenderer->View.View           = g_World->GetView();

    auto lImporter         = New<GlTFImporter>( "C:\\work\\assets\\glTF-Sample-Models\\2.0\\Sponza\\glTF\\Sponza.gltf" );
    math::mat4 l_Transform = math::Rotation( 79.0_degf, math::vec3( 0.0f, 1.f, 0.0f ) );
    g_World->LoadModel( lImporter, l_Transform );

    g_World->AttachScript( g_World->Root, "C:\\GitLab\\LTSimulationEngine\\Tools\\SceneCreator\\Source\\TestScript.lua" );

    // auto lImporter         = New<GlTFImporter>( "C:\\work\\assets\\glTF-Sample-Models\\2.0\\BrainStem\\glTF\\BrainStem.gltf" );
    // math::mat4 l_Transform = math::Rotation( 0.0_degf, math::vec3( 1.0f, 0.0f, 0.0f ) );
    // g_World->LoadModel( lImporter, l_Transform );

    // auto lImporter         = New<GlTFImporter>( "C:\\work\\assets\\glTF-Sample-Models\\2.0\\CesiumMan\\glTF\\CesiumMan.gltf" );
    // math::mat4 l_Transform = math::Rotation( 0.0_degf, math::vec3( 1.0f, 0.0f, 0.0f ) );
    // g_World->LoadModel( lImporter, l_Transform );

    g_World->ForEach<sAnimationComponent>( [&]( auto e, auto &c ) { lAnimations.push_back( e ); } );

    if( lAnimations.size() > 0 )
    {
        LTSE::Logging::Info( "ANIMATION" );
        g_Animate   = true;
        g_Animation = lAnimations[0];
    }

    g_World->BeginScenario();

    // auto lImporter = New<GlTFImporter>( "C:\\work\\assets\\glTF-Sample-Models\\2.0\\DamagedHelmet\\glTF\\DamagedHelmet.gltf" );
    // g_World->LoadModel( lImporter, math::mat4(1.0f) );

    while( g_EngineLoop->Tick() )
    {
    }

    g_World->EndScenario();

    return 0;
}
