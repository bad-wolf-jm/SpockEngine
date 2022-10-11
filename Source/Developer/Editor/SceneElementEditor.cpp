#include "SceneElementEditor.h"

#include "Developer/UI/CanvasView.h"
#include "Developer/UI/UI.h"
#include "Developer/UI/Widgets.h"

#include "Developer/Scene/Components.h"

#include "Developer/Scene/Components/VisualHelpers.h"

using namespace LTSE::Core::EntityComponentSystem::Components;

namespace LTSE::Editor
{

    class MaterialCombo
    {
      public:
        Ref<Scene> World;
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
            World->ForEach<MaterialShaderComponent>(
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

        void Display( _SliderType *Value ) { Changed = UI::Slider( ID, Format.c_str(), MinValue, MaxValue, Value ); }
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
            UI::Text( Label );
            UI::SameLine();
            ImGui::SetNextItemWidth( l_Width - LabelWidth );
            UI::SetCursorPosition( math::vec2( LabelWidth, UI::GetCurrentCursorPosition().y ) );
            ValueChooser.Display( std::forward<_ArgTypes>( a_ArgList )... );
        }
    };

    static bool EditComponent( StaticMeshComponent &a_Component )
    {
        UI::Text( "Static mesh" );
        UI::Text( "{}", a_Component.Name );
        return false;
    }

    static bool EditComponent( LocalTransformComponent &a_Component )
    {
        math::vec3 l_Position = a_Component.GetTranslation();
        math::vec3 l_Rotation = a_Component.GetEulerRotation();
        math::vec3 l_Scale    = a_Component.GetScale();

        UI::VectorComponentEditor( "Position:", l_Position, 0.0, 100 );
        UI::VectorComponentEditor( "Rotation:", l_Rotation, 0.0, 100 );
        UI::VectorComponentEditor( "Scale:", l_Scale, 1.0, 100 );

        a_Component.mMatrix = LocalTransformComponent( l_Position, l_Rotation, l_Scale ).mMatrix;
        return false;
    }

    static bool EditComponent( DirectionalLightComponent &a_Component )
    {
        float l_LabelSize = 175.0f;
        static PropertyEditor<Slider<float>> l_AzimuthEditor( "##azimuth" );
        l_AzimuthEditor.Label                 = "Azimuth:";
        l_AzimuthEditor.LabelWidth            = l_LabelSize;
        l_AzimuthEditor.ValueChooser.MinValue = 0.0f;
        l_AzimuthEditor.ValueChooser.MaxValue = 360.0f;
        l_AzimuthEditor.ValueChooser.Format   = "%.2f";

        static PropertyEditor<Slider<float>> l_ElevationEditor( "##elevation" );
        l_ElevationEditor.Label               = "Elevation:";
        l_ElevationEditor.LabelWidth          = l_LabelSize;
        l_ElevationEditor.ValueChooser.Format = "%.2f";

        static PropertyEditor<Slider<float>> l_IntensityEditor( "##intensity" );
        l_IntensityEditor.Label                 = "Intensity:";
        l_IntensityEditor.LabelWidth            = l_LabelSize;
        l_IntensityEditor.ValueChooser.MinValue = 0.0f;
        l_IntensityEditor.ValueChooser.MaxValue = 50.0f;
        l_IntensityEditor.ValueChooser.Format   = "%.2f";

        l_AzimuthEditor.Display( &a_Component.Azimuth );

        l_ElevationEditor.ValueChooser.MinValue = -90.0f;
        l_ElevationEditor.ValueChooser.MaxValue = 90.0f;
        l_ElevationEditor.Display( &a_Component.Elevation );

        UI::ColorChooser( "Color:", 125, a_Component.Color );

        l_IntensityEditor.Display( &a_Component.Intensity );
        return false;
    }

    static bool EditComponent( PointLightComponent &a_Component )
    {
        float l_LabelSize = 175.0f;

        static PropertyEditor<Slider<float>> l_IntensityEditor( "##intensity" );
        l_IntensityEditor.Label                 = "Intensity:";
        l_IntensityEditor.LabelWidth            = l_LabelSize;
        l_IntensityEditor.ValueChooser.MinValue = 0.0f;
        l_IntensityEditor.ValueChooser.MaxValue = 50.0f;
        l_IntensityEditor.ValueChooser.Format   = "%.2f";

        UI::VectorComponentEditor( "Position:", a_Component.Position, 0.0, 100 );
        UI::ColorChooser( "Color:", 125, a_Component.Color );

        l_IntensityEditor.Display( &a_Component.Intensity );
        return false;
    }

    static bool EditComponent( SpotlightComponent &a_Component )
    {
        float l_LabelSize = 175.0f;

        static PropertyEditor<Slider<float>> l_AzimuthEditor( "##azimuth" );
        l_AzimuthEditor.Label                 = "Azimuth:";
        l_AzimuthEditor.LabelWidth            = l_LabelSize;
        l_AzimuthEditor.ValueChooser.MinValue = 0.0f;
        l_AzimuthEditor.ValueChooser.MaxValue = 360.0f;
        l_AzimuthEditor.ValueChooser.Format   = "%.2f";

        static PropertyEditor<Slider<float>> l_ElevationEditor( "##elevation" );
        l_ElevationEditor.Label               = "Elevation:";
        l_ElevationEditor.LabelWidth          = l_LabelSize;
        l_ElevationEditor.ValueChooser.Format = "%.2f";

        static PropertyEditor<Slider<float>> l_ConeWidthEditor( "##cone_width" );
        l_ConeWidthEditor.Label                 = "Cone width:";
        l_ConeWidthEditor.ValueChooser.MinValue = 0.0f;
        l_ConeWidthEditor.ValueChooser.MaxValue = 180.0f;
        l_ConeWidthEditor.LabelWidth            = l_LabelSize;
        l_ConeWidthEditor.ValueChooser.Format   = "%.2f";

        static PropertyEditor<Slider<float>> l_IntensityEditor( "##intensity" );
        l_IntensityEditor.Label                 = "Intensity:";
        l_IntensityEditor.LabelWidth            = l_LabelSize;
        l_IntensityEditor.ValueChooser.MinValue = 0.0f;
        l_IntensityEditor.ValueChooser.MaxValue = 50.0f;
        l_IntensityEditor.ValueChooser.Format   = "%.2f";

        UI::VectorComponentEditor( "Position:", a_Component.Position, 0.0, 100 );

        l_AzimuthEditor.Display( &a_Component.Azimuth );

        l_ElevationEditor.ValueChooser.MinValue = 0.0f;
        l_ElevationEditor.ValueChooser.MaxValue = 360.0f;
        l_ElevationEditor.Display( &a_Component.Elevation );

        UI::ColorChooser( "Color:", 125, a_Component.Color );

        l_IntensityEditor.Display( &a_Component.Intensity );
        l_ConeWidthEditor.Display( &a_Component.Cone );
        return false;
    }

    static bool EditComponent( LTSE::Graphics::GraphicContext aGraphicContext, LightComponent &a_Component )
    {
        static UI::ComboBox<LightType> l_PrimitiveChooser( "##combo_light_type_chooser" );
        l_PrimitiveChooser.Labels = { "Directional light", "Point light", "Spotlight" };
        l_PrimitiveChooser.Values = { LightType::DIRECTIONAL, LightType::POINT_LIGHT, LightType::SPOTLIGHT };

        l_PrimitiveChooser.Display();

        switch( l_PrimitiveChooser.GetValue() )
        {
        case LightType::POINT_LIGHT:
        {
            if( l_PrimitiveChooser.Changed )
            {
                a_Component.Light.TryRemove<SpotlightComponent>();
                a_Component.Light.TryRemove<SpotlightHelperComponent>();
                a_Component.Light.TryRemove<DirectionalLightComponent>();
                a_Component.Light.TryRemove<DirectionalLightHelperComponent>();
            }
            if( !a_Component.Light.Has<PointLightComponent>() )
            {
                a_Component.Light.Add<PointLightComponent>();

                PointLightHelperComponent &l_VisualizerComponent = a_Component.Light.Add<PointLightHelperComponent>();
                l_VisualizerComponent.LightData                  = a_Component.Light.Get<PointLightComponent>();
                l_VisualizerComponent.UpdateMesh( aGraphicContext );
            }
            a_Component.Light.Get<PointLightHelperComponent>().LightData = a_Component.Light.Get<PointLightComponent>();
            EditComponent( a_Component.Light.Get<PointLightComponent>() );
        }
        break;
        case LightType::SPOTLIGHT:
        {
            if( l_PrimitiveChooser.Changed )
            {
                a_Component.Light.TryRemove<PointLightComponent>();
                a_Component.Light.TryRemove<PointLightHelperComponent>();
                a_Component.Light.TryRemove<DirectionalLightComponent>();
                a_Component.Light.TryRemove<DirectionalLightHelperComponent>();
            }
            if( !a_Component.Light.Has<SpotlightComponent>() )
            {
                a_Component.Light.Add<SpotlightComponent>();

                SpotlightHelperComponent &l_VisualizerComponent = a_Component.Light.Add<SpotlightHelperComponent>();
                l_VisualizerComponent.LightData                 = a_Component.Light.Get<SpotlightComponent>();
                l_VisualizerComponent.UpdateMesh( aGraphicContext );
            }
            a_Component.Light.Get<SpotlightHelperComponent>().LightData = a_Component.Light.Get<SpotlightComponent>();
            EditComponent( a_Component.Light.Get<SpotlightComponent>() );
        }
        break;
        case LightType::DIRECTIONAL:
        default:
        {
            if( l_PrimitiveChooser.Changed )
            {
                a_Component.Light.TryRemove<PointLightComponent>();
                a_Component.Light.TryRemove<PointLightHelperComponent>();
                a_Component.Light.TryRemove<SpotlightComponent>();
                a_Component.Light.TryRemove<SpotlightHelperComponent>();
            }
            if( !a_Component.Light.Has<DirectionalLightComponent>() )
            {
                a_Component.Light.Add<DirectionalLightComponent>();

                DirectionalLightHelperComponent &l_VisualizerComponent = a_Component.Light.Add<DirectionalLightHelperComponent>();
                l_VisualizerComponent.LightData                        = a_Component.Light.Get<DirectionalLightComponent>();
                l_VisualizerComponent.UpdateMesh( aGraphicContext );
            }
            a_Component.Light.Get<DirectionalLightHelperComponent>().LightData = a_Component.Light.Get<DirectionalLightComponent>();
            EditComponent( a_Component.Light.Get<DirectionalLightComponent>() );
        }
        break;
        };

        return false;
    }

    SceneElementEditor::SceneElementEditor( GraphicContext &aGraphicContext )
        : mGraphicContext{ aGraphicContext } {};

    void SceneElementEditor::Display( int32_t width, int32_t height )
    {
        //PropertiesPanel::Display( width, height );

        ImGuiTreeNodeFlags l_Flags = ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_AllowItemOverlap;

        if( !ElementToEdit )
            return;

        char buf[128] = { 0 };
        std::strncpy( buf, ElementToEdit.Get<sTag>().mValue.c_str(), std::min( ElementToEdit.Get<sTag>().mValue.size(), std::size_t( 128 ) ) );
        if( ImGui::InputText( "##TAG_INPUT", buf, ARRAYSIZE( buf ), ImGuiInputTextFlags_EnterReturnsTrue ) )
        {
            ElementToEdit.Get<sTag>().mValue = std::string( buf );
        }
        UI::SameLine();
        if( UI::Button( fmt::format( "{} Add component", ICON_FA_PLUS ).c_str(), math::vec2{ 150.0f, 30.0f } ) )
        {
            ImGui::OpenPopup( "##add_component" );
        }

        if( ImGui::BeginPopup( "##add_component" ) )
        {
            if( ImGui::MenuItem( "Transform component", NULL, false, !ElementToEdit.Has<LocalTransformComponent>() ) )
            {
                ElementToEdit.Add<LocalTransformComponent>();
            }
            ImGui::Separator();
            if( ImGui::MenuItem( "Static mesh component", NULL, false, !ElementToEdit.Has<StaticMeshComponent>() ) )
            {
                ElementToEdit.Add<StaticMeshComponent>();
            }
            ImGui::Separator();
            if( ImGui::MenuItem( "Light component", NULL, false, !ElementToEdit.Has<LightComponent>() ) )
            {
                auto &l_Component = ElementToEdit.Add<LightComponent>();
                l_Component.Light = World->Create( "Light", ElementToEdit );
            }
            ImGui::Separator();
            if( ImGui::MenuItem( "Particle system", NULL, false, !ElementToEdit.Has<ParticleSystemComponent>() ) )
            {
                ElementToEdit.Add<ParticleSystemComponent>();

                auto &l_ParticleShaderConfiguration     = ElementToEdit.Add<ParticleShaderComponent>();
                l_ParticleShaderConfiguration.LineWidth = 1.0f;

                ParticleRendererCreateInfo l_RendererCreateInfo{};
                l_RendererCreateInfo.LineWidth = l_ParticleShaderConfiguration.LineWidth;
            }
            ImGui::EndPopup();
        }

        if( ImGui::CollapsingHeader( "Ray tracing", l_Flags ) )
        {
            bool l_RT = ElementToEdit.Has<RayTracingTargetComponent>();
            ImGui::Checkbox( "Mark as ray tracing target", &l_RT );
            if( l_RT )
                World->MarkAsRayTracingTarget( ElementToEdit );
            else
                ElementToEdit.TryRemove<RayTracingTargetComponent>();
        }

        if( ImGui::CollapsingHeader( "Transform", l_Flags ) )
        {
            ElementToEdit.IfExists<LocalTransformComponent>( [&]( auto &l_Component ) { EditComponent( l_Component ); } );
        }

        if( ImGui::CollapsingHeader( "Mesh", l_Flags ) )
        {
            ElementToEdit.IfExists<StaticMeshComponent>( [&]( auto &l_Component ) { EditComponent( l_Component ); } );

        }

        if( ImGui::CollapsingHeader( "Material", l_Flags ) )
        {
            static MaterialCombo l_MaterialChooser( "##material_chooser" );
            l_MaterialChooser.World = World;

            if( ElementToEdit.Has<RendererComponent>() )
            {
                l_MaterialChooser.Display( ElementToEdit.Get<RendererComponent>().Material );
            }
            else
            {
                auto l_NewRenderer = Entity{};
                l_MaterialChooser.Display( l_NewRenderer );
                if( l_NewRenderer )
                    ElementToEdit.Add<RendererComponent>( l_NewRenderer );
            }
        }

        if( ImGui::CollapsingHeader( "Light", l_Flags ) )
        {
            ElementToEdit.IfExists<LightComponent>( [&]( auto &l_LightComponent ) { EditComponent( mGraphicContext, l_LightComponent ); } );
        }
    }

} // namespace LTSE::Editor