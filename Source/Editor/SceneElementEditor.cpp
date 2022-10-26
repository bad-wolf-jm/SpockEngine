#include "SceneElementEditor.h"

#include "UI/CanvasView.h"
#include "UI/UI.h"
#include "UI/Widgets.h"

#include "Scene/Components.h"

#include "Scene/Components/VisualHelpers.h"

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
            World->ForEach<sMaterialShaderComponent>(
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

    static bool EditComponent( sStaticMeshComponent &a_Component )
    {
        UI::Text( "Static mesh" );
        UI::Text( "{}", a_Component.Name );
        return false;
    }

    static bool EditComponent( sNodeTransformComponent &a_Component )
    {
        math::vec3 l_Position = a_Component.GetTranslation();
        math::vec3 l_Rotation = a_Component.GetEulerRotation();
        math::vec3 l_Scale    = a_Component.GetScale();

        UI::VectorComponentEditor( "Position:", l_Position, 0.0, 100 );
        UI::VectorComponentEditor( "Rotation:", l_Rotation, 0.0, 100 );
        UI::VectorComponentEditor( "Scale:", l_Scale, 1.0, 100 );

        a_Component.mMatrix = sNodeTransformComponent( l_Position, l_Rotation, l_Scale ).mMatrix;
        return false;
    }

    static bool EditComponent( sDirectionalLightComponent &a_Component )
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

    static bool EditComponent( sPointLightComponent &a_Component )
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

    static bool EditComponent( sSpotlightComponent &a_Component )
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

    static bool EditComponent( LTSE::Graphics::GraphicContext aGraphicContext, sLightComponent &a_Component )
    {
        static UI::ComboBox<eLightType> l_PrimitiveChooser( "##combo_light_type_chooser" );
        l_PrimitiveChooser.Labels = { "Directional light", "Point light", "Spotlight" };
        l_PrimitiveChooser.Values = { eLightType::DIRECTIONAL, eLightType::POINT_LIGHT, eLightType::SPOTLIGHT };

        l_PrimitiveChooser.Display();

        switch( l_PrimitiveChooser.GetValue() )
        {
        case eLightType::POINT_LIGHT:
        {
            if( l_PrimitiveChooser.Changed )
            {
                a_Component.Light.TryRemove<sSpotlightComponent>();
                a_Component.Light.TryRemove<SpotlightHelperComponent>();
                a_Component.Light.TryRemove<sDirectionalLightComponent>();
                a_Component.Light.TryRemove<DirectionalLightHelperComponent>();
            }
            if( !a_Component.Light.Has<sPointLightComponent>() )
            {
                a_Component.Light.Add<sPointLightComponent>();

                PointLightHelperComponent &l_VisualizerComponent = a_Component.Light.Add<PointLightHelperComponent>();
                l_VisualizerComponent.LightData                  = a_Component.Light.Get<sPointLightComponent>();
                l_VisualizerComponent.UpdateMesh( aGraphicContext );
            }
            a_Component.Light.Get<PointLightHelperComponent>().LightData = a_Component.Light.Get<sPointLightComponent>();
            EditComponent( a_Component.Light.Get<sPointLightComponent>() );
        }
        break;
        case eLightType::SPOTLIGHT:
        {
            if( l_PrimitiveChooser.Changed )
            {
                a_Component.Light.TryRemove<sPointLightComponent>();
                a_Component.Light.TryRemove<PointLightHelperComponent>();
                a_Component.Light.TryRemove<sDirectionalLightComponent>();
                a_Component.Light.TryRemove<DirectionalLightHelperComponent>();
            }
            if( !a_Component.Light.Has<sSpotlightComponent>() )
            {
                a_Component.Light.Add<sSpotlightComponent>();

                SpotlightHelperComponent &l_VisualizerComponent = a_Component.Light.Add<SpotlightHelperComponent>();
                l_VisualizerComponent.LightData                 = a_Component.Light.Get<sSpotlightComponent>();
                l_VisualizerComponent.UpdateMesh( aGraphicContext );
            }
            a_Component.Light.Get<SpotlightHelperComponent>().LightData = a_Component.Light.Get<sSpotlightComponent>();
            EditComponent( a_Component.Light.Get<sSpotlightComponent>() );
        }
        break;
        case eLightType::DIRECTIONAL:
        default:
        {
            if( l_PrimitiveChooser.Changed )
            {
                a_Component.Light.TryRemove<sPointLightComponent>();
                a_Component.Light.TryRemove<PointLightHelperComponent>();
                a_Component.Light.TryRemove<sSpotlightComponent>();
                a_Component.Light.TryRemove<SpotlightHelperComponent>();
            }
            if( !a_Component.Light.Has<sDirectionalLightComponent>() )
            {
                a_Component.Light.Add<sDirectionalLightComponent>();

                DirectionalLightHelperComponent &l_VisualizerComponent = a_Component.Light.Add<DirectionalLightHelperComponent>();
                l_VisualizerComponent.LightData                        = a_Component.Light.Get<sDirectionalLightComponent>();
                l_VisualizerComponent.UpdateMesh( aGraphicContext );
            }
            a_Component.Light.Get<DirectionalLightHelperComponent>().LightData = a_Component.Light.Get<sDirectionalLightComponent>();
            EditComponent( a_Component.Light.Get<sDirectionalLightComponent>() );
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
            if( ImGui::MenuItem( "Transform component", NULL, false, !ElementToEdit.Has<sNodeTransformComponent>() ) )
            {
                ElementToEdit.Add<sNodeTransformComponent>();
            }
            ImGui::Separator();
            if( ImGui::MenuItem( "Static mesh component", NULL, false, !ElementToEdit.Has<sStaticMeshComponent>() ) )
            {
                ElementToEdit.Add<sStaticMeshComponent>();
            }
            ImGui::Separator();
            if( ImGui::MenuItem( "Light component", NULL, false, !ElementToEdit.Has<sLightComponent>() ) )
            {
                auto &l_Component = ElementToEdit.Add<sLightComponent>();
                l_Component.Light = World->Create( "Light", ElementToEdit );
            }
            ImGui::Separator();
            if( ImGui::MenuItem( "Particle system", NULL, false, !ElementToEdit.Has<sParticleSystemComponent>() ) )
            {
                ElementToEdit.Add<sParticleSystemComponent>();

                auto &l_ParticleShaderConfiguration     = ElementToEdit.Add<sParticleShaderComponent>();
                l_ParticleShaderConfiguration.LineWidth = 1.0f;

                ParticleRendererCreateInfo l_RendererCreateInfo{};
                l_RendererCreateInfo.LineWidth = l_ParticleShaderConfiguration.LineWidth;
            }
            ImGui::EndPopup();
        }

        if( ImGui::CollapsingHeader( "Ray tracing", l_Flags ) )
        {
            bool l_RT = ElementToEdit.Has<sRayTracingTargetComponent>();
            ImGui::Checkbox( "Mark as ray tracing target", &l_RT );
            if( l_RT )
                World->MarkAsRayTracingTarget( ElementToEdit );
            else
                ElementToEdit.TryRemove<sRayTracingTargetComponent>();
        }

        if( ImGui::CollapsingHeader( "Transform", l_Flags ) )
        {
            ElementToEdit.IfExists<sNodeTransformComponent>( [&]( auto &l_Component ) { EditComponent( l_Component ); } );
        }

        if( ImGui::CollapsingHeader( "Mesh", l_Flags ) )
        {
            ElementToEdit.IfExists<sStaticMeshComponent>( [&]( auto &l_Component ) { EditComponent( l_Component ); } );

        }

        if( ImGui::CollapsingHeader( "Material", l_Flags ) )
        {
            static MaterialCombo l_MaterialChooser( "##material_chooser" );
            l_MaterialChooser.World = World;

            // if( ElementToEdit.Has<RendererComponent>() )
            // {
            //     l_MaterialChooser.Display( ElementToEdit.Get<RendererComponent>().Material );
            // }
            // else
            // {
            //     auto l_NewRenderer = Entity{};
            //     l_MaterialChooser.Display( l_NewRenderer );
            //     if( l_NewRenderer )
            //         ElementToEdit.Add<RendererComponent>( l_NewRenderer );
            // }
        }

        if( ImGui::CollapsingHeader( "Light", l_Flags ) )
        {
            ElementToEdit.IfExists<sLightComponent>( [&]( auto &l_LightComponent ) { EditComponent( mGraphicContext, l_LightComponent ); } );
        }
    }

} // namespace LTSE::Editor