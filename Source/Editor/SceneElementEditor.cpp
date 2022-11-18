#include "SceneElementEditor.h"

#include "UI/CanvasView.h"
#include "UI/UI.h"
#include "UI/Widgets.h"

#include "Scene/Components.h"

#include "Scene/Components/VisualHelpers.h"

using namespace SE::Core::EntityComponentSystem::Components;

namespace SE::Editor
{

    class MaterialCombo
    {
      public:
        Ref<Scene>           World;
        std::string          ID            = "";
        float                ThumbnailSize = 50.0f;
        UI::ComboBox<Entity> Dropdown;

      public:
        MaterialCombo() = default;
        MaterialCombo( std::string a_ID )
            : ID{ a_ID }
            , Dropdown{ UI::ComboBox<Entity>( a_ID ) } {};

        ~MaterialCombo() = default;

        Entity GetValue()
        {
            if( Dropdown.Values.size() > 0 ) return Dropdown.Values[Dropdown.CurrentItem];
            return Entity{};
        }

        void Display( Entity &a_TextureEntity )
        {
            Dropdown.Labels = { "None" };
            Dropdown.Values = { Entity{} };

            uint32_t n = 1;
            World->ForEach<sMaterialShaderComponent>(
                [&]( auto a_Entity, auto &aComponent )
                {
                    Dropdown.Labels.push_back( a_Entity.Get<sTag>().mValue );
                    Dropdown.Values.push_back( a_Entity );

                    if( (uint32_t)a_Entity == (uint32_t)a_TextureEntity ) Dropdown.CurrentItem = n;
                    n++;
                } );
            Dropdown.Display();

            if( Dropdown.Changed )
            {
                a_TextureEntity = Dropdown.GetValue();
            }
        }
    };

    template <typename _SliderType>
    class Slider
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

    template <typename _ValueChooserType>
    class PropertyEditor
    {
      public:
        std::string       Label;
        float             LabelWidth;
        _ValueChooserType ValueChooser;

        PropertyEditor( std::string ID ) { ValueChooser = _ValueChooserType( ID ); }

        template <typename... _ArgTypes>
        void Display( _ArgTypes... a_ArgList )
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

    static bool EditComponent( sActorComponent &aComponent )
    {
        UI::Text( "Static mesh" );
        UI::Text( "{}", aComponent.mClassFullName );

        // auto &entityFields = ScriptEngine::GetScriptFieldMap( entity );
        for( const auto &[name, field] : aComponent.mClass.GetFields() )
        {
            UI::Text( "{}", name );
            // // Field has been set in editor
            // if( entityFields.find( name ) != entityFields.end() )
            // {
            //     ScriptFieldInstance &scriptField = entityFields.at( name );

            //     // Display control to set it maybe
            //     if( field.Type == ScriptFieldType::Float )
            //     {
            //         float data = scriptField.GetValue<float>();
            //         if( ImGui::DragFloat( name.c_str(), &data ) ) scriptField.SetValue( data );
            //     }
            // }
            // else
            // {
            //     // Display control to set it maybe
            //     if( field.Type == ScriptFieldType::Float )
            //     {
            //         float data = 0.0f;
            //         if( ImGui::DragFloat( name.c_str(), &data ) )
            //         {
            //             ScriptFieldInstance &fieldInstance = entityFields[name];
            //             fieldInstance.Field                = field;
            //             fieldInstance.SetValue( data );
            //         }
            //     }
            // }
        }

        return false;
    }

    static bool EditComponent( sStaticMeshComponent &aComponent )
    {
        UI::Text( "Class:" );
        UI::Text( "{}", aComponent.Name );
        return false;
    }

    static bool EditComponent( sNodeTransformComponent &aComponent )
    {
        math::vec3 l_Position = aComponent.GetTranslation();
        math::vec3 l_Rotation = aComponent.GetEulerRotation();
        math::vec3 l_Scale    = aComponent.GetScale();

        UI::VectorComponentEditor( "Position:", l_Position, 0.0, 100 );
        UI::VectorComponentEditor( "Rotation:", l_Rotation, 0.0, 100 );
        UI::VectorComponentEditor( "Scale:", l_Scale, 1.0, 100 );

        aComponent.mMatrix = sNodeTransformComponent( l_Position, l_Rotation, l_Scale ).mMatrix;
        return false;
    }

    static bool EditComponent( sDirectionalLightComponent &aComponent )
    {
        float                                l_LabelSize = 175.0f;
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

        l_AzimuthEditor.Display( &aComponent.Azimuth );

        l_ElevationEditor.ValueChooser.MinValue = -90.0f;
        l_ElevationEditor.ValueChooser.MaxValue = 90.0f;
        l_ElevationEditor.Display( &aComponent.Elevation );

        UI::ColorChooser( "Color:", 125, aComponent.Color );

        l_IntensityEditor.Display( &aComponent.Intensity );
        return false;
    }

    static bool EditComponent( sPointLightComponent &aComponent )
    {
        float l_LabelSize = 175.0f;

        static PropertyEditor<Slider<float>> l_IntensityEditor( "##intensity" );
        l_IntensityEditor.Label                 = "Intensity:";
        l_IntensityEditor.LabelWidth            = l_LabelSize;
        l_IntensityEditor.ValueChooser.MinValue = 0.0f;
        l_IntensityEditor.ValueChooser.MaxValue = 50.0f;
        l_IntensityEditor.ValueChooser.Format   = "%.2f";

        UI::VectorComponentEditor( "Position:", aComponent.Position, 0.0, 100 );
        UI::ColorChooser( "Color:", 125, aComponent.Color );

        l_IntensityEditor.Display( &aComponent.Intensity );
        return false;
    }

    static bool EditComponent( sSpotlightComponent &aComponent )
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

        UI::VectorComponentEditor( "Position:", aComponent.Position, 0.0, 100 );

        l_AzimuthEditor.Display( &aComponent.Azimuth );

        l_ElevationEditor.ValueChooser.MinValue = 0.0f;
        l_ElevationEditor.ValueChooser.MaxValue = 360.0f;
        l_ElevationEditor.Display( &aComponent.Elevation );

        UI::ColorChooser( "Color:", 125, aComponent.Color );

        l_IntensityEditor.Display( &aComponent.Intensity );
        l_ConeWidthEditor.Display( &aComponent.Cone );
        return false;
    }

    static bool EditComponent( SE::Graphics::GraphicContext aGraphicContext, sLightComponent &aComponent )
    {
        static UI::ComboBox<eLightType> lPrimitiveChooser( "##combo_light_type_chooser" );
        lPrimitiveChooser.Labels = { "Directional light", "Point light", "Spotlight" };
        lPrimitiveChooser.Values = { eLightType::DIRECTIONAL, eLightType::POINT_LIGHT, eLightType::SPOTLIGHT };

        lPrimitiveChooser.Display();

        switch( lPrimitiveChooser.GetValue() )
        {
        case eLightType::POINT_LIGHT:
        {
            if( lPrimitiveChooser.Changed )
            {
                aComponent.Light.TryRemove<sSpotlightComponent>();
                aComponent.Light.TryRemove<SpotlightHelperComponent>();
                aComponent.Light.TryRemove<sDirectionalLightComponent>();
                aComponent.Light.TryRemove<DirectionalLightHelperComponent>();
            }
            if( !aComponent.Light.Has<sPointLightComponent>() )
            {
                aComponent.Light.Add<sPointLightComponent>();

                PointLightHelperComponent &l_VisualizerComponent = aComponent.Light.Add<PointLightHelperComponent>();
                l_VisualizerComponent.LightData                  = aComponent.Light.Get<sPointLightComponent>();
                l_VisualizerComponent.UpdateMesh( aGraphicContext );
            }
            aComponent.Light.Get<PointLightHelperComponent>().LightData = aComponent.Light.Get<sPointLightComponent>();
            EditComponent( aComponent.Light.Get<sPointLightComponent>() );
        }
        break;
        case eLightType::SPOTLIGHT:
        {
            if( lPrimitiveChooser.Changed )
            {
                aComponent.Light.TryRemove<sPointLightComponent>();
                aComponent.Light.TryRemove<PointLightHelperComponent>();
                aComponent.Light.TryRemove<sDirectionalLightComponent>();
                aComponent.Light.TryRemove<DirectionalLightHelperComponent>();
            }
            if( !aComponent.Light.Has<sSpotlightComponent>() )
            {
                aComponent.Light.Add<sSpotlightComponent>();

                SpotlightHelperComponent &l_VisualizerComponent = aComponent.Light.Add<SpotlightHelperComponent>();
                l_VisualizerComponent.LightData                 = aComponent.Light.Get<sSpotlightComponent>();
                l_VisualizerComponent.UpdateMesh( aGraphicContext );
            }
            aComponent.Light.Get<SpotlightHelperComponent>().LightData = aComponent.Light.Get<sSpotlightComponent>();
            EditComponent( aComponent.Light.Get<sSpotlightComponent>() );
        }
        break;
        case eLightType::DIRECTIONAL:
        default:
        {
            if( lPrimitiveChooser.Changed )
            {
                aComponent.Light.TryRemove<sPointLightComponent>();
                aComponent.Light.TryRemove<PointLightHelperComponent>();
                aComponent.Light.TryRemove<sSpotlightComponent>();
                aComponent.Light.TryRemove<SpotlightHelperComponent>();
            }
            if( !aComponent.Light.Has<sDirectionalLightComponent>() )
            {
                aComponent.Light.Add<sDirectionalLightComponent>();

                DirectionalLightHelperComponent &l_VisualizerComponent = aComponent.Light.Add<DirectionalLightHelperComponent>();
                l_VisualizerComponent.LightData                        = aComponent.Light.Get<sDirectionalLightComponent>();
                l_VisualizerComponent.UpdateMesh( aGraphicContext );
            }
            aComponent.Light.Get<DirectionalLightHelperComponent>().LightData = aComponent.Light.Get<sDirectionalLightComponent>();
            EditComponent( aComponent.Light.Get<sDirectionalLightComponent>() );
        }
        break;
        };

        return false;
    }

    SceneElementEditor::SceneElementEditor( GraphicContext &aGraphicContext )
        : mGraphicContext{ aGraphicContext } {};

    void SceneElementEditor::Display( int32_t width, int32_t height )
    {
        // PropertiesPanel::Display( width, height );

        ImGuiTreeNodeFlags lFlags = ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_AllowItemOverlap;

        if( !ElementToEdit ) return;

        char buf[128] = { 0 };
        std::strncpy( buf, ElementToEdit.Get<sTag>().mValue.c_str(),
                      std::min( ElementToEdit.Get<sTag>().mValue.size(), std::size_t( 128 ) ) );
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
                auto &lComponent = ElementToEdit.Add<sLightComponent>();
                lComponent.Light = World->Create( "Light", ElementToEdit );
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

        if( ImGui::CollapsingHeader( "Ray tracing", lFlags ) )
        {
            bool l_RT = ElementToEdit.Has<sRayTracingTargetComponent>();
            ImGui::Checkbox( "Mark as ray tracing target", &l_RT );
            if( l_RT )
                World->MarkAsRayTracingTarget( ElementToEdit );
            else
                ElementToEdit.TryRemove<sRayTracingTargetComponent>();
        }

        if( ImGui::CollapsingHeader( "Script", lFlags ) )
        {
            ElementToEdit.IfExists<sActorComponent>( [&]( auto &lComponent ) { EditComponent( lComponent ); } );
        }

        if( ImGui::CollapsingHeader( "Transform", lFlags ) )
        {
            ElementToEdit.IfExists<sNodeTransformComponent>( [&]( auto &lComponent ) { EditComponent( lComponent ); } );
        }

        if( ImGui::CollapsingHeader( "Mesh", lFlags ) )
        {
            ElementToEdit.IfExists<sStaticMeshComponent>( [&]( auto &lComponent ) { EditComponent( lComponent ); } );
        }

        if( ImGui::CollapsingHeader( "Material", lFlags ) )
        {
            static MaterialCombo lMaterialChooser( "##material_chooser" );
            lMaterialChooser.World = World;
        }

        if( ImGui::CollapsingHeader( "Light", lFlags ) )
        {
            ElementToEdit.IfExists<sLightComponent>( [&]( auto &lLightComponent )
                                                     { EditComponent( mGraphicContext, lLightComponent ); } );
        }
    }

} // namespace SE::Editor