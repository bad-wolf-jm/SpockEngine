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
        UI::Text( "{}", aComponent.mName );
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

    static bool EditComponent( SE::Graphics::Ref<VkGraphicContext> aGraphicContext, sLightComponent &aComponent )
    {
        static UI::ComboBox<eLightType> lPrimitiveChooser( "##combo_light_type_chooser" );
        lPrimitiveChooser.Labels = { "Point light", "Spotlight", "Directional light" };
        lPrimitiveChooser.Values = { eLightType::POINT_LIGHT, eLightType::SPOTLIGHT, eLightType::DIRECTIONAL };

        lPrimitiveChooser.Display();

        float lLabelSize = 175.0f;
        aComponent.mType = lPrimitiveChooser.GetValue();
        switch( aComponent.mType )
        {
        case eLightType::POINT_LIGHT:
        {

            static PropertyEditor<Slider<float>> l_IntensityEditor( "##intensity" );
            l_IntensityEditor.Label                 = "Intensity:";
            l_IntensityEditor.LabelWidth            = lLabelSize;
            l_IntensityEditor.ValueChooser.MinValue = 0.0f;
            l_IntensityEditor.ValueChooser.MaxValue = 50.0f;
            l_IntensityEditor.ValueChooser.Format   = "%.2f";

            UI::ColorChooser( "Color:", 125, aComponent.mColor );

            l_IntensityEditor.Display( &aComponent.mIntensity );
        }
        break;
        case eLightType::SPOTLIGHT:
        {
            static PropertyEditor<Slider<float>> l_ConeWidthEditor( "##cone_width" );
            l_ConeWidthEditor.Label                 = "Cone width:";
            l_ConeWidthEditor.ValueChooser.MinValue = 0.0f;
            l_ConeWidthEditor.ValueChooser.MaxValue = 180.0f;
            l_ConeWidthEditor.LabelWidth            = lLabelSize;
            l_ConeWidthEditor.ValueChooser.Format   = "%.2f";

            static PropertyEditor<Slider<float>> l_IntensityEditor( "##intensity" );
            l_IntensityEditor.Label                 = "Intensity:";
            l_IntensityEditor.LabelWidth            = lLabelSize;
            l_IntensityEditor.ValueChooser.MinValue = 0.0f;
            l_IntensityEditor.ValueChooser.MaxValue = 50.0f;
            l_IntensityEditor.ValueChooser.Format   = "%.2f";

            UI::ColorChooser( "Color:", 125, aComponent.mColor );

            l_IntensityEditor.Display( &aComponent.mIntensity );
            l_ConeWidthEditor.Display( &aComponent.mCone );
        }
        break;
        case eLightType::DIRECTIONAL:
        default:
        {
            float                                lLabelSize = 175.0f;
            static PropertyEditor<Slider<float>> l_IntensityEditor( "##intensity" );
            l_IntensityEditor.Label                 = "Intensity:";
            l_IntensityEditor.LabelWidth            = lLabelSize;
            l_IntensityEditor.ValueChooser.MinValue = 0.0f;
            l_IntensityEditor.ValueChooser.MaxValue = 50.0f;
            l_IntensityEditor.ValueChooser.Format   = "%.2f";

            UI::ColorChooser( "Color:", 125, aComponent.mColor );

            l_IntensityEditor.Display( &aComponent.mIntensity );
        }
        break;
        };

        return false;
    }

    SceneElementEditor::SceneElementEditor( Ref<VkGraphicContext> aGraphicContext )
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
                ElementToEdit.Add<sLightComponent>();
                // lComponent.Light = World->Create( "Light", ElementToEdit );
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