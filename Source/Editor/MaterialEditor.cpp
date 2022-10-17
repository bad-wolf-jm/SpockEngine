#include "MaterialEditor.h"

#include "UI/CanvasView.h"
#include "UI/UI.h"
#include "UI/Widgets.h"

#include "Scene/Components.h"

using namespace LTSE::Core::EntityComponentSystem;

namespace LTSE::Editor
{
    using namespace LTSE::Core;

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

    class TextureCombo
    {
      public:
        std::string ID       = "";
        uint32_t CurrentItem = 0;
        float ThumbnailSize  = 50.0f;
        bool Changed         = false;
        UI::ComboBox<Entity> Dropdown;
        Ref<Scene> g_World = nullptr;

      public:
        TextureCombo() = default;
        TextureCombo( std::string a_ID )
            : ID{ a_ID }
            , Dropdown{ UI::ComboBox<Entity>( a_ID ) } {};

        ~TextureCombo() = default;

        void Display( Entity &a_TextureEntity )
        {
            Dropdown.Labels = { "No Texture" };
            Dropdown.Values = { Entity{} };

            if( g_World == nullptr )
                return;

            std::vector<Entity> Textures = { Entity{} };
            uint32_t n                   = 1;
            g_World->ForEach<TextureComponent>(
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

    class MaterialTextureChooser
    {
      public:
        std::string ID = "";
        TextureCombo Dropdown;
        float ThumbnailSize = 56.0f;
        bool Changed        = false;

      public:
        MaterialTextureChooser() = default;
        MaterialTextureChooser( std::string a_ID )
            : Dropdown{ TextureCombo( a_ID ) } {};

        ~MaterialTextureChooser() = default;

        void Display( Entity &a_TextureEntity )
        {
            auto &l_ItemPosition   = UI::GetCurrentCursorPosition();
            auto &l_TextPosition   = UI::GetCurrentCursorPosition() + math::vec2( ThumbnailSize + 9.0f, ( ThumbnailSize - 30.0f ) / 2.0f );
            ImDrawList *l_DrawList = ImGui::GetWindowDrawList();
            ImVec2 pos             = ImGui::GetCursorScreenPos();
            if( a_TextureEntity.Has<TextureComponent>() )
            {
                if( a_TextureEntity.Has<TexturePreviewComponent>() )
                {
                    UI::Image( a_TextureEntity.Get<TexturePreviewComponent>().Descriptor, math::vec2{ ThumbnailSize, ThumbnailSize } );
                }
                // else
                // {
                //     UI::Image( g_World->GetDefaultPreviewImageHandle(), math::vec2{ ThumbnailSize, ThumbnailSize } );
                // }
            }
            // else
            // {
            //     UI::Image( g_World->GetDefaultPreviewImageHandle(), math::vec2{ ThumbnailSize, ThumbnailSize } );
            // }
            if( ImGui::BeginDragDropTarget() )
            {
                ImGuiDragDropFlags target_flags = 0;
                target_flags |= ImGuiDragDropFlags_AcceptNoDrawDefaultRect;
                if( const ImGuiPayload *payload = ImGui::AcceptDragDropPayload( "TEXTURE_DATA", target_flags ) )
                {
                    // a_TextureEntity = g_DragElement;
                    // g_DragElement   = Entity{};
                }
                ImGui::EndDragDropTarget();
            }
            l_DrawList->AddRect( ImVec2{ pos.x, pos.y }, ImVec2{ pos.x + ThumbnailSize, pos.y + ThumbnailSize }, IM_COL32( 255, 255, 0, 255 ) );

            UI::SetCursorPosition( l_TextPosition );
            ImGui::SetNextItemWidth( 225.0f );
            Dropdown.Display( a_TextureEntity );
            Changed = Dropdown.Changed;
            l_ItemPosition += math::vec2( 0.0f, ThumbnailSize + 11.0f );
            l_TextPosition = l_ItemPosition + math::vec2( ThumbnailSize + 11.0f, 0.0f );
            UI::SetCursorPosition( l_ItemPosition );
        }
    };

    void MaterialEditor::Display( int32_t width, int32_t height )
    {
        if( !ElementToEdit )
            return;

        float l_LabelSize = 175.0f;

        static MaterialTextureChooser l_AlbedoChooser( "##albedo_choser" );
        static MaterialTextureChooser l_NormalChooser( "##normal_choser" );
        static MaterialTextureChooser l_EmissiveChooser( "##emissive_choser" );
        static MaterialTextureChooser l_OcclusionChooser( "##occlusion_choser" );
        static MaterialTextureChooser l_MetalnessChooser( "##metalness_choser" );

        static PropertyEditor<Slider<float>> l_LineWidthEditor( "##line_width" );
        l_LineWidthEditor.Label                 = "Line width:";
        l_LineWidthEditor.LabelWidth            = l_LabelSize;
        l_LineWidthEditor.ValueChooser.MinValue = 0.0f;
        l_LineWidthEditor.ValueChooser.MaxValue = 5.0f;
        l_LineWidthEditor.ValueChooser.Format   = "%.2f";

        static PropertyEditor<Slider<float>> l_AlphaMaskThresholdEditor( "##alpha_threshold" );
        l_AlphaMaskThresholdEditor.Label                 = "Alpha mask threshold:";
        l_AlphaMaskThresholdEditor.LabelWidth            = l_LabelSize;
        l_AlphaMaskThresholdEditor.ValueChooser.MinValue = 0.0f;
        l_AlphaMaskThresholdEditor.ValueChooser.MaxValue = 1.0f;
        l_AlphaMaskThresholdEditor.ValueChooser.Format   = "%.2f";

        static PropertyEditor<Slider<float>> l_RoughnessEditor( "##roughness" );
        l_RoughnessEditor.Label                 = "Roughness:";
        l_RoughnessEditor.LabelWidth            = l_LabelSize;
        l_RoughnessEditor.ValueChooser.MinValue = 0.0f;
        l_RoughnessEditor.ValueChooser.MaxValue = 1.0f;
        l_RoughnessEditor.ValueChooser.Format   = "%.2f";

        static PropertyEditor<Slider<float>> l_MetalnessEditor( "##metalness" );
        l_MetalnessEditor.Label                 = "Metalness:";
        l_MetalnessEditor.LabelWidth            = l_LabelSize;
        l_MetalnessEditor.ValueChooser.MinValue = 0.0f;
        l_MetalnessEditor.ValueChooser.MaxValue = 1.0f;
        l_MetalnessEditor.ValueChooser.Format   = "%.2f";

        static PropertyEditor<Slider<float>> l_OcclusionStrengthEditor( "##occlusion" );
        l_OcclusionStrengthEditor.Label                 = "Occlusion Strength:";
        l_OcclusionStrengthEditor.LabelWidth            = l_LabelSize;
        l_OcclusionStrengthEditor.ValueChooser.MinValue = 0.0f;
        l_OcclusionStrengthEditor.ValueChooser.MaxValue = 1.0f;
        l_OcclusionStrengthEditor.ValueChooser.Format   = "%.2f";

        char buf[128] = { 0 };
        if( ElementToEdit.Has<sTag>() )
        {
            std::strncpy( buf, ElementToEdit.Get<sTag>().mValue.c_str(), std::min( ElementToEdit.Get<sTag>().mValue.size(), std::size_t( 128 ) ) );
            if( ImGui::InputText( "##TAG_INPUT", buf, ARRAYSIZE( buf ), ImGuiInputTextFlags_EnterReturnsTrue ) )
            {
                ElementToEdit.Get<sTag>().mValue = std::string( buf );
            }
        }
        else
        {
            if( ImGui::InputText( "##TAG_INPUT", buf, ARRAYSIZE( buf ), ImGuiInputTextFlags_EnterReturnsTrue ) )
            {
                ElementToEdit.Add<sTag>( std::string( buf ) );
            }
        }

        math::ivec2 l_WindowSize = UI::GetAvailableContentSpace();

        auto &l_MaterialShaderConfiguration = ElementToEdit.Get<sMaterialShaderComponent>();
        ImGui::Checkbox( "Two sided", &l_MaterialShaderConfiguration.IsTwoSided );
        {
            ImVec2 l_CursorPosition = ImGui::GetCursorPos();
            l_LineWidthEditor.Display( &l_MaterialShaderConfiguration.LineWidth );
        }

        if( ( l_MaterialShaderConfiguration.IsTwoSided != l_MaterialShaderConfiguration.Renderer.Spec.IsTwoSided ) ||
            ( l_MaterialShaderConfiguration.LineWidth != l_MaterialShaderConfiguration.Renderer.Spec.LineWidth ) )
        {
            MeshRendererCreateInfo l_RendererCreateInfo{};
            l_RendererCreateInfo.IsTwoSided = l_MaterialShaderConfiguration.IsTwoSided;
            l_RendererCreateInfo.LineWidth  = l_MaterialShaderConfiguration.LineWidth;
        }

        ImGui::Checkbox( "Use alpha mask", &l_MaterialShaderConfiguration.UseAlphaMask );
        l_AlphaMaskThresholdEditor.Display( &l_MaterialShaderConfiguration.AlphaMaskTheshold );

        auto &l_MaterialConstantsConfiguration = ElementToEdit.Get<MaterialConstantsComponent>();
        l_RoughnessEditor.Display( &l_MaterialConstantsConfiguration.RoughnessFactor );
        l_MetalnessEditor.Display( &l_MaterialConstantsConfiguration.MetalicFactor );
        l_OcclusionStrengthEditor.Display( &l_MaterialConstantsConfiguration.OcclusionStrength );

        auto &l_MaterialColorConfiguration = ElementToEdit.Get<MaterialColorsComponent>();
        UI::ColorChooser( "Albedo Color", l_LabelSize, l_MaterialColorConfiguration.Albedo );
        UI::ColorChooser( "Emissive Color", l_LabelSize, l_MaterialColorConfiguration.Emissive );

        UI::Text( "TEXTURES" );

        bool l_MaterialTextureChanged = false;
        auto &l_MaterialTextures      = ElementToEdit.Get<MaterialTexturesComponent>();
        auto &l_TextureItemPosition   = UI::GetCurrentCursorPosition();
        l_LabelSize                   = 110.0f;
        float l_ItemHeight            = 64;
        float l_LabelHeight           = ImGui::CalcTextSize( "Albedo Map" ).y;
        UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2{ 0.0f, ( l_ItemHeight - l_LabelHeight ) / 2.0f } );
        UI::Text( "Albedo Map" );
        UI::SameLine();
        UI::SetCursorPosition( l_TextureItemPosition + math::vec2{ l_LabelSize, 0.0f } );
        l_AlbedoChooser.Display( l_MaterialTextures.Albedo );
        l_MaterialTextureChanged |= l_AlbedoChooser.Changed;

        l_TextureItemPosition += math::vec2{ 0.0f, 80.0f };
        UI::SetCursorPosition( l_TextureItemPosition + math::vec2{ 0.0f, ( l_ItemHeight - l_LabelHeight ) / 2.0f } );
        UI::Text( "Normal Map" );
        UI::SameLine();
        UI::SetCursorPosition( l_TextureItemPosition + math::vec2{ l_LabelSize, 0.0f } );
        l_NormalChooser.Display( l_MaterialTextures.Normals );
        l_MaterialTextureChanged |= l_NormalChooser.Changed;

        l_TextureItemPosition += math::vec2{ 0.0f, 80.0f };
        UI::SetCursorPosition( l_TextureItemPosition + math::vec2{ 0.0f, ( l_ItemHeight - l_LabelHeight ) / 2.0f } );
        UI::Text( "Emissive Map" );
        UI::SameLine();
        UI::SetCursorPosition( l_TextureItemPosition + math::vec2{ l_LabelSize, 0.0f } );
        l_EmissiveChooser.Display( l_MaterialTextures.Emissive );
        l_MaterialTextureChanged |= l_EmissiveChooser.Changed;

        l_TextureItemPosition += math::vec2{ 0.0f, 80.0f };
        UI::SetCursorPosition( l_TextureItemPosition + math::vec2{ 0.0f, ( l_ItemHeight - l_LabelHeight ) / 2.0f } );
        UI::Text( "Occlusion Map" );
        UI::SameLine();
        UI::SetCursorPosition( l_TextureItemPosition + math::vec2{ l_LabelSize, 0.0f } );
        l_OcclusionChooser.Display( l_MaterialTextures.Occlusion );
        l_MaterialTextureChanged |= l_OcclusionChooser.Changed;

        l_TextureItemPosition += math::vec2{ 0.0f, 80.0f };
        UI::SetCursorPosition( l_TextureItemPosition + math::vec2{ 0.0f, ( l_ItemHeight - l_LabelHeight ) / 2.0f } );
        UI::Text( "Metalness" );
        UI::SameLine();
        UI::SetCursorPosition( l_TextureItemPosition + math::vec2{ l_LabelSize, 0.0f } );
        l_MetalnessChooser.Display( l_MaterialTextures.Metalness );
        l_MaterialTextureChanged |= l_MetalnessChooser.Changed;

        // if( l_MaterialTextureChanged )
        //     l_MaterialTextures.UpdateDescriptors( g_World->GetDefaultImage() );

        l_TextureItemPosition += math::vec2{ 0.0f, 80.0f };
        UI::SetCursorPosition( l_TextureItemPosition );
    }
} // namespace LTSE::Editor