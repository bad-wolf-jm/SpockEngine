#include "Widgets.h"
#include "ImGuizmo.h"
#include <vulkan/vulkan.h>


namespace SE::Core::UI
{

    void HelpMarker( const char *desc )
    {
        ImGui::TextDisabled( "(?)" );
        if( ImGui::IsItemHovered() )
        {
            ImGui::BeginTooltip();
            ImGui::PushTextWrapPos( ImGui::GetFontSize() * 35.0f );
            ImGui::TextUnformatted( desc );
            ImGui::PopTextWrapPos();
            ImGui::EndTooltip();
        }
    }

    void Separator() { ImGui::Separator(); }

    bool ImageButton( SE::Core::UI::ImageHandle a_Texture, math::vec2 a_Size, math::vec4 a_Rect )
    {
        return ImGui::ImageButton( (ImTextureID)a_Texture.Handle->GetVkDescriptorSet(), ImVec2{ a_Size.x, a_Size.y },
            ImVec2{ a_Rect.x, a_Rect.y }, ImVec2{ a_Rect.z, a_Rect.w }, 0, ImVec4{ 0.0f, 0.0f, 0.0f, 0.0f },
            ImVec4{ 0.0f, 0.0f, 0.0f, 0.0f } );
    }

    bool Button( const char *a_Label, math::vec2 a_Size )
    {
        ImVec2 l_Size = { a_Size.x, a_Size.y };
        return ImGui::Button( a_Label, l_Size );
    }

    void ToggleButton( const char *a_OnLabel, const char *a_OffLabel, bool *l_State, math::vec2 a_Size )
    {
        ImVec2 l_Size = { a_Size.x, a_Size.y };
        if( *l_State )
        {
            bool l_Clicked = ImGui::Button( a_OnLabel, l_Size );
            if( l_Clicked )
            {
                *l_State = false;
            }
        }
        else
        {
            bool l_Clicked = ImGui::Button( a_OffLabel, l_Size );
            if( l_Clicked )
            {
                *l_State = true;
            }
        }
    }

    void ToggleButton(
        const char *a_OnLabel, const char *a_OffLabel, bool l_State, math::vec2 a_Size, std::function<void( bool )> a_Action )
    {
        ImVec2 l_Size = { a_Size.x, a_Size.y };
        if( l_State )
        {
            bool l_Clicked = ImGui::Button( a_OnLabel, l_Size );
            if( l_Clicked ) a_Action( false );
        }
        else
        {
            bool l_Clicked = ImGui::Button( a_OffLabel, l_Size );
            if( l_Clicked ) a_Action( true );
        }
    }

    void ToggleButton( const char *a_OnLabel, const char *a_OffLabel, ImFont *a_Font, bool l_State, math::vec2 a_Size,
        std::function<void( bool )> a_Action )
    {
        ImGui::PushFont( a_Font );
        ToggleButton( a_OnLabel, a_OffLabel, l_State, a_Size, a_Action );
        ImGui::PopFont();
    }

    void BulletText( const char *a_Text ) { ImGui::BulletText( a_Text ); }

    void Slider( const char *a_DisplayTemplate, uint16_t a_MinValue, uint16_t a_MaxValue, uint16_t *a_CurrentValue )
    {
        ImGui::PushStyleVar( ImGuiStyleVar_FrameBorderSize, 1.0f );
        ImGui::SliderScalar( "##unique_id", ImGuiDataType_U16, a_CurrentValue, &a_MinValue, &a_MaxValue, a_DisplayTemplate );
        ImGui::PopStyleVar();
    }

    void Checkbox( const char *label, bool *a_Value )
    {
        ImGui::PushStyleVar( ImGuiStyleVar_FrameBorderSize, 1.0f );
        ImGui::PushStyleVar( ImGuiStyleVar_ItemInnerSpacing, ImVec2{ 10.0f, 0.0f } );
        ImGui::Checkbox( label, a_Value );
        ImGui::PopStyleVar( 2 );
    }

    void ColorButton( math::vec4 &a_Color, math::vec2 a_Size )
    {
        ImGui::ColorButton( "##unique_id", ImVec4{ a_Color.r, a_Color.g, a_Color.b, a_Color.a }, 0, ImVec2{ a_Size.x, a_Size.y } );
    }

    bool MenuItem( const char *label, const char *shortcut )
    {
        bool l_Selected = false;
        ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, ImVec2( 30.0f, 10.0f ) );
        auto l_Value = ImGui::MenuItem( label, shortcut, &l_Selected );
        ImGui::PopStyleVar();
        return l_Value;
    }

    bool SelectionMenuItem( const char *label, const char *shortcut, bool *selected )
    {
        ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, ImVec2( 30.0f, 10.0f ) );
        auto l_Value = ImGui::MenuItem( label, shortcut, selected, true );
        ImGui::PopStyleVar();
        return l_Value;
    }

    void Image( SE::Core::UI::ImageHandle a_Texture, math::vec2 a_Size, math::vec4 a_Rect )
    {
        ImGui::Image( (ImTextureID)a_Texture.Handle->GetVkDescriptorSet(), ImVec2{ a_Size.x, a_Size.y }, ImVec2{ a_Rect.x, a_Rect.y },
            ImVec2{ a_Rect.z, a_Rect.w } );
    }

    void Image( SE::Core::UI::ImageHandle a_Texture, math::vec2 a_Size )
    {
        ImGui::Image(
            (ImTextureID)a_Texture.Handle->GetVkDescriptorSet(), ImVec2{ a_Size.x, a_Size.y }, ImVec2{ 0, 0 }, ImVec2{ 1, 1 } );
    }

    void Image( SE::Core::UI::ImageHandle a_Texture, math::ivec2 a_Size )
    {
        ImGui::Image( (ImTextureID)a_Texture.Handle->GetVkDescriptorSet(), ImVec2{ (float)a_Size.x, (float)a_Size.y }, ImVec2{ 0, 0 },
            ImVec2{ 1, 1 } );
    }

    void Menu( const char *a_Label, std::function<void()> a_Elements )
    {
        if( ImGui::BeginMenu( a_Label ) )
        {
            a_Elements();
            ImGui::EndMenu();
        }
    }

    bool Slider(
        std::string a_Title, const char *a_DisplayTemplate, uint16_t a_MinValue, uint16_t a_MaxValue, uint16_t *a_CurrentValue )
    {
        bool changed = ImGui::SliderScalar(
            a_Title.c_str(), ImGuiDataType_U16, a_CurrentValue, &a_MinValue, &a_MaxValue, a_DisplayTemplate, 1.0f );

        return changed;
    }

    bool Slider(
        std::string a_Title, const char *a_DisplayTemplate, uint32_t a_MinValue, uint32_t a_MaxValue, uint32_t *a_CurrentValue )
    {
        bool changed = ImGui::SliderScalar(
            a_Title.c_str(), ImGuiDataType_U32, a_CurrentValue, &a_MinValue, &a_MaxValue, a_DisplayTemplate, 1.0f );

        return changed;
    }

    bool Slider( std::string a_Title, const char *a_DisplayTemplate, int32_t a_MinValue, int32_t a_MaxValue, int32_t *a_CurrentValue )
    {
        bool changed = ImGui::SliderScalar(
            a_Title.c_str(), ImGuiDataType_S32, a_CurrentValue, &a_MinValue, &a_MaxValue, a_DisplayTemplate, 1.0f );

        return changed;
    }

    bool Slider( std::string a_Title, const char *a_DisplayTemplate, float a_MinValue, float a_MaxValue, float *a_CurrentValue )
    {
        bool changed = ImGui::SliderScalar(
            a_Title.c_str(), ImGuiDataType_Float, a_CurrentValue, &a_MinValue, &a_MaxValue, a_DisplayTemplate, 0 );

        return changed;
    }

    void Manipulate( ManipulationConfig a_Config, math::mat4 &a_Transform )
    {
        math::mat4 l_GuizmoProj( a_Config.Projection );
        l_GuizmoProj[1][1] *= -1.0f;

        ImGuizmo::OPERATION l_Op;
        switch( a_Config.Type )
        {
        case ManipulationType::TRANSLATION:
            l_Op = ImGuizmo::OPERATION::TRANSLATE;
            break;
        case ManipulationType::ROTATION:
            l_Op = ImGuizmo::OPERATION::ROTATE;
            break;
        case ManipulationType::SCALE:
            l_Op = ImGuizmo::OPERATION::SCALE;
            break;
        }

        math::mat4 l_TransformationMatrix = a_Transform;
        ImGuizmo::SetRect(
            a_Config.ViewportPosition.x, a_Config.ViewportPosition.y, a_Config.ViewportSize.x, a_Config.ViewportSize.y );
        ImGuizmo::Manipulate( math::ptr( a_Config.WorldTransform ), math::ptr( l_GuizmoProj ), l_Op, ImGuizmo::WORLD,
            math::ptr( l_TransformationMatrix ), nullptr, nullptr );
        if( ImGuizmo::IsUsing() )
        {
            a_Transform = l_TransformationMatrix;
        }
    }

    bool ViewManipulate( math::vec3 a_CameraPosition, math::mat4 &a_CameraView, math::vec2 a_Position )
    {
        auto cameraDistance = math::length( a_CameraPosition );
        cameraDistance      = ( cameraDistance == 0 ) ? 0.0001f : cameraDistance;
        ImGuizmo::ViewManipulate(
            math::ptr( a_CameraView ), cameraDistance, ImVec2{ a_Position.x, a_Position.y }, ImVec2{ 100.0f, 100.0f }, 0x88020202 );

        return ImRect( ImVec2{ a_Position.x, a_Position.y }, ImVec2{ a_Position.x, a_Position.y } + ImVec2{ 100.0f, 100.0f } )
            .Contains( ImGui::GetIO().MousePos );
    }

    bool ColorChooser( const std::string label, int label_width, math::vec4 &color )
    {

        ImGuiColorEditFlags misc_flags = 0;
        ImGui::PushID( label.c_str() );
        char buf[128];
        ImFormatString( buf, IM_ARRAYSIZE( buf ), "##%s", label.c_str() );
        ImGui::AlignTextToFramePadding();
        ImGui::Text( label.c_str() );
        ImGui::SameLine( label_width );
        ImVec4 c = { color.r, color.g, color.b, color.a };

        float  lineHeight = GImGui->Font->FontSize + GImGui->Style.FramePadding.y * 2.0f;
        ImVec2 buttonSize = { lineHeight + 3.0f, lineHeight };

        bool open_popup = ImGui::ColorButton( buf, c, 0, buttonSize );

        static bool   saved_palette_init = true;
        static ImVec4 saved_palette[32]  = {};
        if( saved_palette_init )
        {
            for( int n = 0; n < IM_ARRAYSIZE( saved_palette ); n++ )
            {
                ImGui::ColorConvertHSVtoRGB( n / 31.0f, 0.8f, 0.8f, saved_palette[n].x, saved_palette[n].y, saved_palette[n].z );
                saved_palette[n].w = 1.0f; // Alpha
            }
            saved_palette_init = false;
        }

        static ImVec4 backup_color;

        if( open_popup )
        {
            ImGui::OpenPopup( "mypicker" );
            backup_color = c;
        }

        if( ImGui::BeginPopup( "mypicker", ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove ) )
        {
            ImGui::ColorPicker4(
                "##picker", (float *)&c, misc_flags | ImGuiColorEditFlags_NoSidePreview | ImGuiColorEditFlags_NoSmallPreview );
            ImGui::SameLine();

            ImGui::BeginGroup(); // Lock X position

            ImGui::Text( "Current" );
            ImGui::ColorButton(
                "##current", c, ImGuiColorEditFlags_NoPicker | ImGuiColorEditFlags_AlphaPreviewHalf, ImVec2( 60, 40 ) );

            ImGui::Text( "Previous" );
            if( ImGui::ColorButton( "##previous", backup_color, ImGuiColorEditFlags_NoPicker | ImGuiColorEditFlags_AlphaPreviewHalf,
                    ImVec2( 60, 40 ) ) )
                c = backup_color;

            ImGui::Separator();

            ImGui::Text( "Palette" );
            for( int n = 0; n < IM_ARRAYSIZE( saved_palette ); n++ )
            {
                ImGui::PushID( n );
                if( ( n % 8 ) != 0 ) ImGui::SameLine( 0.0f, ImGui::GetStyle().ItemSpacing.y );

                ImGuiColorEditFlags palette_button_flags =
                    ImGuiColorEditFlags_NoAlpha | ImGuiColorEditFlags_NoPicker | ImGuiColorEditFlags_NoTooltip;
                if( ImGui::ColorButton( "##palette", saved_palette[n], palette_button_flags, ImVec2( 20, 20 ) ) )
                    c = ImVec4( saved_palette[n].x, saved_palette[n].y, saved_palette[n].z, c.w ); // Preserve alpha!

                // Allow user to drop colors into each palette entry. Note that ColorButton() is already a
                // drag source by default, unless specifying the ImGuiColorEditFlags_NoDragDrop flag.
                if( ImGui::BeginDragDropTarget() )
                {
                    if( const ImGuiPayload *payload = ImGui::AcceptDragDropPayload( IMGUI_PAYLOAD_TYPE_COLOR_3F ) )
                        memcpy( (float *)&saved_palette[n], payload->Data, sizeof( float ) * 3 );

                    if( const ImGuiPayload *payload = ImGui::AcceptDragDropPayload( IMGUI_PAYLOAD_TYPE_COLOR_4F ) )
                        memcpy( (float *)&saved_palette[n], payload->Data, sizeof( float ) * 4 );
                    ImGui::EndDragDropTarget();
                }

                ImGui::PopID();
            }
            ImGui::EndGroup();
            ImGui::EndPopup();
        }
        ImGui::PopID();
        color = { c.x, c.y, c.z, c.w };
        return true;
    }

    bool ColorChooser( const std::string label, int label_width, math::vec3 &color )
    {
        math::vec4 l_Color = math::vec4( color, 1.0f );
        bool       x       = ColorChooser( label, label_width, l_Color );
        color              = l_Color;
        return x;
    }

    bool VectorComponentEditor( const std::string &label, math::vec3 &values, float resetValue, float columnWidth )
    {
        bool     Changed      = false;
        ImGuiIO &io           = ImGui::GetIO();
        auto     boldFont     = io.Fonts->Fonts[0];
        float    lineHeight   = GImGui->Font->FontSize + GImGui->Style.FramePadding.y * 2.0f;
        ImVec2   buttonSize   = { lineHeight + 3.0f, lineHeight };
        auto     l_WindowSize = GetAvailableContentSpace();

        ImGui::PushID( label.c_str() );

        auto l_TextSize0 = ImGui::CalcTextSize( label.c_str() );

        if( !label.empty() )
        {
            ImGui::AlignTextToFramePadding();
            ImGui::Text( label.c_str() );
            SameLine();
        }
        SetCursorPosition(
            ImGui::GetCursorPos() + ImVec2( ( columnWidth - l_TextSize0.x ) + ( l_TextSize0.x > 0.0f ? 10.0f : 0.0f ), 0.0f ) );

        float l_ItemWidth = ( l_WindowSize.x - columnWidth - 3 * buttonSize.x - 25.0f ) / 3.0f;
        ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, ImVec2{ 0, 0 } );

        ImGui::PushStyleColor( ImGuiCol_Button, ImVec4{ 0.8f, 0.1f, 0.15f, 1.0f } );
        ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ImVec4{ 0.9f, 0.2f, 0.2f, 1.0f } );
        ImGui::PushStyleColor( ImGuiCol_ButtonActive, ImVec4{ 0.8f, 0.1f, 0.15f, 1.0f } );
        ImGui::PushFont( boldFont );
        if( ImGui::Button( "X", buttonSize ) )
        {
            if( values.x != resetValue ) Changed = true;
            values.x = resetValue;
        }
        ImGui::PopFont();
        ImGui::PopStyleColor( 3 );
        SameLine();
        ImGui::SetNextItemWidth( l_ItemWidth );
        Changed |= ImGui::DragFloat( "##X", &values.x, 0.1f, 0.0f, 0.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp );
        SameLine();

        ImGui::PushStyleColor( ImGuiCol_Button, ImVec4{ 0.2f, 0.7f, 0.2f, 1.0f } );
        ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ImVec4{ 0.3f, 0.8f, 0.3f, 1.0f } );
        ImGui::PushStyleColor( ImGuiCol_ButtonActive, ImVec4{ 0.2f, 0.7f, 0.2f, 1.0f } );
        ImGui::PushFont( boldFont );
        if( ImGui::Button( "Y", buttonSize ) )
        {
            if( values.y != resetValue ) Changed = true;
            values.y = resetValue;
        }
        ImGui::PopFont();
        ImGui::PopStyleColor( 3 );
        SameLine();
        ImGui::SetNextItemWidth( l_ItemWidth );
        Changed |= ImGui::DragFloat( "##Y", &values.y, 0.1f, 0.0f, 0.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp );
        SameLine();

        ImGui::PushStyleColor( ImGuiCol_Button, ImVec4{ 0.1f, 0.25f, 0.8f, 1.0f } );
        ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ImVec4{ 0.2f, 0.35f, 0.9f, 1.0f } );
        ImGui::PushStyleColor( ImGuiCol_ButtonActive, ImVec4{ 0.1f, 0.25f, 0.8f, 1.0f } );
        ImGui::PushFont( boldFont );
        if( ImGui::Button( "Z", buttonSize ) )
        {
            if( values.z != resetValue ) Changed = true;
            values.z = resetValue;
        }
        ImGui::PopFont();
        ImGui::PopStyleColor( 3 );
        SameLine();
        ImGui::SetNextItemWidth( l_ItemWidth );
        Changed |= ImGui::DragFloat( "##Z", &values.z, 0.1f, 0.0f, 0.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp );

        ImGui::PopStyleVar();
        ImGui::PopID();

        return Changed;
    }

    bool VectorComponentEditor( const std::string &label, math::vec4 &values, float resetValue, float columnWidth )
    {
        bool     Changed      = false;
        ImGuiIO &io           = ImGui::GetIO();
        auto     boldFont     = io.Fonts->Fonts[0];
        float    lineHeight   = GImGui->Font->FontSize + GImGui->Style.FramePadding.y * 2.0f;
        ImVec2   buttonSize   = { lineHeight + 3.0f, lineHeight };
        auto     l_WindowSize = GetAvailableContentSpace();

        ImGui::PushID( label.c_str() );

        auto l_TextSize0 = ImGui::CalcTextSize( label.c_str() );

        if( !label.empty() )
        {
            ImGui::AlignTextToFramePadding();
            ImGui::Text( label.c_str() );
            SameLine();
        }
        SetCursorPosition(
            ImGui::GetCursorPos() + ImVec2( ( columnWidth - l_TextSize0.x ) + ( l_TextSize0.x > 0.0f ? 10.0f : 0.0f ), 0.0f ) );

        float l_ItemWidth = ( l_WindowSize.x - columnWidth - 4 * buttonSize.x - 25.0f ) / 4.0f;
        ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, ImVec2{ 0, 0 } );

        ImGui::PushStyleColor( ImGuiCol_Button, ImVec4{ 0.8f, 0.1f, 0.15f, 1.0f } );
        ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ImVec4{ 0.9f, 0.2f, 0.2f, 1.0f } );
        ImGui::PushStyleColor( ImGuiCol_ButtonActive, ImVec4{ 0.8f, 0.1f, 0.15f, 1.0f } );
        ImGui::PushFont( boldFont );
        if( ImGui::Button( "X", buttonSize ) )
        {
            if( values.x != resetValue ) Changed = true;
            values.x = resetValue;
        }
        ImGui::PopFont();
        ImGui::PopStyleColor( 3 );
        SameLine();
        ImGui::SetNextItemWidth( l_ItemWidth );
        Changed |= ImGui::DragFloat( "##X", &values.x, 0.1f, 0.0f, 0.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp );
        SameLine();

        ImGui::PushStyleColor( ImGuiCol_Button, ImVec4{ 0.2f, 0.7f, 0.2f, 1.0f } );
        ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ImVec4{ 0.3f, 0.8f, 0.3f, 1.0f } );
        ImGui::PushStyleColor( ImGuiCol_ButtonActive, ImVec4{ 0.2f, 0.7f, 0.2f, 1.0f } );
        ImGui::PushFont( boldFont );
        if( ImGui::Button( "Y", buttonSize ) )
        {
            if( values.y != resetValue ) Changed = true;
            values.y = resetValue;
        }
        ImGui::PopFont();
        ImGui::PopStyleColor( 3 );
        SameLine();
        ImGui::SetNextItemWidth( l_ItemWidth );
        Changed |= ImGui::DragFloat( "##Y", &values.y, 0.1f, 0.0f, 0.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp );
        SameLine();

        ImGui::PushStyleColor( ImGuiCol_Button, ImVec4{ 0.1f, 0.25f, 0.8f, 1.0f } );
        ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ImVec4{ 0.2f, 0.35f, 0.9f, 1.0f } );
        ImGui::PushStyleColor( ImGuiCol_ButtonActive, ImVec4{ 0.1f, 0.25f, 0.8f, 1.0f } );
        ImGui::PushFont( boldFont );
        if( ImGui::Button( "Z", buttonSize ) )
        {
            if( values.z != resetValue ) Changed = true;
            values.z = resetValue;
        }
        ImGui::PopFont();
        ImGui::PopStyleColor( 3 );
        SameLine();
        ImGui::SetNextItemWidth( l_ItemWidth );
        Changed |= ImGui::DragFloat( "##Z", &values.z, 0.1f, 0.0f, 0.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp );

        SameLine();

        ImGui::PushStyleColor( ImGuiCol_Button, ImVec4{ 0.1f, 0.25f, 0.8f, 1.0f } );
        ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ImVec4{ 0.2f, 0.35f, 0.9f, 1.0f } );
        ImGui::PushStyleColor( ImGuiCol_ButtonActive, ImVec4{ 0.1f, 0.25f, 0.8f, 1.0f } );
        ImGui::PushFont( boldFont );
        if( ImGui::Button( "W", buttonSize ) )
        {
            if( values.z != resetValue ) Changed = true;
            values.z = resetValue;
        }
        ImGui::PopFont();
        ImGui::PopStyleColor( 3 );
        SameLine();
        ImGui::SetNextItemWidth( l_ItemWidth );
        Changed |= ImGui::DragFloat( "##Z", &values.z, 0.1f, 0.0f, 0.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp );

        ImGui::PopStyleVar();
        ImGui::PopID();

        return Changed;
    }

    bool VectorComponentEditor( const std::string &label, uint32_t ID, math::vec2 &values, float a_XMin, float a_XMax, float a_XStep,
        float a_YMin, float a_YMax, float a_YStep, float resetValue, float columnWidth )
    {
        bool     Changed      = false;
        ImGuiIO &io           = ImGui::GetIO();
        auto     boldFont     = io.Fonts->Fonts[0];
        float    lineHeight   = GImGui->Font->FontSize + GImGui->Style.FramePadding.y * 2.0f;
        ImVec2   buttonSize   = { lineHeight + 3.0f, lineHeight };
        auto     l_WindowSize = GetAvailableContentSpace();

        ImGui::PushID( label.c_str() );

        auto l_TextSize0 = ImGui::CalcTextSize( label.c_str() );

        if( !label.empty() )
        {
            ImGui::AlignTextToFramePadding();
            ImGui::Text( label.c_str() );
            SameLine();
        }
        SetCursorPosition(
            ImGui::GetCursorPos() + ImVec2( ( columnWidth - l_TextSize0.x ) + ( l_TextSize0.x > 0.0f ? 10.0f : 0.0f ), 0.0f ) );

        float l_ItemWidth = ( l_WindowSize.x - columnWidth - 3 * buttonSize.x - 25.0f ) / 2.5f;
        ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, ImVec2{ 0, 0 } );

        ImGui::PushStyleColor( ImGuiCol_Button, ImVec4{ 0.8f, 0.1f, 0.15f, 1.0f } );
        ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ImVec4{ 0.9f, 0.2f, 0.2f, 1.0f } );
        ImGui::PushStyleColor( ImGuiCol_ButtonActive, ImVec4{ 0.8f, 0.1f, 0.15f, 1.0f } );
        ImGui::PushFont( boldFont );
        if( ImGui::Button( "X", buttonSize ) )
        {
            if( values.x != resetValue ) Changed = true;
            values.x = resetValue;
        }
        ImGui::PopFont();
        ImGui::PopStyleColor( 3 );
        SameLine();
        ImGui::SetNextItemWidth( l_ItemWidth );
        Changed |= ImGui::DragFloat( "##X", &values.x, .05f, a_XMin, a_XMax, "%.2f", ImGuiSliderFlags_AlwaysClamp );
        SameLine();

        ImGui::PushStyleColor( ImGuiCol_Button, ImVec4{ 0.2f, 0.7f, 0.2f, 1.0f } );
        ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ImVec4{ 0.3f, 0.8f, 0.3f, 1.0f } );
        ImGui::PushStyleColor( ImGuiCol_ButtonActive, ImVec4{ 0.2f, 0.7f, 0.2f, 1.0f } );
        ImGui::PushFont( boldFont );
        if( ImGui::Button( "Y", buttonSize ) )
        {
            if( values.y != resetValue ) Changed = true;
            values.y = resetValue;
        }
        ImGui::PopFont();
        ImGui::PopStyleColor( 3 );
        SameLine();
        ImGui::SetNextItemWidth( l_ItemWidth );
        Changed |= ImGui::DragFloat( "##Y", &values.y, 0.05f, a_YMin, a_YMax, "%.2f", ImGuiSliderFlags_AlwaysClamp );

        ImGui::PopStyleVar();
        ImGui::PopID();

        return Changed;
    }

} // namespace SE::Core::UI
