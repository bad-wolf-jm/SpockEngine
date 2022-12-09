#pragma once

#include <functional>
#include <imgui.h>

#include "Core/Math/Types.h"
#include "Core/Types.h"
#include "UI.h"

// #include "Core/GraphicContext//Texture2D.h"

#include "Graphics/Vulkan/VkSampler2D.h"

namespace SE::Core::UI
{

    void HelpMarker( const char *desc );

    void Separator();

    bool ImageButton( SE::Core::UI::ImageHandle a_Texture, math::vec2 a_Size, math::vec4 a_Rect );

    bool Button( const char *a_Label, math::vec2 a_Size );

    void ToggleButton( const char *a_OnLabel, const char *a_OffLabel, bool *l_state, math::vec2 a_Size );
    void ToggleButton( const char *a_OnLabel, const char *a_OffLabel, bool l_State, math::vec2 a_Size,
                       std::function<void( bool )> a_Action );
    void ToggleButton( const char *a_OnLabel, const char *a_OffLabel, ImFont *a_Font, bool l_State, math::vec2 a_Size,
                       std::function<void( bool )> a_Action );

    void BulletText( const char *a_Text );

    bool Slider( std::string a_Title, const char *a_DisplayTemplate, uint16_t a_MinValue, uint16_t a_MaxValue,
                 uint16_t *a_CurrentValue );
    bool Slider( std::string a_Title, const char *a_DisplayTemplate, uint32_t a_MinValue, uint32_t a_MaxValue,
                 uint32_t *a_CurrentValue );
    bool Slider( std::string a_Title, const char *a_DisplayTemplate, float a_MinValue, float a_MaxValue, float *a_CurrentValue );
    bool Slider( std::string a_Title, const char *a_DisplayTemplate, int32_t a_MinValue, int32_t a_MaxValue, int32_t *a_CurrentValue );

    void Checkbox( const char *label, bool *a_Value );

    void ColorButton( math::vec4 &a_Color, math::vec2 a_Size );

    bool MenuItem( const char *label, const char *shortcut );
    bool SelectionMenuItem( const char *label, const char *shortcut, bool *selected );

    void Image( SE::Core::UI::ImageHandle a_Texture, math::vec2 a_Size, math::vec4 a_Rect );
    void Image( ImageHandle texture, math::vec2 size );
    void Image( ImageHandle texture, math::ivec2 size );

    void Menu( const char *a_Label, std::function<void()> a_Elements );

    enum ManipulationType : uint32_t
    {
        ROTATION    = 0,
        TRANSLATION = 1,
        SCALE       = 2
    };

    struct ManipulationConfig
    {
        ManipulationType Type;
        math::mat4       Projection;
        math::mat4       WorldTransform;
        math::vec2       ViewportPosition;
        math::vec2       ViewportSize;
    };

    void Manipulate( ManipulationConfig a_Config, math::mat4 &a_Tranform );
    bool ViewManipulate( math::vec3 a_CameraPosition, math::mat4 &a_CameraView, math::vec2 a_Position );

    template <typename _Ty>
    class ComboBox
    {
      public:
        std::string              ID          = "";
        uint32_t                 CurrentItem = 0;
        std::vector<_Ty>         Values      = {};
        std::vector<std::string> Labels      = {};
        math::vec2               Size        = { 100.0f, 30.0f };
        bool                     Changed     = false;

      public:
        ComboBox() = default;
        ComboBox( std::string a_ID )
            : ID{ a_ID } {};

        _Ty GetValue() { return Values[CurrentItem]; }

        void Display()
        {
            if( Labels.size() != Values.size() ) return;

            if( CurrentItem >= Values.size() ) CurrentItem = Values.size() - 1;

            if( ImGui::BeginCombo( ID.c_str(), Labels[CurrentItem].c_str() ) )
            {
                Changed = false;
                for( int n = 0; n < Labels.size(); n++ )
                {
                    bool l_ItemIsSelected = ( CurrentItem == n );
                    if( ImGui::Selectable( Labels[n].c_str(), l_ItemIsSelected ) )
                    {
                        CurrentItem = n;
                        Changed |= !l_ItemIsSelected;
                    }
                    if( l_ItemIsSelected ) ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }
        }
    };

    template <typename T>
    static bool DragFloat( const std::string label, T &value, int label_width, float step, float min, float max )
    {
        char buf[128];
        ImFormatString( buf, IM_ARRAYSIZE( buf ), "##%s", label.c_str() );
        float v = (float)value;
        ImGui::AlignTextToFramePadding();
        ImGui::Text( label.c_str() );
        ImGui::SameLine( label_width );
        bool changed = ImGui::DragFloat( buf, &v, step, min, max, NULL );
        ImGui::SameLine();
        ImFormatString( buf, IM_ARRAYSIZE( buf ), "-##%s", label.c_str() );
        if( ImGui::Button( buf, ImVec2{ 22.0, 22.0 } ) )
        {
            v -= step;
            changed = true;
        }
        ImGui::SameLine();
        ImFormatString( buf, IM_ARRAYSIZE( buf ), "+##%s", label.c_str() );
        if( ImGui::Button( buf, ImVec2{ 22.0, 22.0 } ) )
        {
            v += step;
            changed = true;
        }
        value = T{ v };
        return changed;
    }

    template <typename T>
    static bool DragUInt( const std::string label, T &value, int label_width, float step, uint32_t min, uint32_t max )
    {
        char buf[128];
        ImFormatString( buf, IM_ARRAYSIZE( buf ), "##%s", label.c_str() );
        uint32_t v = (uint32_t)value;
        ImGui::AlignTextToFramePadding();
        ImGui::Text( label.c_str() );
        ImGui::SameLine( label_width );
        bool changed = ImGui::DragScalar( buf, ImGuiDataType_U32, (void *)&v, step, (void *)&min, (void *)&max, NULL );

        ImGui::SameLine();
        ImFormatString( buf, IM_ARRAYSIZE( buf ), "-##%s", label.c_str() );
        if( ImGui::Button( buf, ImVec2{ 22.0, 22.0 } ) )
        {
            v -= step;
            changed = true;
        }
        ImGui::SameLine();
        ImFormatString( buf, IM_ARRAYSIZE( buf ), "+##%s", label.c_str() );
        if( ImGui::Button( buf, ImVec2{ 22.0, 22.0 } ) )
        {
            v += step;
            changed = true;
        }

        value = T{ v };
        return changed;
    }

    bool ColorChooser( const std::string label, int label_width, math::vec4 &color );
    bool ColorChooser( const std::string label, int label_width, math::vec3 &color );

    bool VectorComponentEditor( const std::string &a_Label, math::vec3 &values, float a_ResetValue, float a_ColumnWidth );
    bool VectorComponentEditor( const std::string &label, math::vec4 &values, float resetValue, float columnWidth );
    bool VectorComponentEditor( const std::string &a_Label, uint32_t ID, math::vec2 &values, float a_XMin, float a_XMax, float a_XStep,
                                float a_YMin, float a_YMax, float a_YStep, float a_ResetValue, float a_ColumnWidth );

} // namespace SE::Core::UI
