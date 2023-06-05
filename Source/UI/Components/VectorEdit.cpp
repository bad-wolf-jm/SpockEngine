#include "VectorEdit.h"
// #include "ImGuizmo.h"
// #include <vulkan/vulkan.h>

namespace SE::Core
{
    static constexpr ImVec4 gXColors[] = { ImVec4{ 0.8f, 0.1f, 0.15f, 1.0f }, ImVec4{ 0.9f, 0.2f, 0.2f, 1.0f },
                                           ImVec4{ 0.8f, 0.1f, 0.15f, 1.0f } };

    static constexpr ImVec4 gYColors[] = { ImVec4{ 0.2f, 0.7f, 0.2f, 1.0f }, ImVec4{ 0.3f, 0.8f, 0.3f, 1.0f },
                                           ImVec4{ 0.2f, 0.7f, 0.2f, 1.0f } };

    static constexpr ImVec4 gZColors[] = { ImVec4{ 0.1f, 0.25f, 0.8f, 1.0f }, ImVec4{ 0.2f, 0.35f, 0.9f, 1.0f },
                                           ImVec4{ 0.1f, 0.25f, 0.8f, 1.0f } };

    static constexpr ImVec4 gWColors[] = { ImVec4{ 0.1f, 0.25f, 0.8f, 1.0f }, ImVec4{ 0.2f, 0.35f, 0.9f, 1.0f },
                                           ImVec4{ 0.1f, 0.25f, 0.8f, 1.0f } };

    static bool EditVectorComponent( const char *aLabel, const char *aFormat, float *aValue, float aResetValue, ImVec2 aButtonSize,
                                     ImVec2 *aColors )
    {
        bool lHasChanged = false;

        ImGui::PushStyleColor( ImGuiCol_Button, aColors[0] );
        ImGui::PushStyleColor( ImGuiCol_ButtonHovered, aColors[1] );
        ImGui::PushStyleColor( ImGuiCol_ButtonActive, aColors[2] );
        if( ImGui::Button( aLabel, aButtonSize ) )
        {
            if( *aValues != aResetValue )
            {
                lHasChanged = true;
                *aValues    = aResetValue;
            }
        }
        ImGui::PopStyleColor( 3 );

        SameLine();

        ImGui::PushID( aLabel );
        lHasChanged |= ImGui::DragFloat( "##INPUT", aValue, 0.1f, 0.0f, 0.0f, aFormat, ImGuiSliderFlags_AlwaysClamp );
        ImGui::PopID();
    }

    static bool VectorComponentEditor( const char *aFormat, int aDimension, float *aValues, float *aResetValue )
    {
        bool   lHasChanged = false;
        float  lLineHeight = GImGui->Font->FontSize + GImGui->Style.FramePadding.y * 2.0f;
        ImVec2 lButtonSize = { lLineHeight + 3.0f, lLineHeight };
        auto   lWindowSize = GetAvailableContentSpace();

        if( aDimension >= 1 ) lHasChanged |= EditVectorComponent( "X", aFormat, &aValues[0], aResetValue[0], lButtonSize, gXColors );
        if( aDimension >= 2 ) lHasChanged |= EditVectorComponent( "Y", aFormat, &aValues[1], aResetValue[1], lButtonSize, gYColors );
        if( aDimension >= 3 ) lHasChanged |= EditVectorComponent( "Z", aFormat, &aValues[2], aResetValue[2], lButtonSize, gZColors );
        if( aDimension >= 4 ) lHasChanged |= EditVectorComponent( "W", aFormat, &aValues[3], aResetValue[3], lButtonSize, gWColors );

        return lHasChanged;
    }

    // bool VectorComponentEditor( const std::string &label, math::vec4 &aValues, float aResetValue, float columnWidth )
    // {
    //     bool     lHasChanged      = false;
    //     ImGuiIO &io           = ImGui::GetIO();
    //     auto     boldFont     = io.Fonts->Fonts[0];
    //     float    lLineHeight   = GImGui->Font->FontSize + GImGui->Style.FramePadding.y * 2.0f;
    //     ImVec2   lButtonSize   = { lLineHeight + 3.0f, lLineHeight };
    //     auto     lWindowSize = GetAvailableContentSpace();

    //     ImGui::PushID( label.c_str() );

    //     auto l_TextSize0 = ImGui::CalcTextSize( label.c_str() );

    //     if( !label.empty() )
    //     {
    //         ImGui::AlignTextToFramePadding();
    //         ImGui::Text( label.c_str() );
    //         SameLine();
    //     }
    //     SetCursorPosition( ImGui::GetCursorPos() +
    //                        ImVec2( ( columnWidth - l_TextSize0.x ) + ( l_TextSize0.x > 0.0f ? 10.0f : 0.0f ), 0.0f ) );

    //     float l_ItemWidth = ( lWindowSize.x - columnWidth - 4 * lButtonSize.x - 25.0f ) / 4.0f;
    //     ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, ImVec2{ 0, 0 } );

    //     ImGui::PushStyleColor( ImGuiCol_Button, ImVec4{ 0.8f, 0.1f, 0.15f, 1.0f } );
    //     ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ImVec4{ 0.9f, 0.2f, 0.2f, 1.0f } );
    //     ImGui::PushStyleColor( ImGuiCol_ButtonActive, ImVec4{ 0.8f, 0.1f, 0.15f, 1.0f } );
    //     ImGui::PushFont( boldFont );
    //     if( ImGui::Button( "X", lButtonSize ) )
    //     {
    //         if( aValues.x != aResetValue ) lHasChanged = true;
    //         aValues.x = aResetValue;
    //     }
    //     ImGui::PopFont();
    //     ImGui::PopStyleColor( 3 );
    //     SameLine();
    //     ImGui::SetNextItemWidth( l_ItemWidth );
    //     lHasChanged |= ImGui::DragFloat( "##X", &aValues.x, 0.1f, 0.0f, 0.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp );
    //     SameLine();

    //     ImGui::PushStyleColor( ImGuiCol_Button, ImVec4{ 0.2f, 0.7f, 0.2f, 1.0f } );
    //     ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ImVec4{ 0.3f, 0.8f, 0.3f, 1.0f } );
    //     ImGui::PushStyleColor( ImGuiCol_ButtonActive, ImVec4{ 0.2f, 0.7f, 0.2f, 1.0f } );
    //     ImGui::PushFont( boldFont );
    //     if( ImGui::Button( "Y", lButtonSize ) )
    //     {
    //         if( aValues.y != aResetValue ) lHasChanged = true;
    //         aValues.y = aResetValue;
    //     }
    //     ImGui::PopFont();
    //     ImGui::PopStyleColor( 3 );
    //     SameLine();
    //     ImGui::SetNextItemWidth( l_ItemWidth );
    //     lHasChanged |= ImGui::DragFloat( "##Y", &aValues.y, 0.1f, 0.0f, 0.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp );
    //     SameLine();

    //     ImGui::PushStyleColor( ImGuiCol_Button, ImVec4{ 0.1f, 0.25f, 0.8f, 1.0f } );
    //     ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ImVec4{ 0.2f, 0.35f, 0.9f, 1.0f } );
    //     ImGui::PushStyleColor( ImGuiCol_ButtonActive, ImVec4{ 0.1f, 0.25f, 0.8f, 1.0f } );
    //     ImGui::PushFont( boldFont );
    //     if( ImGui::Button( "Z", lButtonSize ) )
    //     {
    //         if( aValues.z != aResetValue ) lHasChanged = true;
    //         aValues.z = aResetValue;
    //     }
    //     ImGui::PopFont();
    //     ImGui::PopStyleColor( 3 );
    //     SameLine();
    //     ImGui::SetNextItemWidth( l_ItemWidth );
    //     lHasChanged |= ImGui::DragFloat( "##Z", &aValues.z, 0.1f, 0.0f, 0.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp );

    //     SameLine();

    //     ImGui::PushStyleColor( ImGuiCol_Button, ImVec4{ 0.1f, 0.25f, 0.8f, 1.0f } );
    //     ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ImVec4{ 0.2f, 0.35f, 0.9f, 1.0f } );
    //     ImGui::PushStyleColor( ImGuiCol_ButtonActive, ImVec4{ 0.1f, 0.25f, 0.8f, 1.0f } );
    //     ImGui::PushFont( boldFont );
    //     if( ImGui::Button( "W", lButtonSize ) )
    //     {
    //         if( aValues.z != aResetValue ) lHasChanged = true;
    //         aValues.z = aResetValue;
    //     }
    //     ImGui::PopFont();
    //     ImGui::PopStyleColor( 3 );
    //     SameLine();
    //     ImGui::SetNextItemWidth( l_ItemWidth );
    //     lHasChanged |= ImGui::DragFloat( "##Z", &aValues.z, 0.1f, 0.0f, 0.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp );

    //     ImGui::PopStyleVar();
    //     ImGui::PopID();

    //     return lHasChanged;
    // }

    // bool VectorComponentEditor( const std::string &label, uint32_t ID, math::vec2 &aValues, float a_XMin, float a_XMax, float
    // a_XStep,
    //                             float a_YMin, float a_YMax, float a_YStep, float aResetValue, float columnWidth )
    // {
    //     bool     lHasChanged      = false;
    //     ImGuiIO &io           = ImGui::GetIO();
    //     auto     boldFont     = io.Fonts->Fonts[0];
    //     float    lLineHeight   = GImGui->Font->FontSize + GImGui->Style.FramePadding.y * 2.0f;
    //     ImVec2   lButtonSize   = { lLineHeight + 3.0f, lLineHeight };
    //     auto     lWindowSize = GetAvailableContentSpace();

    //     ImGui::PushID( label.c_str() );

    //     auto l_TextSize0 = ImGui::CalcTextSize( label.c_str() );

    //     if( !label.empty() )
    //     {
    //         ImGui::AlignTextToFramePadding();
    //         ImGui::Text( label.c_str() );
    //         SameLine();
    //     }
    //     SetCursorPosition( ImGui::GetCursorPos() +
    //                        ImVec2( ( columnWidth - l_TextSize0.x ) + ( l_TextSize0.x > 0.0f ? 10.0f : 0.0f ), 0.0f ) );

    //     float l_ItemWidth = ( lWindowSize.x - columnWidth - 3 * lButtonSize.x - 25.0f ) / 2.5f;
    //     ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, ImVec2{ 0, 0 } );

    //     ImGui::PushStyleColor( ImGuiCol_Button, ImVec4{ 0.8f, 0.1f, 0.15f, 1.0f } );
    //     ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ImVec4{ 0.9f, 0.2f, 0.2f, 1.0f } );
    //     ImGui::PushStyleColor( ImGuiCol_ButtonActive, ImVec4{ 0.8f, 0.1f, 0.15f, 1.0f } );
    //     ImGui::PushFont( boldFont );
    //     if( ImGui::Button( "X", lButtonSize ) )
    //     {
    //         if( aValues.x != aResetValue ) lHasChanged = true;
    //         aValues.x = aResetValue;
    //     }
    //     ImGui::PopFont();
    //     ImGui::PopStyleColor( 3 );
    //     SameLine();
    //     ImGui::SetNextItemWidth( l_ItemWidth );
    //     lHasChanged |= ImGui::DragFloat( "##X", &aValues.x, .05f, a_XMin, a_XMax, "%.2f", ImGuiSliderFlags_AlwaysClamp );
    //     SameLine();

    //     ImGui::PushStyleColor( ImGuiCol_Button, ImVec4{ 0.2f, 0.7f, 0.2f, 1.0f } );
    //     ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ImVec4{ 0.3f, 0.8f, 0.3f, 1.0f } );
    //     ImGui::PushStyleColor( ImGuiCol_ButtonActive, ImVec4{ 0.2f, 0.7f, 0.2f, 1.0f } );
    //     ImGui::PushFont( boldFont );
    //     if( ImGui::Button( "Y", lButtonSize ) )
    //     {
    //         if( aValues.y != aResetValue ) lHasChanged = true;
    //         aValues.y = aResetValue;
    //     }
    //     ImGui::PopFont();
    //     ImGui::PopStyleColor( 3 );
    //     SameLine();
    //     ImGui::SetNextItemWidth( l_ItemWidth );
    //     lHasChanged |= ImGui::DragFloat( "##Y", &aValues.y, 0.05f, a_YMin, a_YMax, "%.2f", ImGuiSliderFlags_AlwaysClamp );

    //     ImGui::PopStyleVar();
    //     ImGui::PopID();

    //     return lHasChanged;
    // }

} // namespace SE::Core