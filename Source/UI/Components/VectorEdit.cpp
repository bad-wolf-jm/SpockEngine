#include "VectorEdit.h"

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
                                     float aWidth, ImVec4 const *aColors )
    {
        bool lHasChanged = false;

        ImGui::PushStyleColor( ImGuiCol_Button, aColors[0] );
        ImGui::PushStyleColor( ImGuiCol_ButtonHovered, aColors[1] );
        ImGui::PushStyleColor( ImGuiCol_ButtonActive, aColors[2] );
        if( ImGui::Button( aLabel, aButtonSize ) )
        {
            if( *aValue != aResetValue )
            {
                lHasChanged = true;
                *aValue     = aResetValue;
            }
        }
        ImGui::PopStyleColor( 3 );

        ImGui::SameLine();

        ImGui::PushID( aLabel );
        ImGui::SetNextItemWidth( aWidth - aButtonSize.x );
        lHasChanged |= ImGui::DragFloat( "##INPUT", aValue, 0.1f, 0.0f, 0.0f, aFormat, ImGuiSliderFlags_AlwaysClamp );
        ImGui::PopID();

        return lHasChanged;
    }

    static bool VectorComponentEditor( const char *aFormat, int aDimension, float *aValues, float *aResetValue, float aWidth )
    {
        bool   lHasChanged = false;
        float  lLineHeight = GImGui->Font->FontSize + GImGui->Style.FramePadding.y * 2.0f;
        ImVec2 lButtonSize = { lLineHeight + 3.0f, lLineHeight };

        if( aDimension >= 1 )
            lHasChanged |= EditVectorComponent( "X", aFormat, &aValues[0], aResetValue[0], lButtonSize, aWidth, gXColors );
        if( aDimension >= 2 )
            lHasChanged |= EditVectorComponent( "Y", aFormat, &aValues[1], aResetValue[1], lButtonSize, aWidth, gYColors );
        if( aDimension >= 3 )
            lHasChanged |= EditVectorComponent( "Z", aFormat, &aValues[2], aResetValue[2], lButtonSize, aWidth, gZColors );
        if( aDimension >= 4 )
            lHasChanged |= EditVectorComponent( "W", aFormat, &aValues[3], aResetValue[3], lButtonSize, aWidth, gWColors );

        return lHasChanged;
    }

    UIVectorInputBase::UIVectorInputBase( int aDimension )
        : mDimension{ aDimension }
    {
    }

    void UIVectorInputBase::PushStyles() {}
    void UIVectorInputBase::PopStyles() {}

    ImVec2 UIVectorInputBase::RequiredSize() { return ImVec2{ 100.0f, ( GImGui->Font->FontSize + GImGui->Style.FramePadding.y * 2.0f ) * mDimension }; }

    void UIVectorInputBase::DrawContent( ImVec2 aPosition, ImVec2 aSize )
    {
        VectorComponentEditor( mFormat.c_str(), mDimension, (float *)&mValues, (float *)&mResetValues, aSize.x );
    }

} // namespace SE::Core