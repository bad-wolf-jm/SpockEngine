#include "ComboBox.h"

namespace SE::Core
{
    UIComboBox::UIComboBox( std::vector<std::string> const &aItems )
        : mItems{ aItems } {};

    ImVec2 UIComboBox::RequiredSize()
    {
        float lTextWidth = 0.0f;

        for( auto const &lItem : mItems ) lTextWidth = math::max( ImGui::CalcTextSize( lItem.c_str() ).x, lTextWidth );

        const float lArrowSize = ImGui::GetFrameHeight();
        return ImVec2{ lTextWidth + lArrowSize, ImGui::GetFrameHeight() };
    }

    void UIComboBox::PushStyles() {}
    void UIComboBox::PopStyles() {}

    void UIComboBox::OnChange( std::function<void( int aIndex )> aOnChange ) { mOnChange = aOnChange; }

    void UIComboBox::SetItemList( std::vector<std::string> aItems ) { mItems = aItems; }

    void UIComboBox::DrawContent( ImVec2 aPosition, ImVec2 aSize )
    {
        bool lEnabled = mIsEnabled;

        auto lItemSize = ImVec2{ aSize.x, RequiredSize().y };

        ImGui::SetCursorPos(
            GetContentAlignedposition( eHorizontalAlignment::LEFT, eVerticalAlignment::CENTER, aPosition, lItemSize, aSize ) );

        if( mCurrentItem >= mItems.size() ) mCurrentItem = mItems.size() - 1;

        bool lBeginCombo = false;

        ImGui::SetNextItemWidth( aSize.x );
        if( ( mItems.size() == 0 ) )
        {
            if( ImGui::BeginCombo( "##", "No Items" ) ) ImGui::EndCombo();
        }
        else if( ImGui::BeginCombo( "##", mItems[mCurrentItem].c_str() ) )
        {
            bool lChanged = false;

            for( int n = 0; n < mItems.size(); n++ )
            {
                bool lIsSelected = ( mCurrentItem == n );
                if( ImGui::Selectable( mItems[n].c_str(), lIsSelected ) )
                {
                    mCurrentItem = n;
                    lChanged |= !lIsSelected;
                }
                if( lIsSelected ) ImGui::SetItemDefaultFocus();
            }

            if( lChanged && mOnChange && lEnabled ) mOnChange( mCurrentItem );

            ImGui::EndCombo();
        }
    }
} // namespace SE::Core
