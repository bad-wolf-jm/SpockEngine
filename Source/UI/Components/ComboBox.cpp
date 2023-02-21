#include "ComboBox.h"

namespace SE::Core
{
    UIComboBox::UIComboBox( std::vector<std::string> const &aItems )
        : mItems{ aItems } {};

    void UIComboBox::PushStyles() {}
    void UIComboBox::PopStyles() {}

    void UIComboBox::OnChange( std::function<void( int aIndex )> aOnChange ) { mOnChange = aOnChange; }

    void UIComboBox::SetItemList( std::vector<std::string> aItems ) { mItems = aItems; }

    void UIComboBox::DrawContent( ImVec2 aPosition, ImVec2 aSize )
    {
        bool lEnabled = mIsEnabled;

        ImGui::SetCursorPos( GetContentAlignedposition( mHAlign, mVAlign, aPosition, ImGui::CalcTextSize( "mText.c_str()" ), aSize ) );

        if( mCurrentItem >= mItems.size() ) mCurrentItem = mItems.size() - 1;

        if( ImGui::BeginCombo( "##", mItems[mCurrentItem].c_str() ) )
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
