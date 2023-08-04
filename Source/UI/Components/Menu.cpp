#include "Menu.h"

namespace SE::Core
{
    UIMenuItem::UIMenuItem( string_t const &aText )
        : mText{ aText }
    {
    }

    UIMenuItem::UIMenuItem( string_t const &aText, string_t const &aShortcut )
        : mText{ aText }
        , mShortcut{ aShortcut }
    {
    }

    void UIMenuItem::PushStyles()
    {
    }
    void UIMenuItem::PopStyles()
    {
    }

    void UIMenuItem::SetText( string_t const &aText )
    {
        mText = aText;
    }
    void UIMenuItem::SetShortcut( string_t const &aShortcut )
    {
        mShortcut = aShortcut;
    }
    void UIMenuItem::SetTextColor( math::vec4 aColor )
    {
        mTextColor = ImVec4{ aColor.x, aColor.y, aColor.z, aColor.w };
    }

    void UIMenuItem::OnTrigger( std::function<void()> aOnTrigger )
    {
        mOnTrigger = aOnTrigger;
    }

    ImVec2 UIMenuItem::RequiredSize()
    {
        auto lTextSize = ImGui::CalcTextSize( mText.c_str() );

        return lTextSize;
    }

    void UIMenuItem::DrawContent( ImVec2 aPosition, ImVec2 aSize )
    {
        bool lTextColorSet =
            ( ( mTextColor.x != 0.0f ) || ( mTextColor.y != 0.0f ) || ( mTextColor.z != 0.0f ) || ( mTextColor.w != 0.0f ) );
        if( lTextColorSet )
            ImGui::PushStyleColor( ImGuiCol_Text, mTextColor );

        const char *lShortcut = mShortcut.empty() ? nullptr : mShortcut.c_str();

        bool lEnabled = mIsEnabled;

        bool lSelected = false;
        ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, ImVec2( 20.0f, 0.0f ) );
        ImGui::SetCursorPosX( ImGui::GetCursorPosX() + 30.0f );
        if( ImGui::MenuItem( mText.c_str(), lShortcut, &lSelected ) && mOnTrigger && lEnabled )
            mOnTrigger();
        ImGui::PopStyleVar();

        if( lTextColorSet )
            ImGui::PopStyleColor();
    }

    void UIMenuSeparator::PushStyles()
    {
        ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, ImVec2( 0.0f, 0.0f ) );
    }
    void UIMenuSeparator::PopStyles()
    {
        ImGui::PopStyleVar();
    }

    ImVec2 UIMenuSeparator::RequiredSize()
    {
        auto lTextSize = ImGui::CalcTextSize( mText.c_str() );

        return lTextSize;
    }

    void UIMenuSeparator::DrawContent( ImVec2 aPosition, ImVec2 aSize )
    {
        ImGuiWindow *window = ImGui::GetCurrentWindow();

        float x1 = window->DC.CursorPos.x;
        float x2 = window->DC.CursorPos.x + window->Size.x;

        ImVec2 lLineStart = ImVec2{ x1, window->DC.CursorPos.y + 1.5f };
        ImVec2 lLineEnd   = ImVec2{ x2, window->DC.CursorPos.y + 1.5f };

        auto lPos = ImGui::GetCursorPos();
        window->DrawList->AddLine( lLineStart, lLineEnd, ImGui::GetColorU32( ImGuiCol_Separator ) );

        ImGui::SetCursorPos( lPos + ImVec2{ 0.0f, 4.0f } );
    }

    UIMenu::UIMenu( string_t const &aText )
        : UIMenuItem( aText )
    {
    }

    void UIMenu::PushStyles()
    {
        ImGui::PushStyleVar( ImGuiStyleVar_PopupRounding, 5.0f );
        ImGui::PushStyleColor( ImGuiCol_Border, ImGui::GetColorU32( ImGuiCol_Separator ) );
    }
    void UIMenu::PopStyles()
    {
        ImGui::PopStyleVar();
        ImGui::PopStyleColor();
    }

    ImVec2 UIMenu::RequiredSize()
    {
        auto lTextSize = ImGui::CalcTextSize( mText.c_str() );

        return lTextSize;
    }

    void UIMenu::DrawContent( ImVec2 aPosition, ImVec2 aSize )
    {
        bool lTextColorSet =
            ( ( mTextColor.x != 0.0f ) || ( mTextColor.y != 0.0f ) || ( mTextColor.z != 0.0f ) || ( mTextColor.w != 0.0f ) );
        if( lTextColorSet )
            ImGui::PushStyleColor( ImGuiCol_Text, mTextColor );

        if( ImGui::BeginMenu( mText.c_str(), &mIsEnabled ) )
        {
            ImGui::SetCursorPosY( ImGui::GetCursorPosY() + 10 );
            for( auto &lItem : mActions )
            {
                lItem->Update( ImGui::GetCursorPos(), ImVec2{} );
                ImGui::SetCursorPosY( ImGui::GetCursorPosY() + 10 );
            }
            ImGui::EndMenu();
        }

        if( lTextColorSet )
            ImGui::PopStyleColor();
    }

    UIMenuItem *UIMenu::AddActionRaw( string_t const &aText, string_t const &aShortcut )
    {
        UIMenuItem *lNewItem = new UIMenuItem( aText, aShortcut );
        mActions.push_back( lNewItem );

        return lNewItem;
    }

    UIMenu *UIMenu::AddMenuRaw( string_t const &aText )
    {
        UIMenu *lNewItem = new UIMenu( aText );
        mActions.push_back( lNewItem );

        return lNewItem;
    }

    UIMenuItem *UIMenu::AddSeparatorRaw()
    {
        UIMenuSeparator *lNewItem = new UIMenuSeparator();
        mActions.push_back( lNewItem );

        return lNewItem;
    }

    Ref<UIMenuItem> UIMenu::AddAction( string_t const &aText, string_t const &aShortcut )
    {
        Ref<UIMenuItem> lNewItem( AddActionRaw( aText, aShortcut ) );

        return lNewItem;
    }

    Ref<UIMenu> UIMenu::AddMenu( string_t const &aText )
    {
        Ref<UIMenu> lNewItem( AddMenuRaw( aText ) );

        return lNewItem;
    }

    Ref<UIMenuItem> UIMenu::AddSeparator()
    {
        Ref<UIMenuItem> lNewItem( AddSeparatorRaw() );

        return lNewItem;
    }

    void UIMenu::Update()
    {
        UIComponent::Update( ImGui::GetCursorPos(), ImVec2{} );
    }
} // namespace SE::Core