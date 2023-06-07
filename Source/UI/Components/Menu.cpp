#include "Menu.h"

#include "DotNet/Runtime.h"
namespace SE::Core
{
    UIMenuItem::UIMenuItem( std::string const &aText )
        : mText{ aText }
    {
    }

    UIMenuItem::UIMenuItem( std::string const &aText, std::string const &aShortcut )
        : mText{ aText }
        , mShortcut{ aShortcut }
    {
    }

    void UIMenuItem::PushStyles() {}
    void UIMenuItem::PopStyles() {}

    void UIMenuItem::SetText( std::string const &aText ) { mText = aText; }
    void UIMenuItem::SetShortcut( std::string const &aShortcut ) { mShortcut = aShortcut; }
    void UIMenuItem::SetTextColor( math::vec4 aColor ) { mTextColor = ImVec4{ aColor.x, aColor.y, aColor.z, aColor.w }; }

    void UIMenuItem::OnTrigger( std::function<void()> aOnTrigger ) { mOnTrigger = aOnTrigger; }

    ImVec2 UIMenuItem::RequiredSize()
    {
        auto lTextSize = ImGui::CalcTextSize( mText.c_str() );

        return lTextSize;
    }

    void UIMenuItem::DrawContent( ImVec2 aPosition, ImVec2 aSize )
    {
        bool lTextColorSet =
            ( ( mTextColor.x != 0.0f ) || ( mTextColor.y != 0.0f ) || ( mTextColor.z != 0.0f ) || ( mTextColor.w != 0.0f ) );
        if( lTextColorSet ) ImGui::PushStyleColor( ImGuiCol_Text, mTextColor );

        const char *lShortcut = mShortcut.empty() ? nullptr : mShortcut.c_str();

        bool lEnabled = mIsEnabled;

        bool lSelected = false;
        ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, ImVec2( 30.0f, 10.0f ) );
        if( ImGui::MenuItem( mText.c_str(), lShortcut, &lSelected ) && mOnTrigger && lEnabled ) mOnTrigger();
        ImGui::PopStyleVar();

        if( lTextColorSet ) ImGui::PopStyleColor();
    }


    void UIMenuSeparator::PushStyles() {}
    void UIMenuSeparator::PopStyles() {}

    ImVec2 UIMenuSeparator::RequiredSize()
    {
        auto lTextSize = ImGui::CalcTextSize( mText.c_str() );

        return lTextSize;
    }

    void UIMenuSeparator::DrawContent( ImVec2 aPosition, ImVec2 aSize ) { ImGui::Separator(); }

    UIMenu::UIMenu( std::string const &aText )
        : UIMenuItem( aText )
    {
    }

    void UIMenu::PushStyles() {}
    void UIMenu::PopStyles() {}

    ImVec2 UIMenu::RequiredSize()
    {
        auto lTextSize = ImGui::CalcTextSize( mText.c_str() );

        return lTextSize;
    }

    void UIMenu::DrawContent( ImVec2 aPosition, ImVec2 aSize )
    {
        bool lTextColorSet =
            ( ( mTextColor.x != 0.0f ) || ( mTextColor.y != 0.0f ) || ( mTextColor.z != 0.0f ) || ( mTextColor.w != 0.0f ) );
        if( lTextColorSet ) ImGui::PushStyleColor( ImGuiCol_Text, mTextColor );

        if( ImGui::BeginMenu( mText.c_str(), &mIsEnabled ) )
        {
            for( auto &lItem : mActions ) lItem->Update( ImGui::GetCursorPos(), ImVec2{} );

            ImGui::EndMenu();
        }

        if( lTextColorSet ) ImGui::PopStyleColor();
    }

    UIMenuItem *UIMenu::AddActionRaw( std::string const &aText, std::string const &aShortcut )
    {
        UIMenuItem *lNewItem = new UIMenuItem( aText, aShortcut );
        mActions.push_back( lNewItem );

        return lNewItem;
    }

    UIMenu *UIMenu::AddMenuRaw( std::string const &aText )
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

    Ref<UIMenuItem> UIMenu::AddAction( std::string const &aText, std::string const &aShortcut )
    {
        Ref<UIMenuItem> lNewItem( AddActionRaw( aText, aShortcut ) );

        return lNewItem;
    }

    Ref<UIMenu> UIMenu::AddMenu( std::string const &aText )
    {
        Ref<UIMenu> lNewItem( AddMenuRaw( aText ) );

        return lNewItem;
    }

    Ref<UIMenuItem> UIMenu::AddSeparator()
    {
        Ref<UIMenuItem> lNewItem( AddSeparatorRaw() );

        return lNewItem;
    }

    void UIMenu::Update() { UIComponent::Update( ImGui::GetCursorPos(), ImVec2{} ); }
} // namespace SE::Core