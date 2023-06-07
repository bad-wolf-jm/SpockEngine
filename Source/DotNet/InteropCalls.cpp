#include "InteropCalls.h"
#include "DotNet/Runtime.h"

#include "UI/Components/BaseImage.h"
#include "UI/Components/Button.h"
#include "UI/Components/CheckBox.h"
#include "UI/Components/ColorButton.h"
#include "UI/Components/ComboBox.h"
#include "UI/Components/Component.h"
#include "UI/Components/DropdownButton.h"
#include "UI/Components/Image.h"
#include "UI/Components/ImageButton.h"
#include "UI/Components/ImageToggleButton.h"
#include "UI/Components/Label.h"
#include "UI/Components/Menu.h"
#include "UI/Components/Plot.h"
#include "UI/Components/ProgressBar.h"
#include "UI/Components/PropertyValue.h"
#include "UI/Components/Slider.h"
#include "UI/Components/Table.h"
#include "UI/Components/TextInput.h"
#include "UI/Components/TextOverlay.h"
#include "UI/Components/TextToggleButton.h"
#include "UI/Components/TreeView.h"
#include "UI/Components/Workspace.h"
#include "UI/Components/VectorEdit.h"

#include "UI/Widgets/FileTree.h"

#include "UI/UI.h"

#include "UI/Layouts/Container.h"
#include "UI/Layouts/Splitter.h"
#include "UI/Layouts/StackLayout.h"
#include "UI/Layouts/ZLayout.h"

#include "UI/Dialog.h"
#include "UI/Form.h"
#include "UI/Layouts/BoxLayout.h"

namespace SE::Core::Interop
{

    void *UIBaseImage_Create()
    {
        auto lNewImage = new UIBaseImage();

        return static_cast<void *>( lNewImage );
    }

    void *UIBaseImage_CreateWithPath( void *aText, math::vec2 aSize )
    {
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );
        auto lNewImage = new UIBaseImage( lString, aSize );

        return static_cast<void *>( lNewImage );
    }

    void UIBaseImage_Destroy( void *aInstance ) { delete static_cast<UIBaseImage *>( aInstance ); }

    void UIBaseImage_SetImage( void *aInstance, void *aPath )
    {
        auto lInstance = static_cast<UIBaseImage *>( aInstance );
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aPath ) );

        lInstance->SetImage( lString );
    }

    void UIBaseImage_SetSize( void *aInstance, math::vec2 aSize )
    {
        auto lInstance = static_cast<UIBaseImage *>( aInstance );

        lInstance->SetSize( aSize );
    }

    math::vec2 UIBaseImage_GetSize( void *aInstance )
    {
        auto lInstance = static_cast<UIBaseImage *>( aInstance );
        auto lV        = lInstance->Size();

        return math::vec2{ lV.y, lV.y };
    }

    void UIBaseImage_SetTopLeft( void *aInstance, math::vec2 aTopLeft )
    {
        auto lInstance = static_cast<UIBaseImage *>( aInstance );

        lInstance->SetTopLeft( aTopLeft );
    }

    math::vec2 UIBaseImage_GetTopLeft( void *aInstance )
    {
        auto lInstance = static_cast<UIBaseImage *>( aInstance );
        auto lV        = lInstance->TopLeft();

        return math::vec2{ lV.y, lV.y };
    }

    void UIBaseImage_SetBottomRight( void *aInstance, math::vec2 aBottomRight )
    {
        auto lInstance = static_cast<UIBaseImage *>( aInstance );

        lInstance->SetBottomRight( aBottomRight );
    }

    math::vec2 UIBaseImage_GetBottomRight( void *aInstance )
    {
        auto lInstance = static_cast<UIBaseImage *>( aInstance );
        auto lV        = lInstance->BottomRight();

        return math::vec2{ lV.x, lV.y };
    }

    void UIBaseImage_SetTintColor( void *aInstance, math::vec4 aColor )
    {
        auto lInstance = static_cast<UIBaseImage *>( aInstance );

        lInstance->SetTintColor( aColor );
    }

    math::vec4 UIBaseImage_GetTintColor( void *aInstance )
    {
        auto lInstance = static_cast<UIBaseImage *>( aInstance );
        auto lV        = lInstance->TintColor();

        return math::vec4{ lV.x, lV.y, lV.z, lV.w };
    }

    void *UIButton_Create()
    {
        auto lNewButton = new UIButton();

        return static_cast<void *>( lNewButton );
    }

    void *UIButton_CreateWithText( void *aText )
    {
        auto lString    = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );
        auto lNewButton = new UIButton( lString );

        return static_cast<void *>( lNewButton );
    }

    void UIButton_Destroy( void *aInstance ) { delete static_cast<UILabel *>( aInstance ); }

    void UIButton_SetText( void *aInstance, void *aText )
    {
        auto lInstance = static_cast<UILabel *>( aInstance );
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );

        lInstance->SetText( lString );
    }

    void UIButton_OnClick( void *aInstance, void *aDelegate )
    {
        auto lInstance = static_cast<UIButton *>( aInstance );
        auto lDelegate = static_cast<MonoObject *>( aDelegate );

        if( lInstance->mOnClickDelegate != nullptr ) mono_gchandle_free( lInstance->mOnClickDelegateHandle );

        lInstance->mOnClickDelegate       = aDelegate;
        lInstance->mOnClickDelegateHandle = mono_gchandle_new( static_cast<MonoObject *>( aDelegate ), true );

        lInstance->OnClick(
            [lInstance, lDelegate]()
            {
                auto lDelegateClass = mono_object_get_class( lDelegate );
                auto lInvokeMethod  = mono_get_delegate_invoke( lDelegateClass );

                mono_runtime_invoke( lInvokeMethod, lDelegate, nullptr, nullptr );
            } );
    }

    void *UICheckBox_Create()
    {
        auto lNewLabel = new UICheckBox();

        return static_cast<void *>( lNewLabel );
    }

    void UICheckBox_Destroy( void *aInstance ) { delete static_cast<UICheckBox *>( aInstance ); }

    void UICheckBox_OnClick( void *aInstance, void *aDelegate )
    {
        auto lInstance = static_cast<UICheckBox *>( aInstance );
        auto lDelegate = static_cast<MonoObject *>( aDelegate );

        if( lInstance->mOnChangeDelegate != nullptr ) mono_gchandle_free( lInstance->mOnChangeDelegateHandle );

        lInstance->mOnChangeDelegate       = aDelegate;
        lInstance->mOnChangeDelegateHandle = mono_gchandle_new( static_cast<MonoObject *>( aDelegate ), true );

        lInstance->OnClick(
            [lInstance, lDelegate]()
            {
                auto lDelegateClass = mono_object_get_class( lDelegate );
                auto lInvokeMethod  = mono_get_delegate_invoke( lDelegateClass );

                mono_runtime_invoke( lInvokeMethod, lDelegate, nullptr, nullptr );
            } );
    }

    bool UICheckBox_IsChecked( void *aInstance )
    {
        auto lInstance = static_cast<UICheckBox *>( aInstance );

        return lInstance->IsChecked();
    }

    void UICheckBox_SetIsChecked( void *aInstance, bool aValue )
    {
        auto lInstance = static_cast<UICheckBox *>( aInstance );

        lInstance->SetIsChecked( aValue );
    }

    void *UIColorButton_Create()
    {
        auto lNewLabel = new UIColorButton();

        return static_cast<void *>( lNewLabel );
    }

    void UIColorButton_Destroy( void *aInstance ) { delete static_cast<UIColorButton *>( aInstance ); }

    void *UIComboBox_Create()
    {
        auto lNewComboBox = new UIComboBox();

        return static_cast<void *>( lNewComboBox );
    }

    void *UIComboBox_CreateWithItems( void *aItems )
    {
        std::vector<std::string> lItemVector;
        for( auto const &x : DotNetRuntime::AsVector<MonoString *>( static_cast<MonoObject *>( aItems ) ) )
            lItemVector.emplace_back( DotNetRuntime::NewString( x ) );

        auto lNewComboBox = new UIComboBox( lItemVector );

        return static_cast<void *>( lNewComboBox );
    }

    void UIComboBox_Destroy( void *aInstance ) { delete static_cast<UIComboBox *>( aInstance ); }

    int UIComboBox_GetCurrent( void *aInstance )
    {
        auto lInstance = static_cast<UIComboBox *>( aInstance );

        return lInstance->Current();
    }

    void UIComboBox_SetCurrent( void *aInstance, int aValue )
    {
        auto lInstance = static_cast<UIComboBox *>( aInstance );

        lInstance->SetCurrent( aValue );
    }

    void UIComboBox_SetItemList( void *aInstance, void *aItems )
    {
        auto lInstance = static_cast<UIComboBox *>( aInstance );

        std::vector<std::string> lItemVector;
        for( auto const &x : DotNetRuntime::AsVector<MonoString *>( static_cast<MonoObject *>( aItems ) ) )
            lItemVector.emplace_back( DotNetRuntime::NewString( x ) );

        lInstance->SetItemList( lItemVector );
    }

    void UIComboBox_OnChanged( void *aInstance, void *aDelegate )
    {
        auto lInstance = static_cast<UIComboBox *>( aInstance );
        auto lDelegate = static_cast<MonoObject *>( aDelegate );

        if( lInstance->mOnChangeDelegate != nullptr ) mono_gchandle_free( lInstance->mOnChangeDelegateHandle );

        lInstance->mOnChangeDelegate       = aDelegate;
        lInstance->mOnChangeDelegateHandle = mono_gchandle_new( static_cast<MonoObject *>( aDelegate ), true );

        lInstance->OnChange(
            [lInstance, lDelegate]( int aValue )
            {
                auto lDelegateClass = mono_object_get_class( lDelegate );
                auto lInvokeMethod  = mono_get_delegate_invoke( lDelegateClass );

                void *lParams[] = { (void *)&aValue };
                mono_runtime_invoke( lInvokeMethod, lDelegate, lParams, nullptr );
            } );
    }

    void UIComponent_SetIsVisible( void *aSelf, bool aIsVisible )
    {
        auto lSelf = static_cast<UIComponent *>( aSelf );

        lSelf->mIsVisible = aIsVisible;
    }

    void UIComponent_SetIsEnabled( void *aSelf, bool aIsEnabled )
    {
        auto lSelf = static_cast<UIComponent *>( aSelf );

        lSelf->mIsEnabled = aIsEnabled;
    }

    void UIComponent_SetAllowDragDrop( void *aSelf, bool aAllowDragDrop )
    {
        auto lSelf = static_cast<UIComponent *>( aSelf );

        lSelf->mAllowDragDrop = aAllowDragDrop;
    }

    void UIComponent_SetPaddingAll( void *aSelf, float aPaddingAll )
    {
        auto lSelf = static_cast<UIComponent *>( aSelf );

        lSelf->SetPadding( aPaddingAll );
    }

    void UIComponent_SetPaddingPairs( void *aSelf, float aPaddingTopBottom, float aPaddingLeftRight )
    {
        auto lSelf = static_cast<UIComponent *>( aSelf );

        lSelf->SetPadding( aPaddingTopBottom, aPaddingLeftRight );
    }

    void UIComponent_SetPaddingIndividual( void *aSelf, float aPaddingTop, float aPaddingBottom, float aPaddingLeft,
                                           float aPaddingRight )

    {
        auto lSelf = static_cast<UIComponent *>( aSelf );

        lSelf->SetPadding( aPaddingTop, aPaddingBottom, aPaddingLeft, aPaddingRight );
    }

    void UIComponent_SetAlignment( void *aSelf, eHorizontalAlignment aHAlignment, eVerticalAlignment aVAlignment )
    {
        auto lSelf = static_cast<UIComponent *>( aSelf );

        lSelf->SetAlignment( aHAlignment, aVAlignment );
    }

    void UIComponent_SetHorizontalAlignment( void *aSelf, eHorizontalAlignment aAlignment )
    {
        auto lSelf = static_cast<UIComponent *>( aSelf );

        lSelf->SetHorizontalAlignment( aAlignment );
    }

    void UIComponent_SetVerticalAlignment( void *aSelf, eVerticalAlignment aAlignment )
    {
        auto lSelf = static_cast<UIComponent *>( aSelf );

        lSelf->SetVerticalAlignment( aAlignment );
    }

    void UIComponent_SetBackgroundColor( void *aSelf, math::vec4 aColor )
    {
        auto lSelf = static_cast<UIComponent *>( aSelf );

        lSelf->SetBackgroundColor( aColor );
    }

    void UIComponent_SetFont( void *aSelf, FontFamilyFlags aFont )
    {
        auto lSelf = static_cast<UIComponent *>( aSelf );

        lSelf->SetFont( aFont );
    }

    void UIComponent_SetTooltip( void *aSelf, void *aTooltip )
    {
        auto lSelf    = static_cast<UIComponent *>( aSelf );
        auto lTooltip = static_cast<UIComponent *>( aTooltip );

        lSelf->SetTooltip( lTooltip );
    }

    void *UIDropdownButton_Create()
    {
        auto lNewImage = new UIDropdownButton();

        return static_cast<void *>( lNewImage );
    }

    void UIDropdownButton_Destroy( void *aInstance ) { delete static_cast<UIDropdownButton *>( aInstance ); }

    void UIDropdownButton_SetContent( void *aInstance, void *aContent )
    {
        auto lInstance = static_cast<UIDropdownButton *>( aInstance );
        auto lContent  = static_cast<UIComponent *>( aContent );

        return lInstance->SetContent( lContent );
    }

    void UIDropdownButton_SetContentSize( void *aInstance, math::vec2 aContentSizse )
    {
        auto lInstance = static_cast<UIDropdownButton *>( aInstance );

        return lInstance->SetContentSize( aContentSizse );
    }

    void UIDropdownButton_SetImage( void *aInstance, void *aImage )
    {
        auto lInstance = static_cast<UIDropdownButton *>( aInstance );
        auto lImage    = static_cast<UIBaseImage *>( aImage );

        lInstance->SetImage( lImage );
    }

    void UIDropdownButton_SetText( void *aInstance, void *aText )
    {
        auto lInstance = static_cast<UIDropdownButton *>( aInstance );
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );

        lInstance->SetText( lString );
    }

    void UIDropdownButton_SetTextColor( void *aInstance, math::vec4 aColor )
    {
        auto lInstance = static_cast<UIDropdownButton *>( aInstance );

        lInstance->SetTextColor( aColor );
    }

    void *UIImage_Create()
    {
        auto lNewImage = new UIImage();

        return static_cast<void *>( lNewImage );
    }

    void *UIImage_CreateWithPath( void *aText, math::vec2 aSize )
    {
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );
        auto lNewImage = new UIImage( lString, aSize );

        return static_cast<void *>( lNewImage );
    }

    void UIImage_Destroy( void *aInstance ) { delete static_cast<UIImage *>( aInstance ); }

    void *UIImageButton_Create()
    {
        auto lNewImage = new UIImageButton();

        return static_cast<void *>( lNewImage );
    }

    void *UIImageButton_CreateWithPath( void *aText, math::vec2 *aSize )
    {
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );
        auto lNewImage = new UIImageButton( lString, *aSize );

        return static_cast<void *>( lNewImage );
    }

    void UIImageButton_Destroy( void *aInstance ) { delete static_cast<UIImageButton *>( aInstance ); }

    void UIImageButton_OnClick( void *aInstance, void *aDelegate )
    {
        auto lInstance = static_cast<UIImageButton *>( aInstance );
        auto lDelegate = static_cast<MonoObject *>( aDelegate );

        if( lInstance->mOnClickDelegate != nullptr ) mono_gchandle_free( lInstance->mOnClickDelegateHandle );

        lInstance->mOnClickDelegate       = aDelegate;
        lInstance->mOnClickDelegateHandle = mono_gchandle_new( static_cast<MonoObject *>( aDelegate ), true );

        lInstance->OnClick(
            [lInstance, lDelegate]()
            {
                auto lDelegateClass = mono_object_get_class( lDelegate );
                auto lInvokeMethod  = mono_get_delegate_invoke( lDelegateClass );

                mono_runtime_invoke( lInvokeMethod, lDelegate, nullptr, nullptr );
            } );
    }

    void *UIImageToggleButton_Create()
    {
        auto lNewImage = new UIImageToggleButton();

        return static_cast<void *>( lNewImage );
    }

    void UIImageToggleButton_Destroy( void *aInstance )
    {
        delete static_cast<UIImageToggleButton *>( aInstance );
    }

    bool UIImageToggleButton_IsActive( void *aInstance )
    {
        auto lInstance = static_cast<UIImageToggleButton *>( aInstance );

        return lInstance->IsActive();
    }

    void UIImageToggleButton_SetActive( void *aInstance, bool aValue )
    {
        auto lInstance = static_cast<UIImageToggleButton *>( aInstance );

        lInstance->SetActive( aValue );
    }

    void UIImageToggleButton_SetActiveImage( void *aInstance, void *aImage )
    {
        auto lInstance = static_cast<UIImageToggleButton *>( aInstance );
        auto lImage    = static_cast<UIBaseImage *>( aImage );

        lInstance->SetActiveImage( lImage );
    }

    void UIImageToggleButton_SetInactiveImage( void *aInstance, void *aImage )
    {
        auto lInstance = static_cast<UIImageToggleButton *>( aInstance );
        auto lImage    = static_cast<UIBaseImage *>( aImage );

        lInstance->SetInactiveImage( lImage );
    }

    void UIImageToggleButton_OnClicked( void *aInstance, void *aDelegate )
    {
        auto lInstance = static_cast<UIImageToggleButton *>( aInstance );
        auto lDelegate = static_cast<MonoObject *>( aDelegate );

        if( lInstance->mOnClickDelegate != nullptr ) mono_gchandle_free( lInstance->mOnClickDelegateHandle );

        lInstance->mOnClickDelegate       = aDelegate;
        lInstance->mOnClickDelegateHandle = mono_gchandle_new( static_cast<MonoObject *>( aDelegate ), true );

        lInstance->OnClick(
            [lInstance, lDelegate]( bool aValue )
            {
                auto lDelegateClass = mono_object_get_class( lDelegate );
                auto lInvokeMethod  = mono_get_delegate_invoke( lDelegateClass );

                void *lParams[] = { (void *)&aValue };
                auto  lValue    = mono_runtime_invoke( lInvokeMethod, lDelegate, lParams, nullptr );

                return *( (bool *)mono_object_unbox( lValue ) );
            } );
    }

    void UIImageToggleButton_OnChanged( void *aInstance, void *aDelegate )
    {
        auto lInstance = static_cast<UIImageToggleButton *>( aInstance );
        auto lDelegate = static_cast<MonoObject *>( aDelegate );

        if( lInstance->mOnChangeDelegate != nullptr ) mono_gchandle_free( lInstance->mOnChangeDelegateHandle );

        lInstance->mOnChangeDelegate       = aDelegate;
        lInstance->mOnChangeDelegateHandle = mono_gchandle_new( static_cast<MonoObject *>( aDelegate ), true );

        lInstance->OnChanged(
            [lInstance, lDelegate]()
            {
                auto lDelegateClass = mono_object_get_class( lDelegate );
                auto lInvokeMethod  = mono_get_delegate_invoke( lDelegateClass );
                auto lValue         = mono_runtime_invoke( lInvokeMethod, lDelegate, nullptr, nullptr );

                return *( (bool *)mono_object_unbox( lValue ) );
            } );
    }

    void *UILabel_Create()
    {
        auto lNewLabel = new UILabel();

        return static_cast<void *>( lNewLabel );
    }

    void *UILabel_CreateWithText( void *aText )
    {
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );
        auto lNewLabel = new UILabel( lString );

        return static_cast<void *>( lNewLabel );
    }

    void UILabel_Destroy( void *aInstance ) { delete static_cast<UILabel *>( aInstance ); }

    void UILabel_SetText( void *aInstance, void *aText )
    {
        auto lInstance = static_cast<UILabel *>( aInstance );
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );

        lInstance->SetText( lString );
    }

    void UILabel_SetTextColor( void *aInstance, math::vec4 aTextColor )
    {
        auto lInstance = static_cast<UILabel *>( aInstance );

        lInstance->SetTextColor( aTextColor );
    }

    void *UIMenuItem_Create()
    {
        auto lNewLabel = new UIMenuItem();

        return static_cast<void *>( lNewLabel );
    }

    void *UIMenuItem_CreateWithText( void *aText )
    {
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );
        auto lNewLabel = new UIMenuItem( lString );

        return static_cast<void *>( lNewLabel );
    }

    void *UIMenuItem_CreateWithTextAndShortcut( void *aText, void *aShortcut )
    {
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );
        auto lShortcut = DotNetRuntime::NewString( static_cast<MonoString *>( aShortcut ) );
        auto lNewLabel = new UIMenuItem( lString, lShortcut );

        return static_cast<void *>( lNewLabel );
    }

    void UIMenuItem_Destroy( void *aInstance ) { delete static_cast<UIMenuItem *>( aInstance ); }

    void UIMenuItem_SetText( void *aInstance, void *aText )
    {
        auto lInstance = static_cast<UIMenuItem *>( aInstance );
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );

        lInstance->SetText( lString );
    }

    void UIMenuItem_SetShortcut( void *aInstance, void *aShortcut )
    {
        auto lInstance = static_cast<UIMenuItem *>( aInstance );
        auto lShortcut = DotNetRuntime::NewString( static_cast<MonoString *>( aShortcut ) );

        lInstance->SetShortcut( lShortcut );
    }

    void UIMenuItem_SetTextColor( void *aInstance, math::vec4 *aTextColor )
    {
        auto lInstance = static_cast<UIMenuItem *>( aInstance );

        lInstance->SetTextColor( *aTextColor );
    }

    void UIMenuItem_OnTrigger( void *aInstance, void *aDelegate )
    {
        auto lInstance = static_cast<UIMenuItem *>( aInstance );
        auto lDelegate = static_cast<MonoObject *>( aDelegate );

        if( lInstance->mOnTriggerDelegate != nullptr ) mono_gchandle_free( lInstance->mOnTriggerDelegateHandle );

        lInstance->mOnTriggerDelegate       = aDelegate;
        lInstance->mOnTriggerDelegateHandle = mono_gchandle_new( static_cast<MonoObject *>( aDelegate ), true );

        lInstance->OnTrigger(
            [lInstance, lDelegate]()
            {
                auto lDelegateClass = mono_object_get_class( lDelegate );
                auto lInvokeMethod  = mono_get_delegate_invoke( lDelegateClass );

                mono_runtime_invoke( lInvokeMethod, lDelegate, nullptr, nullptr );
            } );
    }

    void *UIMenuSeparator_Create()
    {
        auto lNewSeparator = new UIMenuSeparator();

        return static_cast<void *>( lNewSeparator );
    }

    void UIMenuSeparator_Destroy( void *aInstance ) { delete static_cast<UIMenuSeparator *>( aInstance ); }

    void *UIMenu_Create()
    {
        auto lNewLabel = new UIMenu();

        return static_cast<void *>( lNewLabel );
    }

    void *UIMenu_CreateWithText( void *aText )
    {
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );
        auto lNewLabel = new UIMenu( lString );

        return static_cast<void *>( lNewLabel );
    }

    void UIMenu_Destroy( void *aInstance ) { delete static_cast<UIMenu *>( aInstance ); }

    void *UIMenu_AddAction( void *aInstance, void *aText, void *aShortcut )
    {
        auto lInstance  = static_cast<UIMenu *>( aInstance );
        auto lString    = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );
        auto lShortcut  = DotNetRuntime::NewString( static_cast<MonoString *>( aShortcut ) );
        auto lNewAction = lInstance->AddActionRaw( lString, lShortcut );

        return static_cast<void *>( lNewAction );
    }

    void *UIMenu_AddMenu( void *aInstance, void *aText )
    {
        auto lInstance = static_cast<UIMenu *>( aInstance );
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );
        auto lNewMenu  = lInstance->AddMenuRaw( lString );

        return static_cast<void *>( lNewMenu );
    }

    void *UIMenu_AddSeparator( void *aInstance )
    {
        auto lInstance     = static_cast<UIMenu *>( aInstance );
        auto lNewSeparator = lInstance->AddSeparatorRaw();

        return static_cast<void *>( lNewSeparator );
    }

    void UIMenu_Update( void *aInstance )
    {
        auto lInstance = static_cast<UIMenu *>( aInstance );

        lInstance->Update();
    }

    void *UIPlot_Create()
    {
        auto lNewPlot = new UIPlot();

        return static_cast<void *>( lNewPlot );
    }

    void UIPlot_Destroy( void *aInstance ) { delete static_cast<UIPlot *>( aInstance ); }

    void UIPlot_Clear( void *aInstance )
    {
        auto lSelf = static_cast<UIPlot *>( aInstance );

        lSelf->Clear();
    }

    void UIPlot_ConfigureLegend( void *aInstance, math::vec2 *aLegendPadding, math::vec2 *aLegendInnerPadding,
                                         math::vec2 *aLegendSpacing )
    {
        auto lSelf = static_cast<UIPlot *>( aInstance );

        lSelf->ConfigureLegend( *aLegendPadding, *aLegendInnerPadding, *aLegendSpacing );
    }

    void UIPlot_Add( void *aInstance, void *aPlot )
    {
        auto lSelf = static_cast<UIPlot *>( aInstance );
        auto lPlot = static_cast<sPlotData *>( aPlot );

        lSelf->Add( lPlot );
    }

    void UIPlot_SetAxisLimits( void *aInstance, int aAxis, double aMin, double aMax )
    {
        auto lSelf = static_cast<UIPlot *>( aInstance );

        lSelf->mAxisConfiguration[aAxis].mSetLimitRequest = true;

        lSelf->mAxisConfiguration[aAxis].mMin = static_cast<float>( aMin );
        lSelf->mAxisConfiguration[aAxis].mMax = static_cast<float>( aMax );
    }

    void UIPlot_SetAxisTitle( void *aInstance, int aAxis, void *aTitle )
    {
        auto lSelf = static_cast<UIPlot *>( aInstance );

        lSelf->mAxisConfiguration[aAxis].mTitle = DotNetRuntime::NewString( static_cast<MonoString *>( aTitle ) );
    }

    void *UIPlot_GetAxisTitle( void *aInstance, int aAxis )
    {
        auto lSelf = static_cast<UIPlot *>( aInstance );

        return DotNetRuntime::NewString( lSelf->mAxisConfiguration[aAxis].mTitle );
    }

    void UIPlotData_SetLegend( void *aSelf, void *aText )
    {
        auto lSelf   = static_cast<sPlotData *>( aSelf );
        auto lString = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );

        lSelf->mLegend = lString;
    }

    void UIPlotData_SetThickness( void *aSelf, float aThickness )
    {
        auto lSelf = static_cast<sPlotData *>( aSelf );

        lSelf->mThickness = aThickness;
    }

    void UIPlotData_SetColor( void *aSelf, math::vec4 aColor )
    {
        auto lSelf = static_cast<sPlotData *>( aSelf );

        lSelf->mColor = aColor;
    }

    void UIPlotData_SetXAxis( void *aSelf, int aAxis )
    {
        auto lSelf = static_cast<sPlotData *>( aSelf );

        lSelf->mXAxis = static_cast<UIPlotAxis>( aAxis );
    }

    void UIPlotData_SetYAxis( void *aSelf, int aAxis )
    {
        auto lSelf = static_cast<sPlotData *>( aSelf );

        lSelf->mYAxis = static_cast<UIPlotAxis>( aAxis );
    }

    void *UIFloat64LinePlot_Create()
    {
        auto lSelf = new sFloat64LinePlot();

        return static_cast<sFloat64LinePlot *>( lSelf );
    }

    void UIFloat64LinePlot_Destroy( void *aSelf ) { delete static_cast<sFloat64LinePlot *>( aSelf ); }

    void UIFloat64LinePlot_SetX( void *aSelf, void *aValue )
    {
        auto lSelf = static_cast<sFloat64LinePlot *>( aSelf );

        lSelf->mX = DotNetRuntime::AsVector<double>( static_cast<MonoObject *>( aValue ) );
    }

    void UIFloat64LinePlot_SetY( void *aSelf, void *aValue )
    {
        auto lSelf = static_cast<sFloat64LinePlot *>( aSelf );

        lSelf->mY = DotNetRuntime::AsVector<double>( static_cast<MonoObject *>( aValue ) );
    }

    void *UIFloat64ScatterPlot_Create()
    {
        auto lSelf = new sFloat64ScatterPlot();

        return static_cast<sFloat64ScatterPlot *>( lSelf );
    }

    void UIFloat64ScatterPlot_Destroy( void *aSelf ) { delete static_cast<sFloat64ScatterPlot *>( aSelf ); }

    void UIFloat64ScatterPlot_SetX( void *aSelf, void *aValue )
    {
        auto lSelf = static_cast<sFloat64ScatterPlot *>( aSelf );

        lSelf->mX = DotNetRuntime::AsVector<double>( static_cast<MonoObject *>( aValue ) );
    }

    void UIFloat64ScatterPlot_SetY( void *aSelf, void *aValue )
    {
        auto lSelf = static_cast<sFloat64ScatterPlot *>( aSelf );

        lSelf->mY = DotNetRuntime::AsVector<double>( static_cast<MonoObject *>( aValue ) );
    }

    void *UIVLinePlot_Create()
    {
        auto lSelf = new sVLine();

        return static_cast<sVLine *>( lSelf );
    }

    void UIVLinePlot_Destroy( void *aSelf ) { delete static_cast<sVLine *>( aSelf ); }

    void UIVLinePlot_SetX( void *aSelf, void *aValue )
    {
        auto lSelf = static_cast<sVLine *>( aSelf );

        lSelf->mX = DotNetRuntime::AsVector<double>( static_cast<MonoObject *>( aValue ) );
    }

    void *UIHLinePlot_Create()
    {
        auto lSelf = new sHLine();

        return static_cast<sHLine *>( lSelf );
    }

    void UIHLinePlot_Destroy( void *aSelf ) { delete static_cast<sHLine *>( aSelf ); }

    void UIHLinePlot_SetY( void *aSelf, void *aValue )
    {
        auto lSelf = static_cast<sHLine *>( aSelf );

        lSelf->mY = DotNetRuntime::AsVector<double>( static_cast<MonoObject *>( aValue ) );
    }

    void *UIAxisTag_Create()
    {
        auto lSelf = new sAxisTag();

        return static_cast<sAxisTag *>( lSelf );
    }

    void *UIAxisTag_CreateWithTextAndColor( UIPlotAxis aAxis, double aX, void *aText, math::vec4 aColor )
    {
        auto lString = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );

        auto lSelf = new sAxisTag( aAxis, aX, lString, aColor );

        return static_cast<sAxisTag *>( lSelf );
    }

    void UIAxisTag_Destroy( void *aSelf ) { delete static_cast<sAxisTag *>( aSelf ); }

    void UIAxisTag_SetX( void *aSelf, double aValue )
    {
        auto lSelf = static_cast<sAxisTag *>( aSelf );

        lSelf->mX = aValue;
    }

    void UIAxisTag_SetText( void *aSelf, void *aText )
    {
        auto lSelf   = static_cast<sAxisTag *>( aSelf );
        auto lString = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );

        lSelf->mText = lString;
    }

    void UIAxisTag_SetColor( void *aSelf, math::vec4 aColor )
    {
        auto lSelf = static_cast<sAxisTag *>( aSelf );

        lSelf->mColor = aColor;
    }

    math::vec4 UIAxisTag_GetColor( void *aSelf )
    {
        auto lSelf = static_cast<sAxisTag *>( aSelf );

        return lSelf->mColor;
    }

    void UIAxisTag_SetAxis( void *aSelf, int aAxis )
    {
        auto lSelf = static_cast<sAxisTag *>( aSelf );

        lSelf->mAxis = static_cast<UIPlotAxis>( aAxis );
    }

    int UIAxisTag_GetAxis( void *aSelf )
    {
        auto lSelf = static_cast<sAxisTag *>( aSelf );

        return static_cast<int>( lSelf->mXAxis );
    }

    void *UIVRangePlot_Create()
    {
        auto lSelf = new sVRange();

        return static_cast<sVRange *>( lSelf );
    }

    void UIVRangePlot_Destroy( void *aSelf ) { delete static_cast<sVRange *>( aSelf ); }

    void UIVRangePlot_SetMin( void *aSelf, double aValue )
    {
        auto lSelf = static_cast<sVRange *>( aSelf );

        lSelf->mX0 = aValue;
    }

    double UIVRangePlot_GetMin( void *aSelf )
    {
        auto lSelf = static_cast<sVRange *>( aSelf );

        return (double)lSelf->mX0;
    }

    void UIVRangePlot_SetMax( void *aSelf, double aValue )
    {
        auto lSelf = static_cast<sVRange *>( aSelf );

        lSelf->mX1 = aValue;
    }

    double UIVRangePlot_GetMax( void *aSelf )
    {
        auto lSelf = static_cast<sVRange *>( aSelf );

        return (double)lSelf->mX1;
    }

    void *UIHRangePlot_Create()
    {
        auto lSelf = new sHRange();

        return static_cast<sHRange *>( lSelf );
    }

    void UIHRangePlot_Destroy( void *aSelf ) { delete static_cast<sHRange *>( aSelf ); }

    void UIHRangePlot_SetMin( void *aSelf, double aValue )
    {
        auto lSelf = static_cast<sHRange *>( aSelf );

        lSelf->mY0 = aValue;
    }

    double UIHRangePlot_GetMin( void *aSelf )
    {
        auto lSelf = static_cast<sHRange *>( aSelf );

        return (double)lSelf->mY0;
    }

    void UIHRangePlot_SetMax( void *aSelf, double aValue )
    {
        auto lSelf = static_cast<sHRange *>( aSelf );

        lSelf->mY1 = aValue;
    }

    double UIHRangePlot_GetMax( void *aSelf )
    {
        auto lSelf = static_cast<sHRange *>( aSelf );

        return (double)lSelf->mY1;
    }

    void *UIProgressBar_Create()
    {
        auto lNewLabel = new UIProgressBar();

        return static_cast<void *>( lNewLabel );
    }

    void UIProgressBar_Destroy( void *aInstance ) { delete static_cast<UIProgressBar *>( aInstance ); }

    void UIProgressBar_SetProgressValue( void *aInstance, float aValue )
    {
        auto lInstance = static_cast<UIProgressBar *>( aInstance );

        lInstance->SetProgressValue( aValue );
    }

    void UIProgressBar_SetProgressColor( void *aInstance, math::vec4 aTextColor )
    {
        auto lInstance = static_cast<UIProgressBar *>( aInstance );

        lInstance->SetProgressColor( aTextColor );
    }

    void UIProgressBar_SetText( void *aInstance, void *aText )
    {
        auto lInstance = static_cast<UIProgressBar *>( aInstance );
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );

        lInstance->SetText( lString );
    }

    void UIProgressBar_SetTextColor( void *aInstance, math::vec4 aTextColor )
    {
        auto lInstance = static_cast<UIProgressBar *>( aInstance );

        lInstance->SetTextColor( aTextColor );
    }

    void UIProgressBar_SetThickness( void *aInstance, float aValue )
    {
        auto lInstance = static_cast<UIProgressBar *>( aInstance );

        lInstance->SetThickness( aValue );
    }

    void *UIPropertyValue_Create()
    {
        auto lNewLabel = new UIPropertyValue();

        return static_cast<void *>( lNewLabel );
    }

    void *UIPropertyValue_CreateWithText( void *aText )
    {
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );
        auto lNewLabel = new UIPropertyValue( lString );

        return static_cast<void *>( lNewLabel );
    }

    void *UIPropertyValue_CreateWithTextAndOrientation( void *aText, eBoxLayoutOrientation aOrientation )
    {
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );
        auto lNewLabel = new UIPropertyValue( lString, aOrientation );

        return static_cast<void *>( lNewLabel );
    }

    void UIPropertyValue_Destroy( void *aInstance ) { delete static_cast<UIPropertyValue *>( aInstance ); }

    void UIPropertyValue_SetValue( void *aInstance, void *aText )
    {
        auto lInstance = static_cast<UIPropertyValue *>( aInstance );
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );

        lInstance->SetValue( lString );
    }

    void UIPropertyValue_SetValueFont( void *aInstance, FontFamilyFlags aFont )
    {
        auto lInstance = static_cast<UIPropertyValue *>( aInstance );

        lInstance->SetValueFont( aFont );
    }

    void UIPropertyValue_SetNameFont( void *aInstance, FontFamilyFlags aFont )
    {
        auto lInstance = static_cast<UIPropertyValue *>( aInstance );

        lInstance->SetNameFont( aFont );
    }

    void *UISlider_Create()
    {
        auto lNewLabel = new UISlider();

        return static_cast<void *>( lNewLabel );
    }

    void UISlider_Destroy( void *aInstance ) { delete static_cast<UISlider *>( aInstance ); }

    void UITableColumn_SetTooltip( void *aSelf, void *aTooptip )
    {
        auto lSelf = static_cast<sTableColumn *>( aSelf );

        lSelf->mToolTip.clear();
        for( auto const &x : DotNetRuntime::AsVector<UIComponent *>( static_cast<MonoObject *>( aTooptip ) ) )
            lSelf->mToolTip.push_back( x );
    }

    void UITableColumn_SetForegroundColor( void *aSelf, void *aForegroundColor )
    {
        auto lSelf = static_cast<sTableColumn *>( aSelf );

        lSelf->mForegroundColor.clear();
        for( auto const &x : DotNetRuntime::AsVector<ImVec4>( static_cast<MonoObject *>( aForegroundColor ) ) )
            lSelf->mForegroundColor.push_back( ImColor( x ) );
    }

    void UITableColumn_SetBackgroundColor( void *aSelf, void *aBackroundColor )
    {
        auto lSelf = static_cast<sTableColumn *>( aSelf );

        lSelf->mBackgroundColor.clear();
        for( auto const &x : DotNetRuntime::AsVector<ImVec4>( static_cast<MonoObject *>( aBackroundColor ) ) )
            lSelf->mBackgroundColor.push_back( ImColor( x ) );
    }

    void *UITable_Create()
    {
        auto lNewTable = new UITable();

        return static_cast<void *>( lNewTable );
    }

    void UITable_Destroy( void *aSelf ) { delete static_cast<UITable *>( aSelf ); }

    void UITable_OnRowClicked( void *aInstance, void *aDelegate )
    {
        auto lInstance = static_cast<UITable *>( aInstance );
        auto lDelegate = static_cast<MonoObject *>( aDelegate );

        if( lInstance->mOnRowClickDelegate != nullptr ) mono_gchandle_free( lInstance->mOnRowClickDelegateHandle );

        lInstance->mOnRowClickDelegate       = aDelegate;
        lInstance->mOnRowClickDelegateHandle = mono_gchandle_new( static_cast<MonoObject *>( aDelegate ), true );

        lInstance->OnRowClicked(
            [lInstance, lDelegate]( int aValue )
            {
                auto lDelegateClass = mono_object_get_class( lDelegate );
                auto lInvokeMethod  = mono_get_delegate_invoke( lDelegateClass );

                void *lParams[] = { (void *)&aValue };
                auto  lValue    = mono_runtime_invoke( lInvokeMethod, lDelegate, lParams, nullptr );
            } );
    }

    void UITable_AddColumn( void *aSelf, void *aColumn )
    {
        auto lInstance = static_cast<UITable *>( aSelf );
        auto lColumn   = static_cast<sTableColumn *>( aColumn );

        lInstance->AddColumn( lColumn );
    }

    void UITable_SetRowHeight( void *aSelf, float aRowHeight )
    {
        auto lInstance = static_cast<UITable *>( aSelf );

        lInstance->SetRowHeight( aRowHeight );
    }

    void UITable_ClearRowBackgroundColor( void *aSelf )
    {
        auto lSelf = static_cast<UITable *>( aSelf );

        lSelf->mRowBackgroundColor.clear();
    }

    void UITable_SetRowBackgroundColor( void *aSelf, void *aValue )
    {
        auto lSelf = static_cast<UITable *>( aSelf );

        lSelf->mRowBackgroundColor.clear();
        for( auto &x : DotNetRuntime::AsVector<ImVec4>( static_cast<MonoObject *>( aValue ) ) )
            lSelf->mRowBackgroundColor.push_back( ImColor( x ) );
    }

    void UITable_SetDisplayedRowIndices( void *aSelf, void *aValue )
    {
        auto lSelf = static_cast<UITable *>( aSelf );
        if( aValue == nullptr )
            lSelf->mDisplayedRowIndices.reset();
        else
            lSelf->mDisplayedRowIndices = DotNetRuntime::AsVector<int>( static_cast<MonoObject *>( aValue ) );
    }

    void *UIFloat64Column_Create()
    {
        auto lNewColumn = new sFloat64Column();

        return static_cast<void *>( lNewColumn );
    }

    void *UIFloat64Column_CreateFull( void *aHeader, float aInitialSize, void *aFormat, void *aNaNFormat )
    {
        auto lHeader    = DotNetRuntime::NewString( static_cast<MonoString *>( aHeader ) );
        auto lFormat    = DotNetRuntime::NewString( static_cast<MonoString *>( aFormat ) );
        auto lNaNFormat = DotNetRuntime::NewString( static_cast<MonoString *>( aNaNFormat ) );
        auto lNewColumn = new sFloat64Column( lHeader, aInitialSize, lFormat, lNaNFormat );

        return static_cast<void *>( lNewColumn );
    }

    void UIFloat64Column_Destroy( void *aSelf ) { delete static_cast<sFloat64Column *>( aSelf ); }

    void UIFloat64Column_Clear( void *aSelf )
    {
        auto lSelf = static_cast<sFloat64Column *>( aSelf );

        lSelf->Clear();
    }

    void UIFloat64Column_SetData( void *aSelf, void *aValue )
    {
        auto lSelf = static_cast<sFloat64Column *>( aSelf );

        lSelf->mData = DotNetRuntime::AsVector<double>( static_cast<MonoObject *>( aValue ) );
    }

    void *UIUint32Column_Create()
    {
        auto lNewColumn = new sUint32Column();

        return static_cast<void *>( lNewColumn );
    }

    void *UIUint32Column_CreateFull( void *aHeader, float aInitialSize )
    {
        auto lHeader    = DotNetRuntime::NewString( static_cast<MonoString *>( aHeader ) );
        auto lNewColumn = new sUint32Column( lHeader, aInitialSize );

        return static_cast<void *>( lNewColumn );
    }

    void UIUint32Column_Destroy( void *aSelf ) { delete static_cast<sUint32Column *>( aSelf ); }

    void UIUint32Column_Clear( void *aSelf )
    {
        auto lSelf = static_cast<sUint32Column *>( aSelf );

        lSelf->Clear();
    }

    void UIUint32Column_SetData( void *aSelf, void *aValue )
    {
        auto lSelf = static_cast<sUint32Column *>( aSelf );

        lSelf->mData = DotNetRuntime::AsVector<uint32_t>( static_cast<MonoObject *>( aValue ) );
    }

    void *UIStringColumn_Create()
    {
        auto lNewColumn = new sStringColumn();

        return static_cast<void *>( lNewColumn );
    }

    void *UIStringColumn_CreateFull( void *aHeader, float aInitialSize )
    {
        auto lHeader    = DotNetRuntime::NewString( static_cast<MonoString *>( aHeader ) );
        auto lNewColumn = new sStringColumn( lHeader, aInitialSize );

        return static_cast<void *>( lNewColumn );
    }

    void UIStringColumn_Destroy( void *aSelf ) { delete static_cast<sStringColumn *>( aSelf ); }

    void UIStringColumn_Clear( void *aSelf )
    {
        auto lSelf = static_cast<sStringColumn *>( aSelf );

        lSelf->Clear();
    }

    void UIStringColumn_SetData( void *aSelf, void *aValue )
    {
        auto lSelf = static_cast<sStringColumn *>( aSelf );

        lSelf->mData.clear();
        for( auto const &x : DotNetRuntime::AsVector<MonoString *>( static_cast<MonoObject *>( aValue ) ) )
            lSelf->mData.push_back( DotNetRuntime::NewString( x ) );
    }

    void *UITextInput_Create()
    {
        auto lNewTextInput = new UITextInput();

        return static_cast<void *>( lNewTextInput );
    }

    void *UITextInput_CreateWithText( void *aText )
    {
        auto lString       = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );
        auto lNewTextInput = new UITextInput( lString );

        return static_cast<void *>( lNewTextInput );
    }

    void UITextInput_Destroy( void *aInstance ) { delete static_cast<UITextInput *>( aInstance ); }

    void UITextInput_SetHintText( void *aInstance, void *aText )
    {
        auto lInstance = static_cast<UITextInput *>( aInstance );
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );

        lInstance->SetHintText( lString );
    }

    void *UITextInput_GetText( void *aInstance )
    {
        auto lInstance = static_cast<UITextInput *>( aInstance );

        return DotNetRuntime::NewString( lInstance->GetText() );
    }

    void UITextInput_SetTextColor( void *aInstance, math::vec4 *aTextColor )
    {
        auto lInstance = static_cast<UITextInput *>( aInstance );

        lInstance->SetTextColor( *aTextColor );
    }

    void UITextInput_SetBufferSize( void *aInstance, uint32_t aBufferSize )
    {
        auto lInstance = static_cast<UITextInput *>( aInstance );

        lInstance->SetBuffersize( aBufferSize );
    }

    void UITextInput_OnTextChanged( void *aInstance, void *aDelegate )
    {
        auto lInstance = static_cast<UITextInput *>( aInstance );
        auto lDelegate = static_cast<MonoObject *>( aDelegate );

        if( lInstance->mOnTextChangedDelegate != nullptr ) mono_gchandle_free( lInstance->mOnTextChangedDelegateHandle );

        lInstance->mOnTextChangedDelegate       = aDelegate;
        lInstance->mOnTextChangedDelegateHandle = mono_gchandle_new( static_cast<MonoObject *>( aDelegate ), true );

        lInstance->OnTextChanged(
            [lInstance, lDelegate]( std::string aText )
            {
                auto lDelegateClass = mono_object_get_class( lDelegate );
                auto lInvokeMethod  = mono_get_delegate_invoke( lDelegateClass );

                auto  lString   = DotNetRuntime::NewString( aText );
                void *lParams[] = { (void *)lString };
                auto  lValue    = mono_runtime_invoke( lInvokeMethod, lDelegate, lParams, nullptr );
                mono_free( lString );
            } );
    }

    void *UITextOverlay_Create()
    {
        auto lNewLabel = new UITextOverlay();

        return static_cast<void *>( lNewLabel );
    }

    void UITextOverlay_Destroy( void *aInstance ) { delete static_cast<UITextOverlay *>( aInstance ); }

    void UITextOverlay_AddText( void *aInstance, void *aText )
    {
        auto lInstance = static_cast<UITextOverlay *>( aInstance );
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );

        lInstance->AddText( lString );
    }

    void UITextOverlay_Clear( void *aInstance )
    {
        auto lInstance = static_cast<UITextOverlay *>( aInstance );

        lInstance->Clear();
    }

    void *UITextToggleButton_Create()
    {
        auto lNewButton = new UITextToggleButton();

        return static_cast<void *>( lNewButton );
    }

    void *UITextToggleButton_CreateWithText( void *aText )
    {
        auto lString    = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );
        auto lNewButton = new UITextToggleButton( lString );

        return static_cast<void *>( lNewButton );
    }

    void UITextToggleButton_Destroy( void *aInstance ) { delete static_cast<UITextToggleButton *>( aInstance ); }

    bool UITextToggleButton_IsActive( void *aInstance )
    {
        auto lInstance = static_cast<UITextToggleButton *>( aInstance );

        return lInstance->IsActive();
    }

    void UITextToggleButton_SetActive( void *aInstance, bool aValue )
    {
        auto lInstance = static_cast<UITextToggleButton *>( aInstance );

        lInstance->SetActive( aValue );
    }

    void UITextToggleButton_SetActiveColor( void *aInstance, math::vec4 *aColor )
    {
        auto lInstance = static_cast<UITextToggleButton *>( aInstance );

        lInstance->SetActiveColor( *aColor );
    }

    void UITextToggleButton_SetInactiveColor( void *aInstance, math::vec4 *aColor )
    {
        auto lInstance = static_cast<UITextToggleButton *>( aInstance );

        lInstance->SetInactiveColor( *aColor );
    }

    void UITextToggleButton_OnClicked( void *aInstance, void *aDelegate )
    {
        auto lInstance = static_cast<UITextToggleButton *>( aInstance );
        auto lDelegate = static_cast<MonoObject *>( aDelegate );

        if( lInstance->mOnClickDelegate != nullptr ) mono_gchandle_free( lInstance->mOnClickDelegateHandle );

        lInstance->mOnClickDelegate       = aDelegate;
        lInstance->mOnClickDelegateHandle = mono_gchandle_new( static_cast<MonoObject *>( aDelegate ), true );

        lInstance->OnClick(
            [lInstance, lDelegate]( bool aValue )
            {
                auto lDelegateClass = mono_object_get_class( lDelegate );
                auto lInvokeMethod  = mono_get_delegate_invoke( lDelegateClass );

                void *lParams[] = { (void *)&aValue };
                auto  lValue    = mono_runtime_invoke( lInvokeMethod, lDelegate, lParams, nullptr );

                return *( (bool *)mono_object_unbox( lValue ) );
            } );
    }

    void UITextToggleButton_OnChanged( void *aInstance, void *aDelegate )
    {
        auto lInstance = static_cast<UITextToggleButton *>( aInstance );
        auto lDelegate = static_cast<MonoObject *>( aDelegate );

        if( lInstance->mOnChangeDelegate != nullptr ) mono_gchandle_free( lInstance->mOnChangeDelegateHandle );

        lInstance->mOnChangeDelegate       = aDelegate;
        lInstance->mOnChangeDelegateHandle = mono_gchandle_new( static_cast<MonoObject *>( aDelegate ), true );

        lInstance->OnChanged(
            [lInstance, lDelegate]()
            {
                auto lDelegateClass = mono_object_get_class( lDelegate );
                auto lInvokeMethod  = mono_get_delegate_invoke( lDelegateClass );
                auto lValue         = mono_runtime_invoke( lInvokeMethod, lDelegate, nullptr, nullptr );

                return *( (bool *)mono_object_unbox( lValue ) );
            } );
    }

    void *UITreeViewNode_Create()
    {
        auto lNewLabel = new UITreeViewNode();

        return static_cast<void *>( lNewLabel );
    }

    void UITreeViewNode_Destroy( void *aInstance ) { delete static_cast<UITreeViewNode *>( aInstance ); }

    void UITreeViewNode_SetText( void *aInstance, void *aText )
    {
        auto lInstance = static_cast<UITreeViewNode *>( aInstance );
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );

        lInstance->SetText( lString );
    }

    void UITreeViewNode_SetTextColor( void *aInstance, math::vec4 aTextColor )
    {
        auto lInstance = static_cast<UITreeViewNode *>( aInstance );

        lInstance->SetTextColor( aTextColor );
    }

    void UITreeViewNode_SetIcon( void *aInstance, void *aIcon )
    {
        auto lInstance = static_cast<UITreeViewNode *>( aInstance );
        auto lImage    = static_cast<UIImage *>( aIcon );

        lInstance->SetIcon( lImage );
    }

    void UITreeViewNode_SetIndicator( void *aInstance, void *aIndicator )
    {
        auto lInstance = static_cast<UITreeViewNode *>( aInstance );
        auto lImage    = static_cast<UIComponent *>( aIndicator );

        lInstance->SetIndicator( lImage );
    }

    void *UITreeViewNode_Add( void *aInstance )
    {
        auto lInstance = static_cast<UITreeViewNode *>( aInstance );

        return static_cast<void *>( lInstance->Add() );
    }

    void *UITreeView_Create()
    {
        auto lNewLabel = new UITreeView();

        return static_cast<void *>( lNewLabel );
    }

    void UITreeView_Destroy( void *aInstance ) { delete static_cast<UITreeView *>( aInstance ); }

    void UITreeView_SetIndent( void *aInstance, float aIndent )
    {
        auto lInstance = static_cast<UITreeView *>( aInstance );

        lInstance->SetIndent( aIndent );
    }

    void UITreeView_SetIconSpacing( void *aInstance, float aSpacing )
    {
        auto lInstance = static_cast<UITreeView *>( aInstance );

        lInstance->SetIconSpacing( aSpacing );
    }

    void *UITreeView_Add( void *aInstance )
    {
        auto lInstance = static_cast<UITreeView *>( aInstance );

        return static_cast<void *>( lInstance->Add() );
    }

    void *UIVec2Input_Create()
    {
        auto lNewVecInput = new UIVec2Input();

        return static_cast<void *>( lNewVecInput );
    }

    void UIVec2Input_Destroy( void *aInstance ) { delete static_cast<UIVec2Input *>( aInstance ); }

    void UIVec2Input_OnChanged( void *aInstance, void *aDelegate )
    {
        auto lInstance = static_cast<UIVectorInputBase *>( aInstance );
        auto lDelegate = static_cast<MonoObject *>( aDelegate );

        if( lInstance->mOnChangeDelegate != nullptr ) mono_gchandle_free( lInstance->mOnChangeDelegateHandle );

        lInstance->mOnChangeDelegate       = aDelegate;
        lInstance->mOnChangeDelegateHandle = mono_gchandle_new( static_cast<MonoObject *>( aDelegate ), true );

        lInstance->OnChanged(
            [lInstance, lDelegate]( math::vec4 aVector )
            {
                auto lDelegateClass = mono_object_get_class( lDelegate );
                auto lInvokeMethod  = mono_get_delegate_invoke( lDelegateClass );

                math::vec2 lProjection = math::vec2{ aVector.x, aVector.y };
                void      *lParams[]   = { (void *)&lProjection };
                auto       lValue      = mono_runtime_invoke( lInvokeMethod, lDelegate, lParams, nullptr );
            } );
    }

    void UIVec2Input_SetValue( void *aInstance, math::vec2 aValue )
    {
        static_cast<UIVec2Input *>( aInstance )->SetValue( aValue );
    }

    math::vec2 UIVec2Input_GetValue( void *aInstance ) { return static_cast<UIVec2Input *>( aInstance )->Value(); }

    void UIVec2Input_SetResetValues( void *aInstance, math::vec2 aValue )
    {
        static_cast<UIVec2Input *>( aInstance )->SetResetValues( aValue );
    }

    void UIVec2Input_SetFormat( void *aInstance, void *aText )
    {
        auto lInstance = static_cast<UIVectorInputBase *>( aInstance );
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );

        lInstance->SetFormat( lString );
    }

    void *UIVec3Input_Create()
    {
        auto lNewVecInput = new UIVec3Input();

        return static_cast<void *>( lNewVecInput );
    }

    void UIVec3Input_Destroy( void *aInstance ) { delete static_cast<UIVec3Input *>( aInstance ); }

    void UIVec3Input_OnChanged( void *aInstance, void *aDelegate )
    {
        auto lInstance = static_cast<UIVectorInputBase *>( aInstance );
        auto lDelegate = static_cast<MonoObject *>( aDelegate );

        if( lInstance->mOnChangeDelegate != nullptr ) mono_gchandle_free( lInstance->mOnChangeDelegateHandle );

        lInstance->mOnChangeDelegate       = aDelegate;
        lInstance->mOnChangeDelegateHandle = mono_gchandle_new( static_cast<MonoObject *>( aDelegate ), true );

        lInstance->OnChanged(
            [lInstance, lDelegate]( math::vec4 aVector )
            {
                auto lDelegateClass = mono_object_get_class( lDelegate );
                auto lInvokeMethod  = mono_get_delegate_invoke( lDelegateClass );

                math::vec3 lProjection = math::vec3{ aVector.x, aVector.y, aVector.z };
                void      *lParams[]   = { (void *)&lProjection };
                auto       lValue      = mono_runtime_invoke( lInvokeMethod, lDelegate, lParams, nullptr );
            } );
    }

    void UIVec3Input_SetValue( void *aInstance, math::vec3 aValue )
    {
        static_cast<UIVec3Input *>( aInstance )->SetValue( aValue );
    }

    math::vec3 UIVec3Input_GetValue( void *aInstance ) { return static_cast<UIVec3Input *>( aInstance )->Value(); }

    void UIVec3Input_SetResetValues( void *aInstance, math::vec3 aValue )
    {
        static_cast<UIVec3Input *>( aInstance )->SetResetValues( aValue );
    }

    void UIVec3Input_SetFormat( void *aInstance, void *aText )
    {
        auto lInstance = static_cast<UIVectorInputBase *>( aInstance );
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );

        lInstance->SetFormat( lString );
    }

    void *UIVec4Input_Create()
    {
        auto lNewVecInput = new UIVec4Input();

        return static_cast<void *>( lNewVecInput );
    }

    void UIVec4Input_Destroy( void *aInstance ) { delete static_cast<UIVec4Input *>( aInstance ); }

    void UIVec4Input_OnChanged( void *aInstance, void *aDelegate )
    {
        auto lInstance = static_cast<UIVectorInputBase *>( aInstance );
        auto lDelegate = static_cast<MonoObject *>( aDelegate );

        if( lInstance->mOnChangeDelegate != nullptr ) mono_gchandle_free( lInstance->mOnChangeDelegateHandle );

        lInstance->mOnChangeDelegate       = aDelegate;
        lInstance->mOnChangeDelegateHandle = mono_gchandle_new( static_cast<MonoObject *>( aDelegate ), true );

        lInstance->OnChanged(
            [lInstance, lDelegate]( math::vec4 aVector )
            {
                auto lDelegateClass = mono_object_get_class( lDelegate );
                auto lInvokeMethod  = mono_get_delegate_invoke( lDelegateClass );

                void *lParams[] = { (void *)&aVector };
                auto  lValue    = mono_runtime_invoke( lInvokeMethod, lDelegate, lParams, nullptr );
            } );
    }

    void UIVec4Input_SetValue( void *aInstance, math::vec4 aValue )
    {
        static_cast<UIVec4Input *>( aInstance )->SetValue( aValue );
    }

    math::vec4 UIVec4Input_GetValue( void *aInstance ) { return static_cast<UIVec4Input *>( aInstance )->Value(); }

    void UIVec4Input_SetResetValues( void *aInstance, math::vec4 aValue )
    {
        static_cast<UIVec4Input *>( aInstance )->SetResetValues( aValue );
    }

    void UIVec4Input_SetFormat( void *aInstance, void *aText )
    {
        auto lInstance = static_cast<UIVectorInputBase *>( aInstance );
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );

        lInstance->SetFormat( lString );
    }

    void *UIWorkspaceDocument_Create()
    {
        auto lNewDocument = new UIWorkspaceDocument();

        return static_cast<void *>( lNewDocument );
    }

    void UIWorkspaceDocument_Destroy( void *aInstance )
    {
        delete static_cast<UIWorkspaceDocument *>( aInstance );
    }

    void UIWorkspaceDocument_SetContent( void *aInstance, void *aContent )
    {
        auto lInstance = static_cast<UIWorkspaceDocument *>( aInstance );
        auto lContent  = static_cast<UIComponent *>( aContent );

        lInstance->SetContent( lContent );
    }

    void UIWorkspaceDocument_Update( void *aInstance )
    {
        auto lInstance = static_cast<UIWorkspaceDocument *>( aInstance );

        lInstance->Update();
    }

    void UIWorkspaceDocument_SetName( void *aInstance, void *aName )
    {
        auto lInstance = static_cast<UIWorkspaceDocument *>( aInstance );
        auto lName     = DotNetRuntime::NewString( static_cast<MonoString *>( aName ) );

        lInstance->mName = lName;
    }

    bool UIWorkspaceDocument_IsDirty( void *aInstance )
    {
        auto lInstance = static_cast<UIWorkspaceDocument *>( aInstance );

        return lInstance->mDirty;
    }

    void UIWorkspaceDocument_MarkAsDirty( void *aInstance, bool aDirty )
    {
        auto lInstance = static_cast<UIWorkspaceDocument *>( aInstance );

        lInstance->mDirty = aDirty;
    }

    void UIWorkspaceDocument_Open( void *aInstance )
    {
        auto lInstance = static_cast<UIWorkspaceDocument *>( aInstance );

        lInstance->DoOpen();
    }

    void UIWorkspaceDocument_RequestClose( void *aInstance )
    {
        auto lInstance = static_cast<UIWorkspaceDocument *>( aInstance );

        lInstance->DoQueueClose();
    }

    void UIWorkspaceDocument_ForceClose( void *aInstance )
    {
        auto lInstance = static_cast<UIWorkspaceDocument *>( aInstance );

        lInstance->DoForceClose();
    }

    void UIWorkspaceDocument_RegisterSaveDelegate( void *aInstance, void *aDelegate )
    {
        auto lInstance = static_cast<UIWorkspaceDocument *>( aInstance );
        auto lDelegate = static_cast<MonoObject *>( aDelegate );

        if( lInstance->mSaveDelegate != nullptr ) mono_gchandle_free( lInstance->mSaveDelegateHandle );

        lInstance->mSaveDelegate       = aDelegate;
        lInstance->mSaveDelegateHandle = mono_gchandle_new( static_cast<MonoObject *>( aDelegate ), true );

        lInstance->mDoSave = [lInstance, lDelegate]()
        {
            auto lDelegateClass = mono_object_get_class( lDelegate );
            auto lInvokeMethod  = mono_get_delegate_invoke( lDelegateClass );

            auto lValue = mono_runtime_invoke( lInvokeMethod, lDelegate, nullptr, nullptr );

            return *( (bool *)mono_object_unbox( lValue ) );
        };
    }

    void *UIWorkspace_Create()
    {
        auto lNewWorkspace = new UIWorkspace();

        return static_cast<void *>( lNewWorkspace );
    }

    void UIWorkspace_Destroy( void *aSelf ) { delete static_cast<UIWorkspace *>( aSelf ); }

    void UIWorkspace_Add( void *aSelf, void *aDocument )
    {
        auto lSelf     = static_cast<UIWorkspace *>( aSelf );
        auto lDocument = static_cast<UIWorkspaceDocument *>( aDocument );

        lSelf->Add( lDocument );
    }

    void UIWorkspace_RegisterCloseDocumentDelegate( void *aInstance, void *aDelegate )
    {
        auto lInstance = static_cast<UIWorkspace *>( aInstance );
        auto lDelegate = static_cast<MonoObject *>( aDelegate );

        if( lInstance->mCloseDocumentDelegate != nullptr ) mono_gchandle_free( lInstance->mCloseDocumentDelegateHandle );

        lInstance->mCloseDocumentDelegate       = aDelegate;
        lInstance->mCloseDocumentDelegateHandle = mono_gchandle_new( static_cast<MonoObject *>( aDelegate ), true );

        lInstance->mOnCloseDocuments = [lInstance, lDelegate]( std::vector<UIWorkspaceDocument *> aDocuments )
        {
            auto lDelegateClass = mono_object_get_class( lDelegate );
            auto lInvokeMethod  = mono_get_delegate_invoke( lDelegateClass );

            MonoArray *lNewArray = mono_array_new( mono_domain_get(), mono_get_uint64_class(), aDocuments.size() );
            for( uint32_t i = 0; i < aDocuments.size(); i++ ) mono_array_set( lNewArray, uint64_t, i, (uint64_t)aDocuments[i] );
            void *lParams[] = { (void *)lNewArray };
            mono_runtime_invoke( lInvokeMethod, lDelegate, lParams, nullptr );
        };
    }

    void *UIBoxLayout_CreateWithOrientation( eBoxLayoutOrientation aOrientation )
    {
        auto lNewLayout = new UIBoxLayout( aOrientation );

        return static_cast<void *>( lNewLayout );
    }

    void UIBoxLayout_Destroy( void *aInstance ) { delete static_cast<UIBoxLayout *>( aInstance ); }

    void UIBoxLayout_AddAlignedNonFixed( void *aInstance, void *aChild, bool aExpand, bool aFill,
                                                      eHorizontalAlignment aHAlignment, eVerticalAlignment aVAlignment )
    {
        auto lInstance = static_cast<UIBoxLayout *>( aInstance );
        auto lChild    = static_cast<UIComponent *>( aChild );

        lInstance->Add( lChild, aExpand, aFill, aHAlignment, aVAlignment );
    }

    void UIBoxLayout_AddNonAlignedNonFixed( void *aInstance, void *aChild, bool aExpand, bool aFill )
    {
        auto lInstance = static_cast<UIBoxLayout *>( aInstance );
        auto lChild    = static_cast<UIComponent *>( aChild );

        lInstance->Add( lChild, aExpand, aFill );
    }

    void UIBoxLayout_AddAlignedFixed( void *aInstance, void *aChild, float aFixedSize, bool aExpand, bool aFill,
                                                   eHorizontalAlignment aHAlignment, eVerticalAlignment aVAlignment )
    {
        auto lInstance = static_cast<UIBoxLayout *>( aInstance );
        auto lChild    = static_cast<UIComponent *>( aChild );

        lInstance->Add( lChild, aFixedSize, aExpand, aFill, aHAlignment, aVAlignment );
    }

    void UIBoxLayout_AddNonAlignedFixed( void *aInstance, void *aChild, float aFixedSize, bool aExpand, bool aFill )
    {
        auto lInstance = static_cast<UIBoxLayout *>( aInstance );
        auto lChild    = static_cast<UIComponent *>( aChild );

        lInstance->Add( lChild, aFixedSize, aExpand, aFill );
    }

    void UIBoxLayout_AddSeparator( void *aInstance )
    {
        auto lInstance = static_cast<UIBoxLayout *>( aInstance );

        lInstance->AddSeparator();
    }

    void UIBoxLayout_SetItemSpacing( void *aInstance, float aItemSpacing )
    {
        auto lInstance = static_cast<UIBoxLayout *>( aInstance );

        lInstance->SetItemSpacing( aItemSpacing );
    }

    void UIBoxLayout_Clear( void *aInstance )
    {
        auto lInstance = static_cast<UIBoxLayout *>( aInstance );

        lInstance->Clear();
    }

    void *UIContainer_Create()
    {
        auto lNewLayout = new UIContainer();

        return static_cast<void *>( lNewLayout );
    }

    void UIContainer_Destroy( void *aInstance ) { delete static_cast<UIContainer *>( aInstance ); }

    void UIContainer_SetContent( void *aInstance, void *aChild )
    {
        auto lInstance = static_cast<UIContainer *>( aInstance );
        auto lChild    = static_cast<UIComponent *>( aChild );

        lInstance->SetContent( lChild );
    }

    void *UISplitter_Create()
    {
        auto lNewLayout = new UISplitter();

        return static_cast<void *>( lNewLayout );
    }

    void *UISplitter_CreateWithOrientation( eBoxLayoutOrientation aOrientation )
    {
        auto lNewLayout = new UISplitter( aOrientation );

        return static_cast<void *>( lNewLayout );
    }

    void UISplitter_Destroy( void *aInstance ) { delete static_cast<UISplitter *>( aInstance ); }

    void UISplitter_Add1( void *aInstance, void *aChild )
    {
        auto lInstance = static_cast<UISplitter *>( aInstance );
        auto lChild    = static_cast<UIComponent *>( aChild );

        lInstance->Add1( lChild );
    }

    void UISplitter_Add2( void *aInstance, void *aChild )
    {
        auto lInstance = static_cast<UISplitter *>( aInstance );
        auto lChild    = static_cast<UIComponent *>( aChild );

        lInstance->Add2( lChild );
    }

    void UISplitter_SetItemSpacing( void *aInstance, float aItemSpacing )
    {
        auto lInstance = static_cast<UISplitter *>( aInstance );

        lInstance->SetItemSpacing( aItemSpacing );
    }

    void *UIStackLayout_Create()
    {
        auto lNewLayout = new UIStackLayout();

        return static_cast<void *>( lNewLayout );
    }

    void UIStackLayout_Destroy( void *aInstance ) { delete static_cast<UIStackLayout *>( aInstance ); }

    void UIStackLayout_Add( void *aInstance, void *aChild, void *aKey )
    {
        auto lInstance = static_cast<UIStackLayout *>( aInstance );
        auto lChild    = static_cast<UIComponent *>( aChild );
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aKey ) );

        lInstance->Add( lChild, lString );
    }

    void UIStackLayout_SetCurrent( void *aInstance, void *aKey )
    {
        auto lInstance = static_cast<UIStackLayout *>( aInstance );
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aKey ) );

        lInstance->SetCurrent( lString );
    }

    void *UIZLayout_Create()
    {
        auto lNewLayout = new UIZLayout();

        return static_cast<void *>( lNewLayout );
    }

    void UIZLayout_Destroy( void *aInstance ) { delete static_cast<UIZLayout *>( aInstance ); }

    void UIZLayout_AddAlignedNonFixed( void *aInstance, void *aChild, bool aExpand, bool aFill,
                                                  eHorizontalAlignment aHAlignment, eVerticalAlignment aVAlignment )
    {
        auto lInstance = static_cast<UIZLayout *>( aInstance );
        auto lChild    = static_cast<UIComponent *>( aChild );

        lInstance->Add( lChild, aExpand, aFill, aHAlignment, aVAlignment );
    }

    void UIZLayout_AddNonAlignedNonFixed( void *aInstance, void *aChild, bool aExpand, bool aFill )
    {
        auto lInstance = static_cast<UIZLayout *>( aInstance );
        auto lChild    = static_cast<UIComponent *>( aChild );

        lInstance->Add( lChild, aExpand, aFill );
    }

    void UIZLayout_AddAlignedFixed( void *aInstance, void *aChild, math::vec2 aSize, math::vec2 aPosition, bool aExpand,
                                               bool aFill, eHorizontalAlignment aHAlignment, eVerticalAlignment aVAlignment )
    {
        auto lInstance = static_cast<UIZLayout *>( aInstance );
        auto lChild    = static_cast<UIComponent *>( aChild );

        lInstance->Add( lChild, aSize, aPosition, aExpand, aFill, aHAlignment, aVAlignment );
    }

    void UIZLayout_AddNonAlignedFixed( void *aInstance, void *aChild, math::vec2 aSize, math::vec2 aPosition, bool aExpand,
                                                  bool aFill )
    {
        auto lInstance = static_cast<UIZLayout *>( aInstance );
        auto lChild    = static_cast<UIComponent *>( aChild );

        lInstance->Add( lChild, aSize, aPosition, aExpand, aFill );
    }

    void *UIFileTree_Create()
    {
        auto lNewLabel = new UIFileTree();

        return static_cast<void *>( lNewLabel );
    }

    void UIFileTree_Destroy( void *aInstance ) { delete static_cast<UIFileTree *>( aInstance ); }

    void *UIFileTree_Add( void *aInstance, void *aPath )
    {
        auto lInstance = static_cast<UIFileTree *>( aInstance );
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aPath ) );

        return static_cast<void *>( lInstance->Add( lString ) );
    }

    void *UIDialog_Create()
    {
        auto lNewDialog = new UIDialog();

        return static_cast<void *>( lNewDialog );
    }

    void *UIDialog_CreateWithTitleAndSize( void *aTitle, math::vec2 *aSize )
    {
        auto lString    = DotNetRuntime::NewString( static_cast<MonoString *>( aTitle ) );
        auto lNewDialog = new UIDialog( lString, *aSize );

        return static_cast<void *>( lNewDialog );
    }

    void UIDialog_Destroy( void *aInstance ) { delete static_cast<UIDialog *>( aInstance ); }

    void UIDialog_SetTitle( void *aInstance, void *aTitle )
    {
        auto lInstance = static_cast<UIDialog *>( aInstance );
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aTitle ) );

        lInstance->SetTitle( lString );
    }

    void UIDialog_SetSize( void *aInstance, math::vec2 aSize )
    {
        auto lInstance = static_cast<UIDialog *>( aInstance );

        lInstance->SetSize( aSize );
    }

    void UIDialog_SetContent( void *aInstance, void *aContent )
    {
        auto lInstance = static_cast<UIDialog *>( aInstance );
        auto lContent  = static_cast<UIComponent *>( aContent );

        lInstance->SetContent( lContent );
    }

    void UIDialog_Open( void *aInstance )
    {
        auto lInstance = static_cast<UIDialog *>( aInstance );

        lInstance->Open();
    }

    void UIDialog_Close( void *aInstance )
    {
        auto lInstance = static_cast<UIDialog *>( aInstance );

        lInstance->Close();
    }

    void UIDialog_Update( void *aInstance )
    {
        auto lInstance = static_cast<UIDialog *>( aInstance );

        lInstance->Update();
    }

    void *UIForm_Create()
    {
        auto lNewForm = new UIForm();

        return static_cast<void *>( lNewForm );
    }

    void UIForm_Destroy( void *aInstance ) { delete static_cast<UIForm *>( aInstance ); }

    void UIForm_SetTitle( void *aInstance, void *aTitle )
    {
        auto lInstance = static_cast<UIForm *>( aInstance );
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aTitle ) );

        lInstance->SetTitle( lString );
    }

    void UIForm_SetContent( void *aInstance, void *aContent )
    {
        auto lInstance = static_cast<UIForm *>( aInstance );
        auto lContent  = static_cast<UIComponent *>( aContent );

        lInstance->SetContent( lContent );
    }

    void UIForm_Update( void *aInstance )
    {
        auto lInstance = static_cast<UIForm *>( aInstance );

        lInstance->Update();
    }

    void UIForm_SetSize( void *aInstance, float aWidth, float aHeight )
    {
        auto lInstance = static_cast<UIForm *>( aInstance );

        lInstance->SetSize( aWidth, aHeight );
    }
} // namespace SE::Core::Interop