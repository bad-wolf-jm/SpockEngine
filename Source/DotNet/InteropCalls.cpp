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
#include "UI/Components/VectorEdit.h"
#include "UI/Components/Workspace.h"

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

#define BEGIN_INTERFACE_DEFINITION( name )
#define END_INTERFACE_DEFINITION

#define CONSTRUCT_WITHOUT_PARAMETERS( _Ty )       \
    void *_Ty##_Create()                          \
    {                                             \
        auto lNewObject = new _Ty();              \
        return static_cast<void *>( lNewObject ); \
    }

    // clang-format off

BEGIN_INTERFACE_DEFINITION( name )
    CONSTRUCT_WITHOUT_PARAMETERS( UIBaseImage )

    void *UIBaseImage_CreateWithPath( void *aText, math::vec2 aSize )
    {
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );
        auto lNewImage = new UIBaseImage( lString, aSize );

        return static_cast<void *>( lNewImage );
    }

    void UIBaseImage_Destroy( void *aSelf ) { delete static_cast<UIBaseImage *>( aSelf ); }

    void UIBaseImage_SetImage( void *aSelf, void *aPath )
    {
        auto lInstance = static_cast<UIBaseImage *>( aSelf );
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aPath ) );

        lInstance->SetImage( lString );
    }

    void UIBaseImage_SetSize( void *aSelf, math::vec2 aSize )
    {
        auto lInstance = static_cast<UIBaseImage *>( aSelf );

        lInstance->SetSize( aSize );
    }

    math::vec2 UIBaseImage_GetSize( void *aSelf )
    {
        auto lInstance = static_cast<UIBaseImage *>( aSelf );
        auto lV        = lInstance->Size();

        return math::vec2{ lV.y, lV.y };
    }

    void UIBaseImage_SetTopLeft( void *aSelf, math::vec2 aTopLeft )
    {
        auto lInstance = static_cast<UIBaseImage *>( aSelf );

        lInstance->SetTopLeft( aTopLeft );
    }

    math::vec2 UIBaseImage_GetTopLeft( void *aSelf )
    {
        auto lInstance = static_cast<UIBaseImage *>( aSelf );
        auto lV        = lInstance->TopLeft();

        return math::vec2{ lV.y, lV.y };
    }

    void UIBaseImage_SetBottomRight( void *aSelf, math::vec2 aBottomRight )
    {
        auto lInstance = static_cast<UIBaseImage *>( aSelf );

        lInstance->SetBottomRight( aBottomRight );
    }

    math::vec2 UIBaseImage_GetBottomRight( void *aSelf )
    {
        auto lInstance = static_cast<UIBaseImage *>( aSelf );
        auto lV        = lInstance->BottomRight();

        return math::vec2{ lV.x, lV.y };
    }

    void UIBaseImage_SetTintColor( void *aSelf, math::vec4 aColor )
    {
        auto lInstance = static_cast<UIBaseImage *>( aSelf );

        lInstance->SetTintColor( aColor );
    }

    math::vec4 UIBaseImage_GetTintColor( void *aSelf )
    {
        auto lInstance = static_cast<UIBaseImage *>( aSelf );
        auto lV        = lInstance->TintColor();

        return math::vec4{ lV.x, lV.y, lV.z, lV.w };
    }

BEGIN_INTERFACE_DEFINITION( name )
    CONSTRUCT_WITHOUT_PARAMETERS( UIButton )

    void *UIButton_CreateWithText( void *aText )
    {
        auto lString    = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );
        auto lNewButton = new UIButton( lString );

        return static_cast<void *>( lNewButton );
    }

    void UIButton_Destroy( void *aSelf ) { delete static_cast<UILabel *>( aSelf ); }

    void UIButton_SetText( void *aSelf, void *aText )
    {
        auto lInstance = static_cast<UILabel *>( aSelf );
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );

        lInstance->SetText( lString );
    }

    void UIButton_OnClick( void *aSelf, void *aDelegate )
    {
        auto lInstance = static_cast<UIButton *>( aSelf );
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

BEGIN_INTERFACE_DEFINITION( name )
    CONSTRUCT_WITHOUT_PARAMETERS( UICheckBox )

    void UICheckBox_Destroy( void *aSelf ) { delete static_cast<UICheckBox *>( aSelf ); }

    void UICheckBox_OnClick( void *aSelf, void *aDelegate )
    {
        auto lInstance = static_cast<UICheckBox *>( aSelf );
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

    bool UICheckBox_IsChecked( void *aSelf )
    {
        auto lInstance = static_cast<UICheckBox *>( aSelf );

        return lInstance->IsChecked();
    }

    void UICheckBox_SetIsChecked( void *aSelf, bool aValue )
    {
        auto lInstance = static_cast<UICheckBox *>( aSelf );

        lInstance->SetIsChecked( aValue );
    }

BEGIN_INTERFACE_DEFINITION( name )
    CONSTRUCT_WITHOUT_PARAMETERS( UIColorButton )

    void UIColorButton_Destroy( void *aSelf ) { delete static_cast<UIColorButton *>( aSelf ); }

BEGIN_INTERFACE_DEFINITION( name )
    CONSTRUCT_WITHOUT_PARAMETERS( UIComboBox )

    void *UIComboBox_CreateWithItems( void *aItems )
    {
        std::vector<std::string> lItemVector;
        for( auto const &x : DotNetRuntime::AsVector<MonoString *>( static_cast<MonoObject *>( aItems ) ) )
            lItemVector.emplace_back( DotNetRuntime::NewString( x ) );

        auto lNewComboBox = new UIComboBox( lItemVector );

        return static_cast<void *>( lNewComboBox );
    }

    void UIComboBox_Destroy( void *aSelf ) { delete static_cast<UIComboBox *>( aSelf ); }

    int UIComboBox_GetCurrent( void *aSelf )
    {
        auto lInstance = static_cast<UIComboBox *>( aSelf );

        return lInstance->Current();
    }

    void UIComboBox_SetCurrent( void *aSelf, int aValue )
    {
        auto lInstance = static_cast<UIComboBox *>( aSelf );

        lInstance->SetCurrent( aValue );
    }

    void UIComboBox_SetItemList( void *aSelf, void *aItems )
    {
        auto lInstance = static_cast<UIComboBox *>( aSelf );

        std::vector<std::string> lItemVector;
        for( auto const &x : DotNetRuntime::AsVector<MonoString *>( static_cast<MonoObject *>( aItems ) ) )
            lItemVector.emplace_back( DotNetRuntime::NewString( x ) );

        lInstance->SetItemList( lItemVector );
    }

    void UIComboBox_OnChanged( void *aSelf, void *aDelegate )
    {
        auto lInstance = static_cast<UIComboBox *>( aSelf );
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

BEGIN_INTERFACE_DEFINITION( name )
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

BEGIN_INTERFACE_DEFINITION( name )
    CONSTRUCT_WITHOUT_PARAMETERS( UIDropdownButton )

    void UIDropdownButton_Destroy( void *aSelf ) { delete static_cast<UIDropdownButton *>( aSelf ); }

    void UIDropdownButton_SetContent( void *aSelf, void *aContent )
    {
        auto lInstance = static_cast<UIDropdownButton *>( aSelf );
        auto lContent  = static_cast<UIComponent *>( aContent );

        return lInstance->SetContent( lContent );
    }

    void UIDropdownButton_SetContentSize( void *aSelf, math::vec2 aContentSizse )
    {
        auto lInstance = static_cast<UIDropdownButton *>( aSelf );

        return lInstance->SetContentSize( aContentSizse );
    }

    void UIDropdownButton_SetImage( void *aSelf, void *aImage )
    {
        auto lInstance = static_cast<UIDropdownButton *>( aSelf );
        auto lImage    = static_cast<UIBaseImage *>( aImage );

        lInstance->SetImage( lImage );
    }

    void UIDropdownButton_SetText( void *aSelf, void *aText )
    {
        auto lInstance = static_cast<UIDropdownButton *>( aSelf );
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );

        lInstance->SetText( lString );
    }

    void UIDropdownButton_SetTextColor( void *aSelf, math::vec4 aColor )
    {
        auto lInstance = static_cast<UIDropdownButton *>( aSelf );

        lInstance->SetTextColor( aColor );
    }

BEGIN_INTERFACE_DEFINITION( name )
    CONSTRUCT_WITHOUT_PARAMETERS( UIImage )

    void *UIImage_CreateWithPath( void *aText, math::vec2 aSize )
    {
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );
        auto lNewImage = new UIImage( lString, aSize );

        return static_cast<void *>( lNewImage );
    }

    void UIImage_Destroy( void *aSelf ) { delete static_cast<UIImage *>( aSelf ); }

BEGIN_INTERFACE_DEFINITION( name )
    CONSTRUCT_WITHOUT_PARAMETERS( UIImageButton )

    void *UIImageButton_CreateWithPath( void *aText, math::vec2 *aSize )
    {
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );
        auto lNewImage = new UIImageButton( lString, *aSize );

        return static_cast<void *>( lNewImage );
    }

    void UIImageButton_Destroy( void *aSelf ) { delete static_cast<UIImageButton *>( aSelf ); }

    void UIImageButton_OnClick( void *aSelf, void *aDelegate )
    {
        auto lInstance = static_cast<UIImageButton *>( aSelf );
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

BEGIN_INTERFACE_DEFINITION( name )
    CONSTRUCT_WITHOUT_PARAMETERS( UIImageToggleButton )

    void UIImageToggleButton_Destroy( void *aSelf ) { delete static_cast<UIImageToggleButton *>( aSelf ); }

    bool UIImageToggleButton_IsActive( void *aSelf )
    {
        auto lInstance = static_cast<UIImageToggleButton *>( aSelf );

        return lInstance->IsActive();
    }

    void UIImageToggleButton_SetActive( void *aSelf, bool aValue )
    {
        auto lInstance = static_cast<UIImageToggleButton *>( aSelf );

        lInstance->SetActive( aValue );
    }

    void UIImageToggleButton_SetActiveImage( void *aSelf, void *aImage )
    {
        auto lInstance = static_cast<UIImageToggleButton *>( aSelf );
        auto lImage    = static_cast<UIBaseImage *>( aImage );

        lInstance->SetActiveImage( lImage );
    }

    void UIImageToggleButton_SetInactiveImage( void *aSelf, void *aImage )
    {
        auto lInstance = static_cast<UIImageToggleButton *>( aSelf );
        auto lImage    = static_cast<UIBaseImage *>( aImage );

        lInstance->SetInactiveImage( lImage );
    }

    void UIImageToggleButton_OnClicked( void *aSelf, void *aDelegate )
    {
        auto lInstance = static_cast<UIImageToggleButton *>( aSelf );
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

    void UIImageToggleButton_OnChanged( void *aSelf, void *aDelegate )
    {
        auto lInstance = static_cast<UIImageToggleButton *>( aSelf );
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

BEGIN_INTERFACE_DEFINITION( name )
    CONSTRUCT_WITHOUT_PARAMETERS( UILabel )

    void *UILabel_CreateWithText( void *aText )
    {
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );
        auto lNewLabel = new UILabel( lString );

        return static_cast<void *>( lNewLabel );
    }

    void UILabel_Destroy( void *aSelf ) { delete static_cast<UILabel *>( aSelf ); }

    void UILabel_SetText( void *aSelf, void *aText )
    {
        auto lInstance = static_cast<UILabel *>( aSelf );
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );

        lInstance->SetText( lString );
    }

    void UILabel_SetTextColor( void *aSelf, math::vec4 aTextColor )
    {
        auto lInstance = static_cast<UILabel *>( aSelf );

        lInstance->SetTextColor( aTextColor );
    }

BEGIN_INTERFACE_DEFINITION( name )
    CONSTRUCT_WITHOUT_PARAMETERS( UIMenuItem )

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

    void UIMenuItem_Destroy( void *aSelf ) { delete static_cast<UIMenuItem *>( aSelf ); }

    void UIMenuItem_SetText( void *aSelf, void *aText )
    {
        auto lInstance = static_cast<UIMenuItem *>( aSelf );
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );

        lInstance->SetText( lString );
    }

    void UIMenuItem_SetShortcut( void *aSelf, void *aShortcut )
    {
        auto lInstance = static_cast<UIMenuItem *>( aSelf );
        auto lShortcut = DotNetRuntime::NewString( static_cast<MonoString *>( aShortcut ) );

        lInstance->SetShortcut( lShortcut );
    }

    void UIMenuItem_SetTextColor( void *aSelf, math::vec4 *aTextColor )
    {
        auto lInstance = static_cast<UIMenuItem *>( aSelf );

        lInstance->SetTextColor( *aTextColor );
    }

    void UIMenuItem_OnTrigger( void *aSelf, void *aDelegate )
    {
        auto lInstance = static_cast<UIMenuItem *>( aSelf );
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

BEGIN_INTERFACE_DEFINITION( name )
    CONSTRUCT_WITHOUT_PARAMETERS( UIMenuSeparator )

    void UIMenuSeparator_Destroy( void *aSelf ) { delete static_cast<UIMenuSeparator *>( aSelf ); }

BEGIN_INTERFACE_DEFINITION( name )
    CONSTRUCT_WITHOUT_PARAMETERS( UIMenu )

    void *UIMenu_CreateWithText( void *aText )
    {
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );
        auto lNewLabel = new UIMenu( lString );

        return static_cast<void *>( lNewLabel );
    }

    void UIMenu_Destroy( void *aSelf ) { delete static_cast<UIMenu *>( aSelf ); }

    void *UIMenu_AddAction( void *aSelf, void *aText, void *aShortcut )
    {
        auto lInstance  = static_cast<UIMenu *>( aSelf );
        auto lString    = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );
        auto lShortcut  = DotNetRuntime::NewString( static_cast<MonoString *>( aShortcut ) );
        auto lNewAction = lInstance->AddActionRaw( lString, lShortcut );

        return static_cast<void *>( lNewAction );
    }

    void *UIMenu_AddMenu( void *aSelf, void *aText )
    {
        auto lInstance = static_cast<UIMenu *>( aSelf );
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );
        auto lNewMenu  = lInstance->AddMenuRaw( lString );

        return static_cast<void *>( lNewMenu );
    }

    void *UIMenu_AddSeparator( void *aSelf )
    {
        auto lInstance     = static_cast<UIMenu *>( aSelf );
        auto lNewSeparator = lInstance->AddSeparatorRaw();

        return static_cast<void *>( lNewSeparator );
    }

    void UIMenu_Update( void *aSelf )
    {
        auto lInstance = static_cast<UIMenu *>( aSelf );

        lInstance->Update();
    }

BEGIN_INTERFACE_DEFINITION( name )
    CONSTRUCT_WITHOUT_PARAMETERS( UIPlot )

    void UIPlot_Destroy( void *aSelf ) { delete static_cast<UIPlot *>( aSelf ); }

    void UIPlot_Clear( void *aSelf )
    {
        auto lSelf = static_cast<UIPlot *>( aSelf );

        lSelf->Clear();
    }

    void UIPlot_ConfigureLegend( void *aSelf, math::vec2 *aLegendPadding, math::vec2 *aLegendInnerPadding,
                                 math::vec2 *aLegendSpacing )
    {
        auto lSelf = static_cast<UIPlot *>( aSelf );

        lSelf->ConfigureLegend( *aLegendPadding, *aLegendInnerPadding, *aLegendSpacing );
    }

    void UIPlot_Add( void *aSelf, void *aPlot )
    {
        auto lSelf = static_cast<UIPlot *>( aSelf );
        auto lPlot = static_cast<sPlotData *>( aPlot );

        lSelf->Add( lPlot );
    }

    void UIPlot_SetAxisLimits( void *aSelf, int aAxis, double aMin, double aMax )
    {
        auto lSelf = static_cast<UIPlot *>( aSelf );

        lSelf->mAxisConfiguration[aAxis].mSetLimitRequest = true;

        lSelf->mAxisConfiguration[aAxis].mMin = static_cast<float>( aMin );
        lSelf->mAxisConfiguration[aAxis].mMax = static_cast<float>( aMax );
    }

    void UIPlot_SetAxisTitle( void *aSelf, int aAxis, void *aTitle )
    {
        auto lSelf = static_cast<UIPlot *>( aSelf );

        lSelf->mAxisConfiguration[aAxis].mTitle = DotNetRuntime::NewString( static_cast<MonoString *>( aTitle ) );
    }

    void *UIPlot_GetAxisTitle( void *aSelf, int aAxis )
    {
        auto lSelf = static_cast<UIPlot *>( aSelf );

        return DotNetRuntime::NewString( lSelf->mAxisConfiguration[aAxis].mTitle );
    }

BEGIN_INTERFACE_DEFINITION( name )
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

BEGIN_INTERFACE_DEFINITION( name )
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

BEGIN_INTERFACE_DEFINITION( name )
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

BEGIN_INTERFACE_DEFINITION( name )
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

BEGIN_INTERFACE_DEFINITION( name )
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

BEGIN_INTERFACE_DEFINITION( name )
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

BEGIN_INTERFACE_DEFINITION( name )
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

BEGIN_INTERFACE_DEFINITION( name )
    CONSTRUCT_WITHOUT_PARAMETERS( UIProgressBar )

    void UIProgressBar_Destroy( void *aSelf ) { delete static_cast<UIProgressBar *>( aSelf ); }

    void UIProgressBar_SetProgressValue( void *aSelf, float aValue )
    {
        auto lInstance = static_cast<UIProgressBar *>( aSelf );

        lInstance->SetProgressValue( aValue );
    }

    void UIProgressBar_SetProgressColor( void *aSelf, math::vec4 aTextColor )
    {
        auto lInstance = static_cast<UIProgressBar *>( aSelf );

        lInstance->SetProgressColor( aTextColor );
    }

    void UIProgressBar_SetText( void *aSelf, void *aText )
    {
        auto lInstance = static_cast<UIProgressBar *>( aSelf );
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );

        lInstance->SetText( lString );
    }

    void UIProgressBar_SetTextColor( void *aSelf, math::vec4 aTextColor )
    {
        auto lInstance = static_cast<UIProgressBar *>( aSelf );

        lInstance->SetTextColor( aTextColor );
    }

    void UIProgressBar_SetThickness( void *aSelf, float aValue )
    {
        auto lInstance = static_cast<UIProgressBar *>( aSelf );

        lInstance->SetThickness( aValue );
    }

BEGIN_INTERFACE_DEFINITION( name )
    CONSTRUCT_WITHOUT_PARAMETERS( UIPropertyValue )

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

    void UIPropertyValue_Destroy( void *aSelf ) { delete static_cast<UIPropertyValue *>( aSelf ); }

    void UIPropertyValue_SetValue( void *aSelf, void *aText )
    {
        auto lInstance = static_cast<UIPropertyValue *>( aSelf );
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );

        lInstance->SetValue( lString );
    }

    void UIPropertyValue_SetValueFont( void *aSelf, FontFamilyFlags aFont )
    {
        auto lInstance = static_cast<UIPropertyValue *>( aSelf );

        lInstance->SetValueFont( aFont );
    }

    void UIPropertyValue_SetNameFont( void *aSelf, FontFamilyFlags aFont )
    {
        auto lInstance = static_cast<UIPropertyValue *>( aSelf );

        lInstance->SetNameFont( aFont );
    }

BEGIN_INTERFACE_DEFINITION( name )
    CONSTRUCT_WITHOUT_PARAMETERS( UISlider )

    void UISlider_Destroy( void *aSelf ) { delete static_cast<UISlider *>( aSelf ); }

BEGIN_INTERFACE_DEFINITION( name )
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

BEGIN_INTERFACE_DEFINITION( name )
    CONSTRUCT_WITHOUT_PARAMETERS( UITable )

    void UITable_Destroy( void *aSelf ) { delete static_cast<UITable *>( aSelf ); }

    void UITable_OnRowClicked( void *aSelf, void *aDelegate )
    {
        auto lInstance = static_cast<UITable *>( aSelf );
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

BEGIN_INTERFACE_DEFINITION( name )
    // CONSTRUCT_WITHOUT_PARAMETERS( UITable )
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

BEGIN_INTERFACE_DEFINITION( name )
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

BEGIN_INTERFACE_DEFINITION( name )
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

BEGIN_INTERFACE_DEFINITION( name )
    CONSTRUCT_WITHOUT_PARAMETERS( UITextInput )

    void *UITextInput_CreateWithText( void *aText )
    {
        auto lString       = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );
        auto lNewTextInput = new UITextInput( lString );

        return static_cast<void *>( lNewTextInput );
    }

    void UITextInput_Destroy( void *aSelf ) { delete static_cast<UITextInput *>( aSelf ); }

    void UITextInput_SetHintText( void *aSelf, void *aText )
    {
        auto lInstance = static_cast<UITextInput *>( aSelf );
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );

        lInstance->SetHintText( lString );
    }

    void *UITextInput_GetText( void *aSelf )
    {
        auto lInstance = static_cast<UITextInput *>( aSelf );

        return DotNetRuntime::NewString( lInstance->GetText() );
    }

    void UITextInput_SetTextColor( void *aSelf, math::vec4 *aTextColor )
    {
        auto lInstance = static_cast<UITextInput *>( aSelf );

        lInstance->SetTextColor( *aTextColor );
    }

    void UITextInput_SetBufferSize( void *aSelf, uint32_t aBufferSize )
    {
        auto lInstance = static_cast<UITextInput *>( aSelf );

        lInstance->SetBuffersize( aBufferSize );
    }

    void UITextInput_OnTextChanged( void *aSelf, void *aDelegate )
    {
        auto lInstance = static_cast<UITextInput *>( aSelf );
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

BEGIN_INTERFACE_DEFINITION( name )
    CONSTRUCT_WITHOUT_PARAMETERS( UITextOverlay )

    void UITextOverlay_Destroy( void *aSelf ) { delete static_cast<UITextOverlay *>( aSelf ); }

    void UITextOverlay_AddText( void *aSelf, void *aText )
    {
        auto lInstance = static_cast<UITextOverlay *>( aSelf );
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );

        lInstance->AddText( lString );
    }

    void UITextOverlay_Clear( void *aSelf )
    {
        auto lInstance = static_cast<UITextOverlay *>( aSelf );

        lInstance->Clear();
    }

BEGIN_INTERFACE_DEFINITION( name )
    CONSTRUCT_WITHOUT_PARAMETERS( UITextToggleButton )

    void *UITextToggleButton_CreateWithText( void *aText )
    {
        auto lString    = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );
        auto lNewButton = new UITextToggleButton( lString );

        return static_cast<void *>( lNewButton );
    }

    void UITextToggleButton_Destroy( void *aSelf ) { delete static_cast<UITextToggleButton *>( aSelf ); }

    bool UITextToggleButton_IsActive( void *aSelf )
    {
        auto lInstance = static_cast<UITextToggleButton *>( aSelf );

        return lInstance->IsActive();
    }

    void UITextToggleButton_SetActive( void *aSelf, bool aValue )
    {
        auto lInstance = static_cast<UITextToggleButton *>( aSelf );

        lInstance->SetActive( aValue );
    }

    void UITextToggleButton_SetActiveColor( void *aSelf, math::vec4 *aColor )
    {
        auto lInstance = static_cast<UITextToggleButton *>( aSelf );

        lInstance->SetActiveColor( *aColor );
    }

    void UITextToggleButton_SetInactiveColor( void *aSelf, math::vec4 *aColor )
    {
        auto lInstance = static_cast<UITextToggleButton *>( aSelf );

        lInstance->SetInactiveColor( *aColor );
    }

    void UITextToggleButton_OnClicked( void *aSelf, void *aDelegate )
    {
        auto lInstance = static_cast<UITextToggleButton *>( aSelf );
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

    void UITextToggleButton_OnChanged( void *aSelf, void *aDelegate )
    {
        auto lInstance = static_cast<UITextToggleButton *>( aSelf );
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

BEGIN_INTERFACE_DEFINITION( name )
    CONSTRUCT_WITHOUT_PARAMETERS( UITreeViewNode )

    void UITreeViewNode_Destroy( void *aSelf ) { delete static_cast<UITreeViewNode *>( aSelf ); }

    void UITreeViewNode_SetText( void *aSelf, void *aText )
    {
        auto lInstance = static_cast<UITreeViewNode *>( aSelf );
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );

        lInstance->SetText( lString );
    }

    void UITreeViewNode_SetTextColor( void *aSelf, math::vec4 aTextColor )
    {
        auto lInstance = static_cast<UITreeViewNode *>( aSelf );

        lInstance->SetTextColor( aTextColor );
    }

    void UITreeViewNode_SetIcon( void *aSelf, void *aIcon )
    {
        auto lInstance = static_cast<UITreeViewNode *>( aSelf );
        auto lImage    = static_cast<UIImage *>( aIcon );

        lInstance->SetIcon( lImage );
    }

    void UITreeViewNode_SetIndicator( void *aSelf, void *aIndicator )
    {
        auto lInstance = static_cast<UITreeViewNode *>( aSelf );
        auto lImage    = static_cast<UIComponent *>( aIndicator );

        lInstance->SetIndicator( lImage );
    }

    void *UITreeViewNode_Add( void *aSelf )
    {
        auto lInstance = static_cast<UITreeViewNode *>( aSelf );

        return static_cast<void *>( lInstance->Add() );
    }

BEGIN_INTERFACE_DEFINITION( name )
    CONSTRUCT_WITHOUT_PARAMETERS( UITreeView )

    void UITreeView_Destroy( void *aSelf ) { delete static_cast<UITreeView *>( aSelf ); }

    void UITreeView_SetIndent( void *aSelf, float aIndent )
    {
        auto lInstance = static_cast<UITreeView *>( aSelf );

        lInstance->SetIndent( aIndent );
    }

    void UITreeView_SetIconSpacing( void *aSelf, float aSpacing )
    {
        auto lInstance = static_cast<UITreeView *>( aSelf );

        lInstance->SetIconSpacing( aSpacing );
    }

    void *UITreeView_Add( void *aSelf )
    {
        auto lInstance = static_cast<UITreeView *>( aSelf );

        return static_cast<void *>( lInstance->Add() );
    }

BEGIN_INTERFACE_DEFINITION( name )
    CONSTRUCT_WITHOUT_PARAMETERS( UIVec2Input )

    void UIVec2Input_Destroy( void *aSelf ) { delete static_cast<UIVec2Input *>( aSelf ); }

    void UIVec2Input_OnChanged( void *aSelf, void *aDelegate )
    {
        auto lInstance = static_cast<UIVectorInputBase *>( aSelf );
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

    void UIVec2Input_SetValue( void *aSelf, math::vec2 aValue ) { static_cast<UIVec2Input *>( aSelf )->SetValue( aValue ); }

    math::vec2 UIVec2Input_GetValue( void *aSelf ) { return static_cast<UIVec2Input *>( aSelf )->Value(); }

    void UIVec2Input_SetResetValues( void *aSelf, math::vec2 aValue )
    {
        static_cast<UIVec2Input *>( aSelf )->SetResetValues( aValue );
    }

    void UIVec2Input_SetFormat( void *aSelf, void *aText )
    {
        auto lInstance = static_cast<UIVectorInputBase *>( aSelf );
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );

        lInstance->SetFormat( lString );
    }

BEGIN_INTERFACE_DEFINITION( name )
    CONSTRUCT_WITHOUT_PARAMETERS( UIVec3Input )

    void UIVec3Input_Destroy( void *aSelf ) { delete static_cast<UIVec3Input *>( aSelf ); }

    void UIVec3Input_OnChanged( void *aSelf, void *aDelegate )
    {
        auto lInstance = static_cast<UIVectorInputBase *>( aSelf );
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

    void UIVec3Input_SetValue( void *aSelf, math::vec3 aValue ) { static_cast<UIVec3Input *>( aSelf )->SetValue( aValue ); }

    math::vec3 UIVec3Input_GetValue( void *aSelf ) { return static_cast<UIVec3Input *>( aSelf )->Value(); }

    void UIVec3Input_SetResetValues( void *aSelf, math::vec3 aValue )
    {
        static_cast<UIVec3Input *>( aSelf )->SetResetValues( aValue );
    }

    void UIVec3Input_SetFormat( void *aSelf, void *aText )
    {
        auto lInstance = static_cast<UIVectorInputBase *>( aSelf );
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );

        lInstance->SetFormat( lString );
    }

BEGIN_INTERFACE_DEFINITION( name )
     CONSTRUCT_WITHOUT_PARAMETERS( UIVec4Input )

    void UIVec4Input_Destroy( void *aSelf ) { delete static_cast<UIVec4Input *>( aSelf ); }

    void UIVec4Input_OnChanged( void *aSelf, void *aDelegate )
    {
        auto lInstance = static_cast<UIVectorInputBase *>( aSelf );
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

    void UIVec4Input_SetValue( void *aSelf, math::vec4 aValue ) { static_cast<UIVec4Input *>( aSelf )->SetValue( aValue ); }

    math::vec4 UIVec4Input_GetValue( void *aSelf ) { return static_cast<UIVec4Input *>( aSelf )->Value(); }

    void UIVec4Input_SetResetValues( void *aSelf, math::vec4 aValue )
    {
        static_cast<UIVec4Input *>( aSelf )->SetResetValues( aValue );
    }

    void UIVec4Input_SetFormat( void *aSelf, void *aText )
    {
        auto lInstance = static_cast<UIVectorInputBase *>( aSelf );
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );

        lInstance->SetFormat( lString );
    }

BEGIN_INTERFACE_DEFINITION( name )
     CONSTRUCT_WITHOUT_PARAMETERS( UIWorkspaceDocument )

    void UIWorkspaceDocument_Destroy( void *aSelf ) { delete static_cast<UIWorkspaceDocument *>( aSelf ); }

    void UIWorkspaceDocument_SetContent( void *aSelf, void *aContent )
    {
        auto lInstance = static_cast<UIWorkspaceDocument *>( aSelf );
        auto lContent  = static_cast<UIComponent *>( aContent );

        lInstance->SetContent( lContent );
    }

    void UIWorkspaceDocument_Update( void *aSelf )
    {
        auto lInstance = static_cast<UIWorkspaceDocument *>( aSelf );

        lInstance->Update();
    }

    void UIWorkspaceDocument_SetName( void *aSelf, void *aName )
    {
        auto lInstance = static_cast<UIWorkspaceDocument *>( aSelf );
        auto lName     = DotNetRuntime::NewString( static_cast<MonoString *>( aName ) );

        lInstance->mName = lName;
    }

    bool UIWorkspaceDocument_IsDirty( void *aSelf )
    {
        auto lInstance = static_cast<UIWorkspaceDocument *>( aSelf );

        return lInstance->mDirty;
    }

    void UIWorkspaceDocument_MarkAsDirty( void *aSelf, bool aDirty )
    {
        auto lInstance = static_cast<UIWorkspaceDocument *>( aSelf );

        lInstance->mDirty = aDirty;
    }

    void UIWorkspaceDocument_Open( void *aSelf )
    {
        auto lInstance = static_cast<UIWorkspaceDocument *>( aSelf );

        lInstance->DoOpen();
    }

    void UIWorkspaceDocument_RequestClose( void *aSelf )
    {
        auto lInstance = static_cast<UIWorkspaceDocument *>( aSelf );

        lInstance->DoQueueClose();
    }

    void UIWorkspaceDocument_ForceClose( void *aSelf )
    {
        auto lInstance = static_cast<UIWorkspaceDocument *>( aSelf );

        lInstance->DoForceClose();
    }

    void UIWorkspaceDocument_RegisterSaveDelegate( void *aSelf, void *aDelegate )
    {
        auto lInstance = static_cast<UIWorkspaceDocument *>( aSelf );
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

BEGIN_INTERFACE_DEFINITION( name )
    CONSTRUCT_WITHOUT_PARAMETERS( UIWorkspace )

    void UIWorkspace_Destroy( void *aSelf ) { delete static_cast<UIWorkspace *>( aSelf ); }

    void UIWorkspace_Add( void *aSelf, void *aDocument )
    {
        auto lSelf     = static_cast<UIWorkspace *>( aSelf );
        auto lDocument = static_cast<UIWorkspaceDocument *>( aDocument );

        lSelf->Add( lDocument );
    }

    void UIWorkspace_RegisterCloseDocumentDelegate( void *aSelf, void *aDelegate )
    {
        auto lInstance = static_cast<UIWorkspace *>( aSelf );
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

BEGIN_INTERFACE_DEFINITION( name )
    void *UIBoxLayout_CreateWithOrientation( eBoxLayoutOrientation aOrientation )
    {
        auto lNewLayout = new UIBoxLayout( aOrientation );

        return static_cast<void *>( lNewLayout );
    }

    void UIBoxLayout_Destroy( void *aSelf ) { delete static_cast<UIBoxLayout *>( aSelf ); }

    void UIBoxLayout_AddAlignedNonFixed( void *aSelf, void *aChild, bool aExpand, bool aFill, eHorizontalAlignment aHAlignment,
                                         eVerticalAlignment aVAlignment )
    {
        auto lInstance = static_cast<UIBoxLayout *>( aSelf );
        auto lChild    = static_cast<UIComponent *>( aChild );

        lInstance->Add( lChild, aExpand, aFill, aHAlignment, aVAlignment );
    }

    void UIBoxLayout_AddNonAlignedNonFixed( void *aSelf, void *aChild, bool aExpand, bool aFill )
    {
        auto lInstance = static_cast<UIBoxLayout *>( aSelf );
        auto lChild    = static_cast<UIComponent *>( aChild );

        lInstance->Add( lChild, aExpand, aFill );
    }

    void UIBoxLayout_AddAlignedFixed( void *aSelf, void *aChild, float aFixedSize, bool aExpand, bool aFill,
                                      eHorizontalAlignment aHAlignment, eVerticalAlignment aVAlignment )
    {
        auto lInstance = static_cast<UIBoxLayout *>( aSelf );
        auto lChild    = static_cast<UIComponent *>( aChild );

        lInstance->Add( lChild, aFixedSize, aExpand, aFill, aHAlignment, aVAlignment );
    }

    void UIBoxLayout_AddNonAlignedFixed( void *aSelf, void *aChild, float aFixedSize, bool aExpand, bool aFill )
    {
        auto lInstance = static_cast<UIBoxLayout *>( aSelf );
        auto lChild    = static_cast<UIComponent *>( aChild );

        lInstance->Add( lChild, aFixedSize, aExpand, aFill );
    }

    void UIBoxLayout_AddSeparator( void *aSelf )
    {
        auto lInstance = static_cast<UIBoxLayout *>( aSelf );

        lInstance->AddSeparator();
    }

    void UIBoxLayout_SetItemSpacing( void *aSelf, float aItemSpacing )
    {
        auto lInstance = static_cast<UIBoxLayout *>( aSelf );

        lInstance->SetItemSpacing( aItemSpacing );
    }

    void UIBoxLayout_Clear( void *aSelf )
    {
        auto lInstance = static_cast<UIBoxLayout *>( aSelf );

        lInstance->Clear();
    }

BEGIN_INTERFACE_DEFINITION( name )
    CONSTRUCT_WITHOUT_PARAMETERS( UIContainer )

    void UIContainer_Destroy( void *aSelf ) { delete static_cast<UIContainer *>( aSelf ); }

    void UIContainer_SetContent( void *aSelf, void *aChild )
    {
        auto lInstance = static_cast<UIContainer *>( aSelf );
        auto lChild    = static_cast<UIComponent *>( aChild );

        lInstance->SetContent( lChild );
    }

BEGIN_INTERFACE_DEFINITION( name )
    CONSTRUCT_WITHOUT_PARAMETERS( UISplitter )

    void *UISplitter_CreateWithOrientation( eBoxLayoutOrientation aOrientation )
    {
        auto lNewLayout = new UISplitter( aOrientation );

        return static_cast<void *>( lNewLayout );
    }

    void UISplitter_Destroy( void *aSelf ) { delete static_cast<UISplitter *>( aSelf ); }

    void UISplitter_Add1( void *aSelf, void *aChild )
    {
        auto lInstance = static_cast<UISplitter *>( aSelf );
        auto lChild    = static_cast<UIComponent *>( aChild );

        lInstance->Add1( lChild );
    }

    void UISplitter_Add2( void *aSelf, void *aChild )
    {
        auto lInstance = static_cast<UISplitter *>( aSelf );
        auto lChild    = static_cast<UIComponent *>( aChild );

        lInstance->Add2( lChild );
    }

    void UISplitter_SetItemSpacing( void *aSelf, float aItemSpacing )
    {
        auto lInstance = static_cast<UISplitter *>( aSelf );

        lInstance->SetItemSpacing( aItemSpacing );
    }

BEGIN_INTERFACE_DEFINITION( name )
    CONSTRUCT_WITHOUT_PARAMETERS( UIStackLayout )

    void UIStackLayout_Destroy( void *aSelf ) { delete static_cast<UIStackLayout *>( aSelf ); }

    void UIStackLayout_Add( void *aSelf, void *aChild, void *aKey )
    {
        auto lInstance = static_cast<UIStackLayout *>( aSelf );
        auto lChild    = static_cast<UIComponent *>( aChild );
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aKey ) );

        lInstance->Add( lChild, lString );
    }

    void UIStackLayout_SetCurrent( void *aSelf, void *aKey )
    {
        auto lInstance = static_cast<UIStackLayout *>( aSelf );
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aKey ) );

        lInstance->SetCurrent( lString );
    }

BEGIN_INTERFACE_DEFINITION( name )
    CONSTRUCT_WITHOUT_PARAMETERS( UIZLayout )

    void UIZLayout_Destroy( void *aSelf ) { delete static_cast<UIZLayout *>( aSelf ); }

    void UIZLayout_AddAlignedNonFixed( void *aSelf, void *aChild, bool aExpand, bool aFill, eHorizontalAlignment aHAlignment,
                                       eVerticalAlignment aVAlignment )
    {
        auto lInstance = static_cast<UIZLayout *>( aSelf );
        auto lChild    = static_cast<UIComponent *>( aChild );

        lInstance->Add( lChild, aExpand, aFill, aHAlignment, aVAlignment );
    }

    void UIZLayout_AddNonAlignedNonFixed( void *aSelf, void *aChild, bool aExpand, bool aFill )
    {
        auto lInstance = static_cast<UIZLayout *>( aSelf );
        auto lChild    = static_cast<UIComponent *>( aChild );

        lInstance->Add( lChild, aExpand, aFill );
    }

    void UIZLayout_AddAlignedFixed( void *aSelf, void *aChild, math::vec2 aSize, math::vec2 aPosition, bool aExpand, bool aFill,
                                    eHorizontalAlignment aHAlignment, eVerticalAlignment aVAlignment )
    {
        auto lInstance = static_cast<UIZLayout *>( aSelf );
        auto lChild    = static_cast<UIComponent *>( aChild );

        lInstance->Add( lChild, aSize, aPosition, aExpand, aFill, aHAlignment, aVAlignment );
    }

    void UIZLayout_AddNonAlignedFixed( void *aSelf, void *aChild, math::vec2 aSize, math::vec2 aPosition, bool aExpand,
                                       bool aFill )
    {
        auto lInstance = static_cast<UIZLayout *>( aSelf );
        auto lChild    = static_cast<UIComponent *>( aChild );

        lInstance->Add( lChild, aSize, aPosition, aExpand, aFill );
    }

BEGIN_INTERFACE_DEFINITION( name )
    CONSTRUCT_WITHOUT_PARAMETERS( UIFileTree )

    void UIFileTree_Destroy( void *aSelf ) { delete static_cast<UIFileTree *>( aSelf ); }

    void *UIFileTree_Add( void *aSelf, void *aPath )
    {
        auto lInstance = static_cast<UIFileTree *>( aSelf );
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aPath ) );

        return static_cast<void *>( lInstance->Add( lString ) );
    }

BEGIN_INTERFACE_DEFINITION( name )
    CONSTRUCT_WITHOUT_PARAMETERS( UIDialog )

    void *UIDialog_CreateWithTitleAndSize( void *aTitle, math::vec2 *aSize )
    {
        auto lString    = DotNetRuntime::NewString( static_cast<MonoString *>( aTitle ) );
        auto lNewDialog = new UIDialog( lString, *aSize );

        return static_cast<void *>( lNewDialog );
    }

    void UIDialog_Destroy( void *aSelf ) { delete static_cast<UIDialog *>( aSelf ); }

    void UIDialog_SetTitle( void *aSelf, void *aTitle )
    {
        auto lInstance = static_cast<UIDialog *>( aSelf );
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aTitle ) );

        lInstance->SetTitle( lString );
    }

    void UIDialog_SetSize( void *aSelf, math::vec2 aSize )
    {
        auto lInstance = static_cast<UIDialog *>( aSelf );

        lInstance->SetSize( aSize );
    }

    void UIDialog_SetContent( void *aSelf, void *aContent )
    {
        auto lInstance = static_cast<UIDialog *>( aSelf );
        auto lContent  = static_cast<UIComponent *>( aContent );

        lInstance->SetContent( lContent );
    }

    void UIDialog_Open( void *aSelf )
    {
        auto lInstance = static_cast<UIDialog *>( aSelf );

        lInstance->Open();
    }

    void UIDialog_Close( void *aSelf )
    {
        auto lInstance = static_cast<UIDialog *>( aSelf );

        lInstance->Close();
    }

    void UIDialog_Update( void *aSelf )
    {
        auto lInstance = static_cast<UIDialog *>( aSelf );

        lInstance->Update();
    }

BEGIN_INTERFACE_DEFINITION( name )
    CONSTRUCT_WITHOUT_PARAMETERS( UIForm )

    void UIForm_Destroy( void *aSelf ) { delete static_cast<UIForm *>( aSelf ); }

    void UIForm_SetTitle( void *aSelf, void *aTitle )
    {
        auto lInstance = static_cast<UIForm *>( aSelf );
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aTitle ) );

        lInstance->SetTitle( lString );
    }

    void UIForm_SetContent( void *aSelf, void *aContent )
    {
        auto lInstance = static_cast<UIForm *>( aSelf );
        auto lContent  = static_cast<UIComponent *>( aContent );

        lInstance->SetContent( lContent );
    }

    void UIForm_Update( void *aSelf )
    {
        auto lInstance = static_cast<UIForm *>( aSelf );

        lInstance->Update();
    }

    void UIForm_SetSize( void *aSelf, float aWidth, float aHeight )
    {
        auto lInstance = static_cast<UIForm *>( aSelf );

        lInstance->SetSize( aWidth, aHeight );
    }

    // clang-format on
} // namespace SE::Core::Interop