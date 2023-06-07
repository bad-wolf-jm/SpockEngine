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

#define __SELF__ void *aSelf

#define CONSTRUCT_WITHOUT_PARAMETERS( _Ty ) \
    void *_Ty##_Create()                    \
    {                                       \
        auto lNewObject = new _Ty();        \
        return CAST( void, lNewObject );    \
    }

#define DESTROY_INTERFACE( _Ty ) \
    void _Ty##_Destroy( __SELF__ ) { delete SELF( _Ty ); }

#define SELF( _Ty ) static_cast<_Ty *>( aSelf )

#define CAST( _Ty, v ) static_cast<_Ty *>( v )

    BEGIN_INTERFACE_DEFINITION( name )
    CONSTRUCT_WITHOUT_PARAMETERS( UIBaseImage )

    void *UIBaseImage_CreateWithPath( void *aText, vec2 aSize )
    {
        auto lString   = DotNetRuntime::NewString( CAST( MonoString, aText ) );
        auto lNewImage = new UIBaseImage( lString, aSize );

        return CAST( void, lNewImage );
    }

    DESTROY_INTERFACE( UIBaseImage )

    void UIBaseImage_SetImage( __SELF__, void *aPath )
    {
        auto lString = DotNetRuntime::NewString( CAST( MonoString, aPath ) );

        SELF( UIBaseImage )->SetImage( lString );
    }

    void UIBaseImage_SetSize( __SELF__, vec2 aSize )
    {
        auto lInstance = static_cast<UIBaseImage *>( aSelf );

        SELF( UIBaseImage )->SetSize( aSize );
    }

    vec2 UIBaseImage_GetSize( __SELF__ )
    {
        auto lV = SELF( UIBaseImage )->Size();

        return vec2{ lV.y, lV.y };
    }

    void UIBaseImage_SetTopLeft( __SELF__, vec2 aTopLeft ) { SELF( UIBaseImage )->SetTopLeft( aTopLeft ); }

    vec2 UIBaseImage_GetTopLeft( __SELF__ )
    {
        auto lV = SELF( UIBaseImage )->TopLeft();

        return vec2{ lV.y, lV.y };
    }

    void UIBaseImage_SetBottomRight( __SELF__, vec2 aBottomRight ) { SELF( UIBaseImage )->SetBottomRight( aBottomRight ); }

    vec2 UIBaseImage_GetBottomRight( __SELF__ )
    {
        auto lV = SELF( UIBaseImage )->BottomRight();

        return vec2{ lV.x, lV.y };
    }

    void UIBaseImage_SetTintColor( __SELF__, vec4 aColor ) { SELF( UIBaseImage )->SetTintColor( aColor ); }

    vec4 UIBaseImage_GetTintColor( __SELF__ )
    {
        auto lV = SELF( UIBaseImage )->TintColor();

        return vec4{ lV.x, lV.y, lV.z, lV.w };
    }

    BEGIN_INTERFACE_DEFINITION( name )
    CONSTRUCT_WITHOUT_PARAMETERS( UIButton )

    void *UIButton_CreateWithText( void *aText )
    {
        auto lString    = DotNetRuntime::NewString( CAST( MonoString, aText ) );
        auto lNewButton = new UIButton( lString );

        return CAST( void, lNewButton );
    }

    DESTROY_INTERFACE( UIButton )

    void UIButton_SetText( __SELF__, void *aText )
    {
        auto lString = DotNetRuntime::NewString( CAST( MonoString, aText ) );

        SELF( UIButton )->SetText( lString );
    }

    void UIButton_OnClick( __SELF__, void *aDelegate )
    {
        auto lInstance = static_cast<UIButton *>( aSelf );
        auto lDelegate = CAST( MonoObject, aDelegate );

        if( lInstance->mOnClickDelegate != nullptr ) mono_gchandle_free( lInstance->mOnClickDelegateHandle );

        lInstance->mOnClickDelegate       = aDelegate;
        lInstance->mOnClickDelegateHandle = mono_gchandle_new( CAST( MonoObject, aDelegate ), true );

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

    DESTROY_INTERFACE( UICheckBox )

    void UICheckBox_OnClick( __SELF__, void *aDelegate )
    {
        auto lInstance = static_cast<UICheckBox *>( aSelf );
        auto lDelegate = CAST( MonoObject, aDelegate );

        if( lInstance->mOnChangeDelegate != nullptr ) mono_gchandle_free( lInstance->mOnChangeDelegateHandle );

        lInstance->mOnChangeDelegate       = aDelegate;
        lInstance->mOnChangeDelegateHandle = mono_gchandle_new( CAST( MonoObject, aDelegate ), true );

        lInstance->OnClick(
            [lInstance, lDelegate]()
            {
                auto lDelegateClass = mono_object_get_class( lDelegate );
                auto lInvokeMethod  = mono_get_delegate_invoke( lDelegateClass );

                mono_runtime_invoke( lInvokeMethod, lDelegate, nullptr, nullptr );
            } );
    }

    bool UICheckBox_IsChecked( __SELF__ ) { return SELF( UICheckBox )->IsChecked(); }

    void UICheckBox_SetIsChecked( __SELF__, bool aValue ) { SELF( UICheckBox )->SetIsChecked( aValue ); }

    BEGIN_INTERFACE_DEFINITION( name )
    CONSTRUCT_WITHOUT_PARAMETERS( UIColorButton )
    DESTROY_INTERFACE( UIColorButton )

    BEGIN_INTERFACE_DEFINITION( name )
    CONSTRUCT_WITHOUT_PARAMETERS( UIComboBox )
    DESTROY_INTERFACE( UIComboBox )

    void *UIComboBox_CreateWithItems( void *aItems )
    {
        std::vector<std::string> lItemVector;
        for( auto const &x : DotNetRuntime::AsVector<MonoString *>( sCAST( MonoObject, aItems ) ) )
            lItemVector.emplace_back( DotNetRuntime::NewString( x ) );

        auto lNewComboBox = new UIComboBox( lItemVector );

        return CAST( void, lNewComboBox );
    }

    int UIComboBox_GetCurrent( __SELF__ )
    {
        auto lInstance = static_cast<UIComboBox *>( aSelf );

        return lInstance->Current();
    }

    void UIComboBox_SetCurrent( __SELF__, int aValue ) { SELF( UIComboBox )->SetCurrent( aValue ); }

    void UIComboBox_SetItemList( __SELF__, void *aItems )
    {
        std::vector<std::string> lItemVector;
        for( auto const &x : DotNetRuntime::AsVector<MonoString *>( sCAST( MonoObject, aItems ) ) )
            lItemVector.emplace_back( DotNetRuntime::NewString( x ) );

        SELF( UIComboBox )->SetItemList( lItemVector );
    }

    void UIComboBox_OnChanged( __SELF__, void *aDelegate )
    {
        auto lInstance = static_cast<UIComboBox *>( aSelf );
        auto lDelegate = CAST( MonoObject, aDelegate );

        if( lInstance->mOnChangeDelegate != nullptr ) mono_gchandle_free( lInstance->mOnChangeDelegateHandle );

        lInstance->mOnChangeDelegate       = aDelegate;
        lInstance->mOnChangeDelegateHandle = mono_gchandle_new( CAST( MonoObject, aDelegate ), true );

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
    void UIComponent_SetIsVisible( __SELF__, bool aIsVisible ) { SELF( UIComponent )->mIsVisible = aIsVisible; }

    void UIComponent_SetIsEnabled( __SELF__, bool aIsEnabled ) { SELF( UIComponent )->mIsEnabled = aIsEnabled; }

    void UIComponent_SetAllowDragDrop( __SELF__, bool aAllowDragDrop ) { SELF( UIComponent )->mAllowDragDrop = aAllowDragDrop; }

    void UIComponent_SetPaddingAll( __SELF__, float aPaddingAll ) { SELF( UIComponent )->SetPadding( aPaddingAll ); }

    void UIComponent_SetPaddingPairs( __SELF__, float aPaddingTopBottom, float aPaddingLeftRight )
    {
        SELF( UIComponent )->SetPadding( aPaddingTopBottom, aPaddingLeftRight );
    }

    void UIComponent_SetPaddingIndividual( __SELF__, float aPaddingTop, float aPaddingBottom, float aPaddingLeft, float aPaddingRight )

    {
        SELF( UIComponent )->SetPadding( aPaddingTop, aPaddingBottom, aPaddingLeft, aPaddingRight );
    }

    void UIComponent_SetAlignment( __SELF__, eHorizontalAlignment aHAlignment, eVerticalAlignment aVAlignment )
    {

        SELF( UIComponent )->SetAlignment( aHAlignment, aVAlignment );
    }

    void UIComponent_SetHorizontalAlignment( __SELF__, eHorizontalAlignment aAlignment )
    {
        SELF( UIComponent )->SetHorizontalAlignment( aAlignment );
    }

    void UIComponent_SetVerticalAlignment( __SELF__, eVerticalAlignment aAlignment )
    {
        SELF( UIComponent )->SetVerticalAlignment( aAlignment );
    }

    void UIComponent_SetBackgroundColor( __SELF__, vec4 aColor ) { SELF( UIComponent )->SetBackgroundColor( aColor ); }

    void UIComponent_SetFont( __SELF__, FontFamilyFlags aFont ) { SELF( UIComponent )->SetFont( aFont ); }

    void UIComponent_SetTooltip( __SELF__, void *aTooltip )
    {
        auto lTooltip = static_cast<UIComponent *>( aTooltip );

        SELF( UIComponent )->SetTooltip( lTooltip );
    }

    BEGIN_INTERFACE_DEFINITION( name )
    CONSTRUCT_WITHOUT_PARAMETERS( UIDropdownButton )
    DESTROY_INTERFACE( UIDropdownButton )

    void UIDropdownButton_SetContent( __SELF__, void *aContent )
    {
        auto lContent = static_cast<UIComponent *>( aContent );

        return SELF( UIDropdownButton )->SetContent( lContent );
    }

    void UIDropdownButton_SetContentSize( __SELF__, vec2 aContentSizse )
    {
        return SELF( UIDropdownButton )->SetContentSize( aContentSizse );
    }

    void UIDropdownButton_SetImage( __SELF__, void *aImage )
    {
        auto lImage = static_cast<UIBaseImage *>( aImage );

        SELF( UIDropdownButton )->SetImage( lImage );
    }

    void UIDropdownButton_SetText( __SELF__, void *aText )
    {
        auto lString = DotNetRuntime::NewString( CAST( MonoString, aText ) );

        SELF( UIDropdownButton )->SetText( lString );
    }

    void UIDropdownButton_SetTextColor( __SELF__, vec4 aColor ) { SELF( UIDropdownButton )->SetTextColor( aColor ); }

    BEGIN_INTERFACE_DEFINITION( name )
    CONSTRUCT_WITHOUT_PARAMETERS( UIImage )
    DESTROY_INTERFACE( UIImage )

    void *UIImage_CreateWithPath( void *aText, vec2 aSize )
    {
        auto lString   = DotNetRuntime::NewString( CAST( MonoString, aText ) );
        auto lNewImage = new UIImage( lString, aSize );

        return CAST( void, lNewImage );
    }

    BEGIN_INTERFACE_DEFINITION( name )
    CONSTRUCT_WITHOUT_PARAMETERS( UIImageButton )
    DESTROY_INTERFACE( UIImageButton )

    void *UIImageButton_CreateWithPath( void *aText, vec2 *aSize )
    {
        auto lString   = DotNetRuntime::NewString( CAST( MonoString, aText ) );
        auto lNewImage = new UIImageButton( lString, *aSize );

        return CAST( void, lNewImage );
    }

    void UIImageButton_OnClick( __SELF__, void *aDelegate )
    {
        auto lInstance = static_cast<UIImageButton *>( aSelf );
        auto lDelegate = CAST( MonoObject, aDelegate );

        if( lInstance->mOnClickDelegate != nullptr ) mono_gchandle_free( lInstance->mOnClickDelegateHandle );

        lInstance->mOnClickDelegate       = aDelegate;
        lInstance->mOnClickDelegateHandle = mono_gchandle_new( CAST( MonoObject, aDelegate ), true );

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
    DESTROY_INTERFACE( UIImageToggleButton )

    bool UIImageToggleButton_IsActive( __SELF__ ) { return SELF( UIImageToggleButton )->IsActive(); }

    void UIImageToggleButton_SetActive( __SELF__, bool aValue ) { SELF( UIImageToggleButton )->SetActive( aValue ); }

    void UIImageToggleButton_SetActiveImage( __SELF__, void *aImage )
    {
        auto lImage = static_cast<UIBaseImage *>( aImage );

        SELF( UIImageToggleButton )->SetActiveImage( lImage );
    }

    void UIImageToggleButton_SetInactiveImage( __SELF__, void *aImage )
    {
        auto lImage = static_cast<UIBaseImage *>( aImage );

        SELF( UIImageToggleButton )->SetInactiveImage( lImage );
    }

    void UIImageToggleButton_OnClicked( __SELF__, void *aDelegate )
    {
        auto lInstance = static_cast<UIImageToggleButton *>( aSelf );
        auto lDelegate = CAST( MonoObject, aDelegate );

        if( lInstance->mOnClickDelegate != nullptr ) mono_gchandle_free( lInstance->mOnClickDelegateHandle );

        lInstance->mOnClickDelegate       = aDelegate;
        lInstance->mOnClickDelegateHandle = mono_gchandle_new( CAST( MonoObject, aDelegate ), true );

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

    void UIImageToggleButton_OnChanged( __SELF__, void *aDelegate )
    {
        auto lInstance = static_cast<UIImageToggleButton *>( aSelf );
        auto lDelegate = CAST( MonoObject, aDelegate );

        if( lInstance->mOnChangeDelegate != nullptr ) mono_gchandle_free( lInstance->mOnChangeDelegateHandle );

        lInstance->mOnChangeDelegate       = aDelegate;
        lInstance->mOnChangeDelegateHandle = mono_gchandle_new( CAST( MonoObject, aDelegate ), true );

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
    DESTROY_INTERFACE( UILabel )

    void *UILabel_CreateWithText( void *aText )
    {
        auto lString   = DotNetRuntime::NewString( CAST( MonoString, aText ) );
        auto lNewLabel = new UILabel( lString );

        return CAST( void, lNewLabel );
    }

    void UILabel_SetText( __SELF__, void *aText )
    {
        auto lString = DotNetRuntime::NewString( CAST( MonoString, aText ) );

        SELF( UILabel )->SetText( lString );
    }

    void UILabel_SetTextColor( __SELF__, vec4 aTextColor ) { SELF( UILabel )->SetTextColor( aTextColor ); }

    BEGIN_INTERFACE_DEFINITION( name )
    CONSTRUCT_WITHOUT_PARAMETERS( UIMenuItem )
    DESTROY_INTERFACE( UIMenuItem )

    void *UIMenuItem_CreateWithText( void *aText )
    {
        auto lString   = DotNetRuntime::NewString( CAST( MonoString, aText ) );
        auto lNewLabel = new UIMenuItem( lString );

        return CAST( void, lNewLabel );
    }

    void *UIMenuItem_CreateWithTextAndShortcut( void *aText, void *aShortcut )
    {
        auto lString   = DotNetRuntime::NewString( CAST( MonoString, aText ) );
        auto lShortcut = DotNetRuntime::NewString( CAST( MonoString, aShortcut ) );
        auto lNewLabel = new UIMenuItem( lString, lShortcut );

        return CAST( void, lNewLabel );
    }

    void UIMenuItem_SetText( __SELF__, void *aText )
    {
        auto lString = DotNetRuntime::NewString( CAST( MonoString, aText ) );

        SELF( UIMenuItem )->SetText( lString );
    }

    void UIMenuItem_SetShortcut( __SELF__, void *aShortcut )
    {
        auto lShortcut = DotNetRuntime::NewString( CAST( MonoString, aShortcut ) );

        SELF( UIMenuItem )->SetShortcut( lShortcut );
    }

    void UIMenuItem_SetTextColor( __SELF__, vec4 *aTextColor ) { SELF( UIMenuItem )->SetTextColor( *aTextColor ); }

    void UIMenuItem_OnTrigger( __SELF__, void *aDelegate )
    {
        auto lInstance = static_cast<UIMenuItem *>( aSelf );
        auto lDelegate = CAST( MonoObject, aDelegate );

        if( lInstance->mOnTriggerDelegate != nullptr ) mono_gchandle_free( lInstance->mOnTriggerDelegateHandle );

        lInstance->mOnTriggerDelegate       = aDelegate;
        lInstance->mOnTriggerDelegateHandle = mono_gchandle_new( CAST( MonoObject, aDelegate ), true );

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
    DESTROY_INTERFACE( UIMenuSeparator )

    BEGIN_INTERFACE_DEFINITION( name )
    CONSTRUCT_WITHOUT_PARAMETERS( UIMenu )
    DESTROY_INTERFACE( UIMenu )

    void *UIMenu_CreateWithText( void *aText )
    {
        auto lString   = DotNetRuntime::NewString( CAST( MonoString, aText ) );
        auto lNewLabel = new UIMenu( lString );

        return CAST( void, lNewLabel );
    }

    void *UIMenu_AddAction( __SELF__, void *aText, void *aShortcut )
    {
        auto lString    = DotNetRuntime::NewString( CAST( MonoString, aText ) );
        auto lShortcut  = DotNetRuntime::NewString( CAST( MonoString, aShortcut ) );
        auto lNewAction = SELF( UIMenu )->AddActionRaw( lString, lShortcut );

        return CAST( void, lNewAction );
    }

    void *UIMenu_AddMenu( __SELF__, void *aText )
    {
        auto lString  = DotNetRuntime::NewString( CAST( MonoString, aText ) );
        auto lNewMenu = SELF( UIMenu )->AddMenuRaw( lString );

        return CAST( void, lNewMenu );
    }

    void *UIMenu_AddSeparator( __SELF__ )
    {
        auto lNewSeparator = SELF( UIMenu )->AddSeparatorRaw();

        return CAST( void, lNewSeparator );
    }

    void UIMenu_Update( __SELF__ )
    {
        auto lInstance = static_cast<UIMenu *>( aSelf );

        lInstance->Update();
    }

    BEGIN_INTERFACE_DEFINITION( name )
    CONSTRUCT_WITHOUT_PARAMETERS( UIPlot )
    DESTROY_INTERFACE( UIPlot )

    void UIPlot_Clear( __SELF__ ) { SELF( UIPlot )->Clear(); }

    void UIPlot_ConfigureLegend( __SELF__, vec2 *aLegendPadding, vec2 *aLegendInnerPadding, vec2 *aLegendSpacing )
    {
        auto lSelf = static_cast<UIPlot *>( aSelf );

        SELF( UIPlot )->ConfigureLegend( *aLegendPadding, *aLegendInnerPadding, *aLegendSpacing );
    }

    void UIPlot_Add( __SELF__, void *aPlot )
    {
        auto lSelf = static_cast<UIPlot *>( aSelf );
        auto lPlot = static_cast<sPlotData *>( aPlot );

        SELF( UIPlot )->Add( lPlot );
    }

    void UIPlot_SetAxisLimits( __SELF__, int aAxis, double aMin, double aMax )
    {
        auto lSelf = SELF( UIPlot );

        lSelf->mAxisConfiguration[aAxis].mSetLimitRequest = true;
        lSelf->mAxisConfiguration[aAxis].mMin             = static_cast<float>( aMin );
        lSelf->mAxisConfiguration[aAxis].mMax             = static_cast<float>( aMax );
    }

    void UIPlot_SetAxisTitle( __SELF__, int aAxis, void *aTitle )
    {
        SELF( UIPlot )->mAxisConfiguration[aAxis].mTitle = DotNetRuntime::NewString( static_cast<MonoString *>( aTitle ) );
    }

    void *UIPlot_GetAxisTitle( __SELF__, int aAxis )
    {
        return DotNetRuntime::NewString( SELF( UIPlot )->mAxisConfiguration[aAxis].mTitle );
    }

    BEGIN_INTERFACE_DEFINITION( name )
    void UIPlotData_SetLegend( __SELF__, void *aText )
    {
        auto lString = DotNetRuntime::NewString( CAST( MonoString, aText ) );

        SELF( sPlotData )->mLegend = lString;
    }

    void UIPlotData_SetThickness( __SELF__, float aThickness ) { SELF( sPlotData )->mThickness = aThickness; }

    void UIPlotData_SetColor( __SELF__, vec4 aColor ) { SELF( sPlotData )->mColor = aColor; }

    void UIPlotData_SetXAxis( __SELF__, int aAxis ) { SELF( sPlotData )->mXAxis = static_cast<UIPlotAxis>( aAxis ); }

    void UIPlotData_SetYAxis( __SELF__, int aAxis ) { SELF( sPlotData )->mYAxis = static_cast<UIPlotAxis>( aAxis ); }

    BEGIN_INTERFACE_DEFINITION( name )
    void *UIFloat64LinePlot_Create()
    {
        auto lSelf = new sFloat64LinePlot();

        return static_cast<sFloat64LinePlot *>( lSelf );
    }

    void UIFloat64LinePlot_Destroy( __SELF__ ) { delete static_cast<sFloat64LinePlot *>( aSelf ); }

    void UIFloat64LinePlot_SetX( __SELF__, void *aValue )
    {
        SELF( sFloat64LinePlot )->mX = DotNetRuntime::AsVector<double>( static_cast<MonoObject *>( aValue ) );
    }

    void UIFloat64LinePlot_SetY( __SELF__, void *aValue )
    {
        SELF( sFloat64LinePlot )->mY = DotNetRuntime::AsVector<double>( static_cast<MonoObject *>( aValue ) );
    }

    void *UIFloat64ScatterPlot_Create()
    {
        auto lSelf = new sFloat64ScatterPlot();

        return static_cast<sFloat64ScatterPlot *>( lSelf );
    }

    void UIFloat64ScatterPlot_Destroy( __SELF__ ) { delete static_cast<sFloat64ScatterPlot *>( aSelf ); }

    void UIFloat64ScatterPlot_SetX( __SELF__, void *aValue )
    {
        SELF( sFloat64ScatterPlot )->mX = DotNetRuntime::AsVector<double>( static_cast<MonoObject *>( aValue ) );
    }

    void UIFloat64ScatterPlot_SetY( __SELF__, void *aValue )
    {
        SELF( sFloat64ScatterPlot )->mY = DotNetRuntime::AsVector<double>( static_cast<MonoObject *>( aValue ) );
    }

    BEGIN_INTERFACE_DEFINITION( name )
    void *UIVLinePlot_Create()
    {
        auto lSelf = new sVLine();

        return static_cast<sVLine *>( lSelf );
    }

    void UIVLinePlot_Destroy( __SELF__ ) { delete static_cast<sVLine *>( aSelf ); }

    void UIVLinePlot_SetX( __SELF__, void *aValue )
    {
        SELF( sVLine )->mX = DotNetRuntime::AsVector<double>( static_cast<MonoObject *>( aValue ) );
    }

    BEGIN_INTERFACE_DEFINITION( name )
    void *UIHLinePlot_Create()
    {
        auto lSelf = new sHLine();

        return CAST( void, lSelf );
    }

    void UIHLinePlot_Destroy( __SELF__ ) { delete static_cast<sHLine *>( aSelf ); }

    void UIHLinePlot_SetY( __SELF__, void *aValue )
    {
        SELF( sHLine )->mY = DotNetRuntime::AsVector<double>( static_cast<MonoObject *>( aValue ) );
    }

    BEGIN_INTERFACE_DEFINITION( name )
    void *UIAxisTag_Create()
    {
        auto lSelf = new sAxisTag();

        return CAST( void, lSelf );
    }

    void *UIAxisTag_CreateWithTextAndColor( UIPlotAxis aAxis, double aX, void *aText, vec4 aColor )
    {
        auto lString = DotNetRuntime::NewString( CAST( MonoString, aText ) );

        auto lSelf = new sAxisTag( aAxis, aX, lString, aColor );

        return CAST( void, lSelf );
    }

    void UIAxisTag_Destroy( __SELF__ ) { delete static_cast<sAxisTag *>( aSelf ); }

    void UIAxisTag_SetX( __SELF__, double aValue ) { SELF( sAxisTag )->mX = aValue; }

    void UIAxisTag_SetText( __SELF__, void *aText )
    {
        auto lString = DotNetRuntime::NewString( CAST( MonoString, aText ) );

        SELF( sAxisTag )->mText = lString;
    }

    void UIAxisTag_SetColor( __SELF__, vec4 aColor ) { SELF( sAxisTag )->mColor = aColor; }

    vec4 UIAxisTag_GetColor( __SELF__ ) { return SELF( sAxisTag )->mColor; }

    void UIAxisTag_SetAxis( __SELF__, int aAxis ) { SELF( sAxisTag )->mAxis = static_cast<UIPlotAxis>( aAxis ); }

    int UIAxisTag_GetAxis( __SELF__ ) { return static_cast<int>( SELF( sAxisTag )->mXAxis ); }

    BEGIN_INTERFACE_DEFINITION( name )
    void *UIVRangePlot_Create()
    {
        auto lSelf = new sVRange();

        return CAST( void, lSelf );
    }

    void UIVRangePlot_Destroy( __SELF__ ) { delete static_cast<sVRange *>( aSelf ); }

    void UIVRangePlot_SetMin( __SELF__, double aValue ) { SELF( sVRange )->mX0 = aValue; }

    double UIVRangePlot_GetMin( __SELF__ ) { return (double)SELF( sVRange )->mX0; }

    void UIVRangePlot_SetMax( __SELF__, double aValue ) { SELF( sVRange )->mX1 = aValue; }

    double UIVRangePlot_GetMax( __SELF__ ) { return (double)SELF( sVRange )->mX1; }

    BEGIN_INTERFACE_DEFINITION( name )
    void *UIHRangePlot_Create()
    {
        auto lSelf = new sHRange();

        return static_cast<sHRange *>( lSelf );
    }

    void UIHRangePlot_Destroy( __SELF__ ) { delete static_cast<sHRange *>( aSelf ); }

    void UIHRangePlot_SetMin( __SELF__, double aValue ) { SELF( sHRange )->mY0 = aValue; }

    double UIHRangePlot_GetMin( __SELF__ ) { return (double)SELF( sHRange )->mY0; }

    void UIHRangePlot_SetMax( __SELF__, double aValue ) { SELF( sHRange )->mY1 = aValue; }

    double UIHRangePlot_GetMax( __SELF__ ) { return (double)SELF( sHRange )->mY1; }

    BEGIN_INTERFACE_DEFINITION( name )
    CONSTRUCT_WITHOUT_PARAMETERS( UIProgressBar )
    DESTROY_INTERFACE( UIProgressBar )

    void UIProgressBar_SetProgressValue( __SELF__, float aValue ) { SELF( UIProgressBar )->SetProgressValue( aValue ); }

    void UIProgressBar_SetProgressColor( __SELF__, vec4 aTextColor ) { SELF( UIProgressBar )->SetProgressColor( aTextColor ); }

    void UIProgressBar_SetText( __SELF__, void *aText )
    {
        auto lString = DotNetRuntime::NewString( CAST( MonoString, aText ) );

        SELF( UIProgressBar )->SetText( lString );
    }

    void UIProgressBar_SetTextColor( __SELF__, vec4 aTextColor ) { SELF( UIProgressBar )->SetTextColor( aTextColor ); }

    void UIProgressBar_SetThickness( __SELF__, float aValue ) { SELF( UIProgressBar )->SetThickness( aValue ); }

    BEGIN_INTERFACE_DEFINITION( name )
    CONSTRUCT_WITHOUT_PARAMETERS( UIPropertyValue )
    DESTROY_INTERFACE( UIPropertyValue )

    void *UIPropertyValue_CreateWithText( void *aText )
    {
        auto lString   = DotNetRuntime::NewString( CAST( MonoString, aText ) );
        auto lNewLabel = new UIPropertyValue( lString );

        return CAST( void, lNewLabel );
    }

    void *UIPropertyValue_CreateWithTextAndOrientation( void *aText, eBoxLayoutOrientation aOrientation )
    {
        auto lString   = DotNetRuntime::NewString( CAST( MonoString, aText ) );
        auto lNewLabel = new UIPropertyValue( lString, aOrientation );

        return CAST( void, lNewLabel );
    }

    void UIPropertyValue_SetValue( __SELF__, void *aText )
    {
        auto lString = DotNetRuntime::NewString( CAST( MonoString, aText ) );

        SELF( UIPropertyValue )->SetValue( lString );
    }

    void UIPropertyValue_SetValueFont( __SELF__, FontFamilyFlags aFont ) { SELF( UIPropertyValue )->SetValueFont( aFont ); }

    void UIPropertyValue_SetNameFont( __SELF__, FontFamilyFlags aFont ) { SELF( UIPropertyValue )->SetNameFont( aFont ); }

    BEGIN_INTERFACE_DEFINITION( name )
    CONSTRUCT_WITHOUT_PARAMETERS( UISlider )
    DESTROY_INTERFACE( UISlider )

    BEGIN_INTERFACE_DEFINITION( name )
    void UITableColumn_SetTooltip( __SELF__, void *aTooptip )
    {
        auto lSelf = SELF( sTableColumn );

        lSelf->mToolTip.clear();
        for( auto const &x : DotNetRuntime::AsVector<UIComponent *>( static_cast<MonoObject *>( aTooptip ) ) )
            lSelf->mToolTip.push_back( x );
    }

    void UITableColumn_SetForegroundColor( __SELF__, void *aForegroundColor )
    {
        auto lSelf = SELF( sTableColumn );

        lSelf->mForegroundColor.clear();
        for( auto const &x : DotNetRuntime::AsVector<ImVec4>( static_cast<MonoObject *>( aForegroundColor ) ) )
            lSelf->mForegroundColor.push_back( ImColor( x ) );
    }

    void UITableColumn_SetBackgroundColor( __SELF__, void *aBackroundColor )
    {
        auto lSelf = SELF( sTableColumn );

        lSelf->mBackgroundColor.clear();
        for( auto const &x : DotNetRuntime::AsVector<ImVec4>( static_cast<MonoObject *>( aBackroundColor ) ) )
            lSelf->mBackgroundColor.push_back( ImColor( x ) );
    }

    BEGIN_INTERFACE_DEFINITION( name )
    CONSTRUCT_WITHOUT_PARAMETERS( UITable )
    DESTROY_INTERFACE( UITable )

    void UITable_OnRowClicked( __SELF__, void *aDelegate )
    {
        auto lInstance = SELF( UITable );
        auto lDelegate = CAST( MonoObject, aDelegate );

        if( lInstance->mOnRowClickDelegate != nullptr ) mono_gchandle_free( lInstance->mOnRowClickDelegateHandle );

        lInstance->mOnRowClickDelegate       = aDelegate;
        lInstance->mOnRowClickDelegateHandle = mono_gchandle_new( CAST( MonoObject, aDelegate ), true );

        lInstance->OnRowClicked(
            [lInstance, lDelegate]( int aValue )
            {
                auto lDelegateClass = mono_object_get_class( lDelegate );
                auto lInvokeMethod  = mono_get_delegate_invoke( lDelegateClass );

                void *lParams[] = { (void *)&aValue };
                auto  lValue    = mono_runtime_invoke( lInvokeMethod, lDelegate, lParams, nullptr );
            } );
    }

    void UITable_AddColumn( __SELF__, void *aColumn )
    {
        auto lColumn = static_cast<sTableColumn *>( aColumn );

        SELF( UITable )->AddColumn( lColumn );
    }

    void UITable_SetRowHeight( __SELF__, float aRowHeight ) { SELF( UITable )->SetRowHeight( aRowHeight ); }

    void UITable_ClearRowBackgroundColor( __SELF__ ) { SELF( UITable )->mRowBackgroundColor.clear(); }

    void UITable_SetRowBackgroundColor( __SELF__, void *aValue )
    {
        auto lSelf = SELF( UITable );

        lSelf->mRowBackgroundColor.clear();
        for( auto &x : DotNetRuntime::AsVector<ImVec4>( static_cast<MonoObject *>( aValue ) ) )
            lSelf->mRowBackgroundColor.push_back( ImColor( x ) );
    }

    void UITable_SetDisplayedRowIndices( __SELF__, void *aValue )
    {
        auto lSelf = SELF( UITable );
        if( aValue == nullptr )
            lSelf->mDisplayedRowIndices.reset();
        else
            lSelf->mDisplayedRowIndices = DotNetRuntime::AsVector<int>( static_cast<MonoObject *>( aValue ) );
    }

    BEGIN_INTERFACE_DEFINITION( name )
    void *UIFloat64Column_Create()
    {
        auto lNewColumn = new sFloat64Column();

        return CAST( void, lNewColumn );
    }

    void *UIFloat64Column_CreateFull( void *aHeader, float aInitialSize, void *aFormat, void *aNaNFormat )
    {
        auto lHeader    = DotNetRuntime::NewString( static_cast<MonoString *>( aHeader ) );
        auto lFormat    = DotNetRuntime::NewString( static_cast<MonoString *>( aFormat ) );
        auto lNaNFormat = DotNetRuntime::NewString( static_cast<MonoString *>( aNaNFormat ) );
        auto lNewColumn = new sFloat64Column( lHeader, aInitialSize, lFormat, lNaNFormat );

        return CAST( void, lNewColumn );
    }

    void UIFloat64Column_Destroy( __SELF__ ) { delete SELF( sFloat64Column ); }

    void UIFloat64Column_Clear( __SELF__ ) { SELF( sFloat64Column )->Clear(); }

    void UIFloat64Column_SetData( __SELF__, void *aValue )
    {
        SELF( sFloat64Column )->mData = DotNetRuntime::AsVector<double>( static_cast<MonoObject *>( aValue ) );
    }

    BEGIN_INTERFACE_DEFINITION( name )
    void *UIUint32Column_Create()
    {
        auto lNewColumn = new sUint32Column();

        return CAST( void, lNewColumn );
    }

    void *UIUint32Column_CreateFull( void *aHeader, float aInitialSize )
    {
        auto lHeader    = DotNetRuntime::NewString( static_cast<MonoString *>( aHeader ) );
        auto lNewColumn = new sUint32Column( lHeader, aInitialSize );

        return CAST( void, lNewColumn );
    }

    void UIUint32Column_Destroy( __SELF__ ) { delete SELF( sUint32Column ); }

    void UIUint32Column_Clear( __SELF__ ) { SELF( sUint32Column )->Clear(); }

    void UIUint32Column_SetData( __SELF__, void *aValue )
    {
        SELF( sUint32Column )->mData = DotNetRuntime::AsVector<uint32_t>( static_cast<MonoObject *>( aValue ) );
    }

    BEGIN_INTERFACE_DEFINITION( name )
    void *UIStringColumn_Create()
    {
        auto lNewColumn = new sStringColumn();

        return CAST( void, lNewColumn );
    }

    void *UIStringColumn_CreateFull( void *aHeader, float aInitialSize )
    {
        auto lHeader    = DotNetRuntime::NewString( static_cast<MonoString *>( aHeader ) );
        auto lNewColumn = new sStringColumn( lHeader, aInitialSize );

        return CAST( void, lNewColumn );
    }

    void UIStringColumn_Destroy( __SELF__ ) { delete SELF( sStringColumn ); }

    void UIStringColumn_Clear( __SELF__ ) { SELF( sStringColumn )->Clear(); }

    void UIStringColumn_SetData( __SELF__, void *aValue )
    {
        auto lSelf = SELF( sStringColumn );

        lSelf->mData.clear();
        for( auto const &x : DotNetRuntime::AsVector<MonoString *>( static_cast<MonoObject *>( aValue ) ) )
            lSelf->mData.push_back( DotNetRuntime::NewString( x ) );
    }

    BEGIN_INTERFACE_DEFINITION( name )
    CONSTRUCT_WITHOUT_PARAMETERS( UITextInput )
    DESTROY_INTERFACE( UITextInput )

    void *UITextInput_CreateWithText( void *aText )
    {
        auto lString       = DotNetRuntime::NewString( CAST( MonoString, aText ) );
        auto lNewTextInput = new UITextInput( lString );

        return CAST( void, lNewTextInput );
    }

    void UITextInput_SetHintText( __SELF__, void *aText )
    {
        auto lString = DotNetRuntime::NewString( CAST( MonoString, aText ) );

        SELF( UITextInput )->SetHintText( lString );
    }

    void *UITextInput_GetText( __SELF__ ) { return DotNetRuntime::NewString( SELF( UITextInput )->GetText() ); }

    void UITextInput_SetTextColor( __SELF__, vec4 *aTextColor ) { SELF( UITextInput )->SetTextColor( *aTextColor ); }

    void UITextInput_SetBufferSize( __SELF__, uint32_t aBufferSize ) { SELF( UITextInput )->SetBuffersize( aBufferSize ); }

    void UITextInput_OnTextChanged( __SELF__, void *aDelegate )
    {
        auto lInstance = SELF( UITextInput );
        auto lDelegate = CAST( MonoObject, aDelegate );

        if( lInstance->mOnTextChangedDelegate != nullptr ) mono_gchandle_free( lInstance->mOnTextChangedDelegateHandle );

        lInstance->mOnTextChangedDelegate       = aDelegate;
        lInstance->mOnTextChangedDelegateHandle = mono_gchandle_new( CAST( MonoObject, aDelegate ), true );

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
    DESTROY_INTERFACE( UITextOverlay )

    void UITextOverlay_AddText( __SELF__, void *aText )
    {
        auto lString = DotNetRuntime::NewString( CAST( MonoString, aText ) );

        SELF( UITextOverlay )->AddText( lString );
    }

    void UITextOverlay_Clear( __SELF__ ) { SELF( UITextOverlay )->Clear(); }

    BEGIN_INTERFACE_DEFINITION( name )
    CONSTRUCT_WITHOUT_PARAMETERS( UITextToggleButton )
    DESTROY_INTERFACE( UITextToggleButton )

    void *UITextToggleButton_CreateWithText( void *aText )
    {
        auto lString    = DotNetRuntime::NewString( CAST( MonoString, aText ) );
        auto lNewButton = new UITextToggleButton( lString );

        return CAST( void, lNewButton );
    }

    bool UITextToggleButton_IsActive( __SELF__ ) { return SELF( UITextToggleButton )->IsActive(); }

    void UITextToggleButton_SetActive( __SELF__, bool aValue ) { SELF( UITextToggleButton )->SetActive( aValue ); }

    void UITextToggleButton_SetActiveColor( __SELF__, vec4 *aColor ) { SELF( UITextToggleButton )->SetActiveColor( *aColor ); }

    void UITextToggleButton_SetInactiveColor( __SELF__, vec4 *aColor ) { SELF( UITextToggleButton )->SetInactiveColor( *aColor ); }

    void UITextToggleButton_OnClicked( __SELF__, void *aDelegate )
    {
        auto lInstance = SELF( UITextToggleButton );
        auto lDelegate = CAST( MonoObject, aDelegate );

        if( lInstance->mOnClickDelegate != nullptr ) mono_gchandle_free( lInstance->mOnClickDelegateHandle );

        lInstance->mOnClickDelegate       = aDelegate;
        lInstance->mOnClickDelegateHandle = mono_gchandle_new( CAST( MonoObject, aDelegate ), true );

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

    void UITextToggleButton_OnChanged( __SELF__, void *aDelegate )
    {
        auto lInstance = SELF( UITextToggleButton );
        auto lDelegate = CAST( MonoObject, aDelegate );

        if( lInstance->mOnChangeDelegate != nullptr ) mono_gchandle_free( lInstance->mOnChangeDelegateHandle );

        lInstance->mOnChangeDelegate       = aDelegate;
        lInstance->mOnChangeDelegateHandle = mono_gchandle_new( CAST( MonoObject, aDelegate ), true );

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
    DESTROY_INTERFACE( UITreeViewNode )

    void UITreeViewNode_SetText( __SELF__, void *aText )
    {
        auto lString = DotNetRuntime::NewString( CAST( MonoString, aText ) );

        SELF( UITreeViewNode )->SetText( lString );
    }

    void UITreeViewNode_SetTextColor( __SELF__, vec4 aTextColor ) { SELF( UITreeViewNode )->SetTextColor( aTextColor ); }

    void UITreeViewNode_SetIcon( __SELF__, void *aIcon )
    {
        auto lImage = static_cast<UIImage *>( aIcon );

        SELF( UITreeViewNode )->SetIcon( lImage );
    }

    void UITreeViewNode_SetIndicator( __SELF__, void *aIndicator )
    {
        auto lImage = static_cast<UIComponent *>( aIndicator );

        SELF( UITreeViewNode )->SetIndicator( lImage );
    }

    void *UITreeViewNode_Add( __SELF__ ) { return CAST( void, SELF( UITreeViewNode )->Add() ); }

    BEGIN_INTERFACE_DEFINITION( name )
    CONSTRUCT_WITHOUT_PARAMETERS( UITreeView )
    DESTROY_INTERFACE( UITreeView )

    void UITreeView_SetIndent( __SELF__, float aIndent ) { SELF( UITreeView )->SetIndent( aIndent ); }

    void UITreeView_SetIconSpacing( __SELF__, float aSpacing ) { SELF( UITreeView )->SetIconSpacing( aSpacing ); }

    void *UITreeView_Add( __SELF__ ) { return CAST( void, SELF( UITreeView )->Add() ); }

    BEGIN_INTERFACE_DEFINITION( name )
    CONSTRUCT_WITHOUT_PARAMETERS( UIVec2Input )
    DESTROY_INTERFACE( UIVec2Input )

    void UIVec2Input_OnChanged( __SELF__, void *aDelegate )
    {
        auto lInstance = static_cast<UIVectorInputBase *>( aSelf );
        auto lDelegate = CAST( MonoObject, aDelegate );

        if( lInstance->mOnChangeDelegate != nullptr ) mono_gchandle_free( lInstance->mOnChangeDelegateHandle );

        lInstance->mOnChangeDelegate       = aDelegate;
        lInstance->mOnChangeDelegateHandle = mono_gchandle_new( CAST( MonoObject, aDelegate ), true );

        lInstance->OnChanged(
            [lInstance, lDelegate]( vec4 aVector )
            {
                auto lDelegateClass = mono_object_get_class( lDelegate );
                auto lInvokeMethod  = mono_get_delegate_invoke( lDelegateClass );

                vec2  lProjection = vec2{ aVector.x, aVector.y };
                void *lParams[]   = { (void *)&lProjection };
                auto  lValue      = mono_runtime_invoke( lInvokeMethod, lDelegate, lParams, nullptr );
            } );
    }

    void UIVec2Input_SetValue( __SELF__, vec2 aValue ) { SELF( UIVec2Input )->SetValue( aValue ); }

    vec2 UIVec2Input_GetValue( __SELF__ ) { return SELF( UIVec2Input )->Value(); }

    void UIVec2Input_SetResetValues( __SELF__, vec2 aValue ) { SELF( UIVec2Input )->SetResetValues( aValue ); }

    void UIVec2Input_SetFormat( __SELF__, void *aText )
    {
        auto lString = DotNetRuntime::NewString( CAST( MonoString, aText ) );

        SELF( UIVectorInputBase )->SetFormat( lString );
    }

    BEGIN_INTERFACE_DEFINITION( name )
    CONSTRUCT_WITHOUT_PARAMETERS( UIVec3Input )
    DESTROY_INTERFACE( UIVec3Input )

    void UIVec3Input_OnChanged( __SELF__, void *aDelegate )
    {
        auto lInstance = static_cast<UIVectorInputBase *>( aSelf );
        auto lDelegate = CAST( MonoObject, aDelegate );

        if( lInstance->mOnChangeDelegate != nullptr ) mono_gchandle_free( lInstance->mOnChangeDelegateHandle );

        lInstance->mOnChangeDelegate       = aDelegate;
        lInstance->mOnChangeDelegateHandle = mono_gchandle_new( CAST( MonoObject, aDelegate ), true );

        lInstance->OnChanged(
            [lInstance, lDelegate]( vec4 aVector )
            {
                auto lDelegateClass = mono_object_get_class( lDelegate );
                auto lInvokeMethod  = mono_get_delegate_invoke( lDelegateClass );

                vec3  lProjection = vec3{ aVector.x, aVector.y, aVector.z };
                void *lParams[]   = { (void *)&lProjection };
                auto  lValue      = mono_runtime_invoke( lInvokeMethod, lDelegate, lParams, nullptr );
            } );
    }

    void UIVec3Input_SetValue( __SELF__, vec3 aValue ) { SELF( UIVec3Input )->SetValue( aValue ); }

    vec3 UIVec3Input_GetValue( __SELF__ ) { return SELF( UIVec3Input )->Value(); }

    void UIVec3Input_SetResetValues( __SELF__, vec3 aValue ) { SELF( UIVec3Input )->SetResetValues( aValue ); }

    void UIVec3Input_SetFormat( __SELF__, void *aText )
    {
        auto lString = DotNetRuntime::NewString( CAST( MonoString, aText ) );

        SELF( UIVectorInputBase )->SetFormat( lString );
    }

    BEGIN_INTERFACE_DEFINITION( name )
    CONSTRUCT_WITHOUT_PARAMETERS( UIVec4Input )
    DESTROY_INTERFACE( UIVec4Input )

    void UIVec4Input_OnChanged( __SELF__, void *aDelegate )
    {
        auto lInstance = static_cast<UIVectorInputBase *>( aSelf );
        auto lDelegate = CAST( MonoObject, aDelegate );

        if( lInstance->mOnChangeDelegate != nullptr ) mono_gchandle_free( lInstance->mOnChangeDelegateHandle );

        lInstance->mOnChangeDelegate       = aDelegate;
        lInstance->mOnChangeDelegateHandle = mono_gchandle_new( CAST( MonoObject, aDelegate ), true );

        lInstance->OnChanged(
            [lInstance, lDelegate]( vec4 aVector )
            {
                auto lDelegateClass = mono_object_get_class( lDelegate );
                auto lInvokeMethod  = mono_get_delegate_invoke( lDelegateClass );

                void *lParams[] = { (void *)&aVector };
                auto  lValue    = mono_runtime_invoke( lInvokeMethod, lDelegate, lParams, nullptr );
            } );
    }

    void UIVec4Input_SetValue( __SELF__, vec4 aValue ) { SELF( UIVec4Input )->SetValue( aValue ); }

    vec4 UIVec4Input_GetValue( __SELF__ ) { return SELF( UIVec4Input )->Value(); }

    void UIVec4Input_SetResetValues( __SELF__, vec4 aValue ) { SELF( UIVec4Input )->SetResetValues( aValue ); }

    void UIVec4Input_SetFormat( __SELF__, void *aText )
    {
        auto lString = DotNetRuntime::NewString( CAST( MonoString, aText ) );

        SELF( UIVectorInputBase )->SetFormat( lString );
    }

    BEGIN_INTERFACE_DEFINITION( name )
    CONSTRUCT_WITHOUT_PARAMETERS( UIWorkspaceDocument )
    DESTROY_INTERFACE( UIWorkspaceDocument )

    void UIWorkspaceDocument_SetContent( __SELF__, void *aContent )
    {
        auto lContent = static_cast<UIComponent *>( aContent );

        SELF( UIWorkspaceDocument )->SetContent( lContent );
    }

    void UIWorkspaceDocument_Update( __SELF__ ) { SELF( UIWorkspaceDocument )->Update(); }

    void UIWorkspaceDocument_SetName( __SELF__, void *aName )
    {
        auto lName = DotNetRuntime::NewString( static_cast<MonoString *>( aName ) );

        SELF( UIWorkspaceDocument )->mName = lName;
    }

    bool UIWorkspaceDocument_IsDirty( __SELF__ ) { return SELF( UIWorkspaceDocument )->mDirty; }

    void UIWorkspaceDocument_MarkAsDirty( __SELF__, bool aDirty ) { SELF( UIWorkspaceDocument )->mDirty = aDirty; }

    void UIWorkspaceDocument_Open( __SELF__ ) { SELF( UIWorkspaceDocument )->DoOpen(); }

    void UIWorkspaceDocument_RequestClose( __SELF__ ) { SELF( UIWorkspaceDocument )->DoQueueClose(); }

    void UIWorkspaceDocument_ForceClose( __SELF__ ) { SELF( UIWorkspaceDocument )->DoForceClose(); }

    void UIWorkspaceDocument_RegisterSaveDelegate( __SELF__, void *aDelegate )
    {
        auto lInstance = SELF( UIWorkspaceDocument );
        auto lDelegate = CAST( MonoObject, aDelegate );

        if( lInstance->mSaveDelegate != nullptr ) mono_gchandle_free( lInstance->mSaveDelegateHandle );

        lInstance->mSaveDelegate       = aDelegate;
        lInstance->mSaveDelegateHandle = mono_gchandle_new( CAST( MonoObject, aDelegate ), true );

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
    DESTROY_INTERFACE( UIWorkspace )

    void UIWorkspace_Add( __SELF__, void *aDocument )
    {
        auto lDocument = static_cast<UIWorkspaceDocument *>( aDocument );

        SELF( UIWorkspace )->Add( lDocument );
    }

    void UIWorkspace_RegisterCloseDocumentDelegate( __SELF__, void *aDelegate )
    {
        auto lInstance = SELF( UIWorkspace );
        auto lDelegate = CAST( MonoObject, aDelegate );

        if( lInstance->mCloseDocumentDelegate != nullptr ) mono_gchandle_free( lInstance->mCloseDocumentDelegateHandle );

        lInstance->mCloseDocumentDelegate       = aDelegate;
        lInstance->mCloseDocumentDelegateHandle = mono_gchandle_new( CAST( MonoObject, aDelegate ), true );

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

        return CAST( void, lNewLayout );
    }

    DESTROY_INTERFACE( UIBoxLayout )

    void UIBoxLayout_AddAlignedNonFixed( __SELF__, void *aChild, bool aExpand, bool aFill, eHorizontalAlignment aHAlignment,
                                         eVerticalAlignment aVAlignment )
    {
        auto lChild = static_cast<UIComponent *>( aChild );

        SELF( UIBoxLayout )->Add( lChild, aExpand, aFill, aHAlignment, aVAlignment );
    }

    void UIBoxLayout_AddNonAlignedNonFixed( __SELF__, void *aChild, bool aExpand, bool aFill )
    {
        auto lChild = static_cast<UIComponent *>( aChild );

        SELF( UIBoxLayout )->Add( lChild, aExpand, aFill );
    }

    void UIBoxLayout_AddAlignedFixed( __SELF__, void *aChild, float aFixedSize, bool aExpand, bool aFill,
                                      eHorizontalAlignment aHAlignment, eVerticalAlignment aVAlignment )
    {
        auto lChild = static_cast<UIComponent *>( aChild );

        SELF( UIBoxLayout )->Add( lChild, aFixedSize, aExpand, aFill, aHAlignment, aVAlignment );
    }

    void UIBoxLayout_AddNonAlignedFixed( __SELF__, void *aChild, float aFixedSize, bool aExpand, bool aFill )
    {
        auto lChild = static_cast<UIComponent *>( aChild );

        SELF( UIBoxLayout )->Add( lChild, aFixedSize, aExpand, aFill );
    }

    void UIBoxLayout_AddSeparator( __SELF__ ) { SELF( UIBoxLayout )->AddSeparator(); }

    void UIBoxLayout_SetItemSpacing( __SELF__, float aItemSpacing )
    {
        auto lInstance = SELF( UIBoxLayout );

        lInstance->SetItemSpacing( aItemSpacing );
    }

    void UIBoxLayout_Clear( __SELF__ ) { SELF( UIBoxLayout )->Clear(); }

    BEGIN_INTERFACE_DEFINITION( name )
    CONSTRUCT_WITHOUT_PARAMETERS( UIContainer )
    DESTROY_INTERFACE( UIContainer )

    void UIContainer_SetContent( __SELF__, void *aChild )
    {
        auto lChild = static_cast<UIComponent *>( aChild );

        SELF( UIContainer )->SetContent( lChild );
    }

    BEGIN_INTERFACE_DEFINITION( name )
    CONSTRUCT_WITHOUT_PARAMETERS( UISplitter )
    DESTROY_INTERFACE( UISplitter )

    void *UISplitter_CreateWithOrientation( eBoxLayoutOrientation aOrientation )
    {
        auto lNewLayout = new UISplitter( aOrientation );

        return CAST( void, lNewLayout );
    }

    void UISplitter_Add1( __SELF__, void *aChild )
    {
        auto lChild = static_cast<UIComponent *>( aChild );

        SELF( UISplitter )->Add1( lChild );
    }

    void UISplitter_Add2( __SELF__, void *aChild )
    {
        auto lChild = static_cast<UIComponent *>( aChild );

        SELF( UISplitter )->Add2( lChild );
    }

    void UISplitter_SetItemSpacing( __SELF__, float aItemSpacing ) { SELF( UISplitter )->SetItemSpacing( aItemSpacing ); }

    BEGIN_INTERFACE_DEFINITION( name )
    CONSTRUCT_WITHOUT_PARAMETERS( UIStackLayout )
    DESTROY_INTERFACE( UIStackLayout )

    void UIStackLayout_Add( __SELF__, void *aChild, void *aKey )
    {
        auto lChild  = static_cast<UIComponent *>( aChild );
        auto lString = DotNetRuntime::NewString( static_cast<MonoString *>( aKey ) );

        SELF( UIStackLayout )->Add( lChild, lString );
    }

    void UIStackLayout_SetCurrent( __SELF__, void *aKey )
    {
        auto lString = DotNetRuntime::NewString( static_cast<MonoString *>( aKey ) );

        SELF( UIStackLayout )->SetCurrent( lString );
    }

    BEGIN_INTERFACE_DEFINITION( name )
    CONSTRUCT_WITHOUT_PARAMETERS( UIZLayout )
    DESTROY_INTERFACE( UIZLayout )

    void UIZLayout_AddAlignedNonFixed( __SELF__, void *aChild, bool aExpand, bool aFill, eHorizontalAlignment aHAlignment,
                                       eVerticalAlignment aVAlignment )
    {
        auto lChild = static_cast<UIComponent *>( aChild );

        SELF( UIZLayout )->Add( lChild, aExpand, aFill, aHAlignment, aVAlignment );
    }

    void UIZLayout_AddNonAlignedNonFixed( __SELF__, void *aChild, bool aExpand, bool aFill )
    {
        auto lChild = static_cast<UIComponent *>( aChild );

        SELF( UIZLayout )->Add( lChild, aExpand, aFill );
    }

    void UIZLayout_AddAlignedFixed( __SELF__, void *aChild, vec2 aSize, vec2 aPosition, bool aExpand, bool aFill,
                                    eHorizontalAlignment aHAlignment, eVerticalAlignment aVAlignment )
    {
        auto lChild = static_cast<UIComponent *>( aChild );

        SELF( UIZLayout )->Add( lChild, aSize, aPosition, aExpand, aFill, aHAlignment, aVAlignment );
    }

    void UIZLayout_AddNonAlignedFixed( __SELF__, void *aChild, vec2 aSize, vec2 aPosition, bool aExpand, bool aFill )
    {
        auto lChild = static_cast<UIComponent *>( aChild );

        SELF( UIZLayout )->Add( lChild, aSize, aPosition, aExpand, aFill );
    }

    BEGIN_INTERFACE_DEFINITION( name )
    CONSTRUCT_WITHOUT_PARAMETERS( UIFileTree )
    DESTROY_INTERFACE( UIFileTree )

    void *UIFileTree_Add( __SELF__, void *aPath )
    {
        auto lString = DotNetRuntime::NewString( CAST( MonoString, aPath ) );

        return CAST( void, SELF( UIFileTree )->Add( lString ) );
    }

    BEGIN_INTERFACE_DEFINITION( name )
    CONSTRUCT_WITHOUT_PARAMETERS( UIDialog )
    DESTROY_INTERFACE( UIDialog )

    void *UIDialog_CreateWithTitleAndSize( void *aTitle, vec2 *aSize )
    {
        auto lString    = DotNetRuntime::NewString( static_cast<MonoString *>( aTitle ) );
        auto lNewDialog = new UIDialog( lString, *aSize );

        return CAST( void, lNewDialog );
    }

    void UIDialog_SetTitle( __SELF__, void *aTitle )
    {
        auto lString = DotNetRuntime::NewString( static_cast<MonoString *>( aTitle ) );

        SELF( UIDialog )->SetTitle( lString );
    }

    void UIDialog_SetSize( __SELF__, vec2 aSize ) { SELF( UIDialog )->SetSize( aSize ); }

    void UIDialog_SetContent( __SELF__, void *aContent )
    {
        auto lContent = static_cast<UIComponent *>( aContent );

        SELF( UIDialog )->SetContent( lContent );
    }

    void UIDialog_Open( __SELF__ ) { SELF( UIDialog )->Open(); }

    void UIDialog_Close( __SELF__ ) { SELF( UIDialog )->Close(); }

    void UIDialog_Update( __SELF__ ) { SELF( UIDialog )->Update(); }

    BEGIN_INTERFACE_DEFINITION( name )
    CONSTRUCT_WITHOUT_PARAMETERS( UIForm )
    DESTROY_INTERFACE( UIForm )

    void UIForm_SetTitle( __SELF__, void *aTitle )
    {
        auto lString = DotNetRuntime::NewString( static_cast<MonoString *>( aTitle ) );

        SELF( UIForm )->SetTitle( lString );
    }

    void UIForm_SetContent( __SELF__, void *aContent )
    {
        auto lContent = static_cast<UIComponent *>( aContent );

        SELF( UIForm )->SetContent( lContent );
    }

    void UIForm_Update( __SELF__ ) { SELF( UIForm )->Update(); }

    void UIForm_SetSize( __SELF__, float aWidth, float aHeight ) { SELF( UIForm )->SetSize( aWidth, aHeight ); }
    // clang-format on
} // namespace SE::Core::Interop