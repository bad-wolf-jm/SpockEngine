#include "UI/Components/Component.h"
#include "UI/Components/Plot.h"
#include "UI/Layouts/BoxLayout.h"

namespace SE::Core::Interop
{
    using namespace math;

    void UIComponent_SetIsVisible( void *aSelf, bool aIsVisible );
    void UIComponent_SetIsEnabled( void *aSelf, bool aIsEnabled );
    void UIComponent_SetAllowDragDrop( void *aSelf, bool aAllowDragDrop );

    void UIComponent_SetPaddingAll( void *aSelf, float aPaddingAll );
    void UIComponent_SetPaddingPairs( void *aSelf, float aPaddingTopBottom, float aPaddingLeftRight );
    void UIComponent_SetPaddingIndividual( void *aSelf, float aPaddingTop, float aPaddingBottom, float aPaddingLeft,
                                           float aPaddingRight );
    void UIComponent_SetAlignment( void *aSelf, eHorizontalAlignment aHAlignment, eVerticalAlignment aVAlignment );
    void UIComponent_SetHorizontalAlignment( void *aSelf, eHorizontalAlignment aAlignment );
    void UIComponent_SetVerticalAlignment( void *aSelf, eVerticalAlignment aAlignment );
    void UIComponent_SetBackgroundColor( void *aSelf, vec4 aColor );
    void UIComponent_SetFont( void *aSelf, FontFamilyFlags aFont );
    void UIComponent_SetTooltip( void *aSelf, void *aTooltip );

    void *UIBaseImage_Create();
    void *UIBaseImage_CreateWithPath( void *aText, vec2 aSize );
    void  UIBaseImage_Destroy( void *aInstance );
    void  UIBaseImage_SetImage( void *aInstance, void *aPath );

    void UIBaseImage_SetSize( void *aInstance, vec2 aSize );
    vec2 UIBaseImage_GetSize( void *aInstance );
    void UIBaseImage_SetTopLeft( void *aInstance, vec2 aTopLeft );
    vec2 UIBaseImage_GetTopLeft( void *aInstance );
    void UIBaseImage_SetBottomRight( void *aInstance, vec2 aBottomRight );
    vec2 UIBaseImage_GetBottomRight( void *aInstance );
    void UIBaseImage_SetTintColor( void *aInstance, vec4 aColor );
    vec4 UIBaseImage_GetTintColor( void *aInstance );

    void *UIButton_Create();
    void *UIButton_CreateWithText( void *aText );
    void  UIButton_Destroy( void *aInstance );
    void  UIButton_OnClick( void *aInstance, void *aDelegate );
    void  UIButton_SetText( void *aInstance, void *aText );

    void *UICheckBox_Create();
    void  UICheckBox_Destroy( void *aInstance );
    void  UICheckBox_OnClick( void *aInstance, void *aTextColor );
    bool  UICheckBox_IsChecked( void *aInstance );
    void  UICheckBox_SetIsChecked( void *aInstance, bool aValue );

    void *UIColorButton_Create();
    void  UIColorButton_Destroy( void *aInstance );

    void *UIComboBox_Create();
    void *UIComboBox_CreateWithItems( void *aItems );
    void  UIComboBox_Destroy( void *aInstance );
    int   UIComboBox_GetCurrent( void *aInstance );
    void  UIComboBox_SetCurrent( void *aInstance, int aValue );
    void  UIComboBox_SetItemList( void *aInstance, void *aItems );
    void  UIComboBox_OnChanged( void *aInstance, void *aDelegate );

    void *UIDropdownButton_Create();
    void  UIDropdownButton_Destroy( void *aInstance );
    void  UIDropdownButton_SetContent( void *aInstance, void *aContent );
    void  UIDropdownButton_SetContentSize( void *aInstance, vec2 aSize );
    void  UIDropdownButton_SetImage( void *aInstance, void *aImage );
    void  UIDropdownButton_SetText( void *aInstance, void *aText );
    void  UIDropdownButton_SetTextColor( void *aInstance, vec4 aColor );

    void *UIImage_Create();
    void *UIImage_CreateWithPath( void *aText, vec2 aSize );
    void  UIImage_Destroy( void *aInstance );

    void *UIImageButton_Create();
    void *UIImageButton_CreateWithPath( void *aText, vec2 *aSize );
    void  UIImageButton_Destroy( void *aInstance );
    void  UIImageButton_OnClick( void *aInstance, void *aDelegate );

    void *UIImageToggleButton_Create();
    void  UIImageToggleButton_Destroy( void *aInstance );
    void  UIImageToggleButton_OnClicked( void *aInstance, void *aHandler );
    void  UIImageToggleButton_OnChanged( void *aInstance, void *aHandler );
    bool  UIImageToggleButton_IsActive( void *aInstance );
    void  UIImageToggleButton_SetActive( void *aInstance, bool aValue );
    void  UIImageToggleButton_SetActiveImage( void *aInstance, void *aImage );
    void  UIImageToggleButton_SetInactiveImage( void *aInstance, void *aImage );

    void *UILabel_Create();
    void *UILabel_CreateWithText( void *aText );
    void  UILabel_Destroy( void *aInstance );
    void  UILabel_SetText( void *aInstance, void *aText );
    void  UILabel_SetTextColor( void *aInstance, vec4 aTextColor );

    void *UIMenuItem_Create();
    void *UIMenuItem_CreateWithText( void *aText );
    void *UIMenuItem_CreateWithTextAndShortcut( void *aText, void *aShortcut );
    void  UIMenuItem_Destroy( void *aInstance );
    void  UIMenuItem_SetText( void *aInstance, void *aText );
    void  UIMenuItem_SetShortcut( void *aInstance, void *aShortcut );
    void  UIMenuItem_SetTextColor( void *aInstance, vec4 *aTextColor );
    void  UIMenuItem_OnTrigger( void *aInstance, void *aDelegate );

    void *UIMenuSeparator_Create();
    void  UIMenuSeparator_Destroy( void *aInstance );

    void *UIMenu_Create();
    void *UIMenu_CreateWithText( void *aText );
    void  UIMenu_Destroy( void *aInstance );
    void *UIMenu_AddAction( void *aInstance, void *aText, void *aShortcut );
    void *UIMenu_AddMenu( void *aInstance, void *aText );
    void *UIMenu_AddSeparator( void *aInstance );
    void  UIMenu_Update( void *aInstance );

    void UIPlotData_SetThickness( void *aSelf, float aThickness );
    void UIPlotData_SetLegend( void *aSelf, void *aText );
    void UIPlotData_SetColor( void *aSelf, vec4 aColor );
    void UIPlotData_SetXAxis( void *aSelf, int aAxis );
    void UIPlotData_SetYAxis( void *aSelf, int aAxis );

    void *UIFloat64LinePlot_Create();
    void  UIFloat64LinePlot_Destroy( void *aSelf );
    void  UIFloat64LinePlot_SetX( void *aSelf, void *aValue );
    void  UIFloat64LinePlot_SetY( void *aSelf, void *aValue );

    void *UIFloat64ScatterPlot_Create();
    void  UIFloat64ScatterPlot_Destroy( void *aSelf );
    void  UIFloat64ScatterPlot_SetX( void *aSelf, void *aValue );
    void  UIFloat64ScatterPlot_SetY( void *aSelf, void *aValue );

    void *UIVLinePlot_Create();
    void  UIVLinePlot_Destroy( void *aSelf );
    void  UIVLinePlot_SetX( void *aSelf, void *aValue );

    void *UIHLinePlot_Create();
    void  UIHLinePlot_Destroy( void *aSelf );
    void  UIHLinePlot_SetY( void *aSelf, void *aValue );

    void  *UIVRangePlot_Create();
    void   UIVRangePlot_Destroy( void *aSelf );
    double UIVRangePlot_GetMin( void *aSelf );
    void   UIVRangePlot_SetMin( void *aSelf, double aValue );
    double UIVRangePlot_GetMax( void *aSelf );
    void   UIVRangePlot_SetMax( void *aSelf, double aValue );

    void  *UIHRangePlot_Create();
    void   UIHRangePlot_Destroy( void *aSelf );
    double UIHRangePlot_GetMin( void *aSelf );
    void   UIHRangePlot_SetMin( void *aSelf, double aValue );
    double UIHRangePlot_GetMax( void *aSelf );
    void   UIHRangePlot_SetMax( void *aSelf, double aValue );

    void *UIAxisTag_Create();
    void *UIAxisTag_CreateWithTextAndColor( UIPlotAxis aAxis, double aX, void *aText, vec4 aColor );
    void  UIAxisTag_Destroy( void *aSelf );
    void  UIAxisTag_SetX( void *aSelf, double aValue );
    void  UIAxisTag_SetText( void *aSelf, void *aText );
    vec4  UIAxisTag_GetColor( void *aSelf );
    void  UIAxisTag_SetColor( void *aSelf, vec4 aColor );
    int   UIAxisTag_GetAxis( void *aSelf );
    void  UIAxisTag_SetAxis( void *aSelf, int aAxis );

    void *UIPlot_Create();
    void  UIPlot_Destroy( void *aInstance );
    void  UIPlot_Clear( void *aInstance );
    void  UIPlot_ConfigureLegend( void *aInstance, vec2 *aLegendPadding, vec2 *aLegendInnerPadding, vec2 *aLegendSpacing );
    void  UIPlot_Add( void *aInstance, void *aPlot );
    void  UIPlot_SetAxisLimits( void *aInstance, int aAxis, double aMin, double aMax );
    void *UIPlot_GetAxisTitle( void *aInstance, int aAxis );
    void  UIPlot_SetAxisTitle( void *aInstance, int aAxis, void *aTitle );

    void *UIProgressBar_Create();
    void  UIProgressBar_Destroy( void *aInstance );
    void  UIProgressBar_SetProgressValue( void *aInstance, float aValue );
    void  UIProgressBar_SetProgressColor( void *aInstance, vec4 aProgressColor );
    void  UIProgressBar_SetText( void *aInstance, void *aValue );
    void  UIProgressBar_SetTextColor( void *aInstance, vec4 aTextColor );
    void  UIProgressBar_SetThickness( void *aInstance, float aValue );

    void *UIPropertyValue_Create();
    void *UIPropertyValue_CreateWithText( void *aText );
    void *UIPropertyValue_CreateWithTextAndOrientation( void *aText, eBoxLayoutOrientation aOrientation );
    void  UIPropertyValue_Destroy( void *aInstance );
    void  UIPropertyValue_SetValue( void *aInstance, void *aText );
    void  UIPropertyValue_SetValueFont( void *aInstance, FontFamilyFlags aFont );
    void  UIPropertyValue_SetNameFont( void *aInstance, FontFamilyFlags aFont );

    void *UISlider_Create();
    void  UISlider_Destroy( void *aInstance );

    void UITableColumn_SetTooltip( void *aSelf, void *aTooptip );
    void UITableColumn_SetForegroundColor( void *aSelf, void *aForegroundColor );
    void UITableColumn_SetBackgroundColor( void *aSelf, void *aBackroundColor );

    void *UIFloat64Column_Create();
    void *UIFloat64Column_CreateFull( void *aHeader, float aInitialSize, void *aFormat, void *aNaNFormat );
    void  UIFloat64Column_Destroy( void *aSelf );
    void  UIFloat64Column_Clear( void *aSelf );
    void  UIFloat64Column_SetData( void *aSelf, void *aValue );

    void *UIUint32Column_Create();
    void *UIUint32Column_CreateFull( void *aHeader, float aInitialSize );
    void  UIUint32Column_Destroy( void *aSelf );
    void  UIUint32Column_Clear( void *aSelf );
    void  UIUint32Column_SetData( void *aSelf, void *aValue );

    void *UIStringColumn_Create();
    void *UIStringColumn_CreateFull( void *aHeader, float aInitialSize );
    void  UIStringColumn_Destroy( void *aSelf );
    void  UIStringColumn_Clear( void *aSelf );
    void  UIStringColumn_SetData( void *aSelf, void *aValue );

    void *UITable_Create();
    void  UITable_Destroy( void *aSelf );
    void  UITable_OnRowClicked( void *aSelf, void *aHandler );
    void  UITable_AddColumn( void *aSelf, void *aColumn );
    void  UITable_SetRowHeight( void *aSelf, float aRowHeight );
    void  UITable_SetRowBackgroundColor( void *aSelf, void *aColors );
    void  UITable_SetDisplayedRowIndices( void *aSelf, void *aIndices );
    void  UITable_ClearRowBackgroundColor( void *aSelf );

    void *UITextInput_Create();
    void *UITextInput_CreateWithText( void *aText );
    void  UITextInput_Destroy( void *aInstance );
    void *UITextInput_GetText( void *aInstance );
    void  UITextInput_SetHintText( void *aInstance, void *aText );
    void  UITextInput_SetTextColor( void *aInstance, vec4 *aTextColor );
    void  UITextInput_SetBufferSize( void *aInstance, uint32_t aNewSize );
    void  UITextInput_OnTextChanged( void *aInstance, void *aDelegate );

    void *UITextOverlay_Create();
    void  UITextOverlay_Destroy( void *aInstance );
    void  UITextOverlay_AddText( void *aInstance, void *aText );
    void  UITextOverlay_Clear( void *aInstance );

    void *UITextToggleButton_Create();
    void *UITextToggleButton_CreateWithText( void *aText );
    void  UITextToggleButton_Destroy( void *aInstance );
    void  UITextToggleButton_OnClicked( void *aInstance, void *aHandler );
    void  UITextToggleButton_OnChanged( void *aInstance, void *aHandler );
    bool  UITextToggleButton_IsActive( void *aInstance );
    void  UITextToggleButton_SetActive( void *aInstance, bool aValue );
    void  UITextToggleButton_SetActiveColor( void *aInstance, vec4 *aColor );
    void  UITextToggleButton_SetInactiveColor( void *aInstance, vec4 *aColor );

    void *UITreeViewNode_Create();
    void  UITreeViewNode_Destroy( void *aInstance );
    void  UITreeViewNode_SetIcon( void *aInstance, void *aIcon );
    void  UITreeViewNode_SetIndicator( void *aInstance, void *aIndicator );
    void  UITreeViewNode_SetText( void *aInstance, void *aText );
    void  UITreeViewNode_SetTextColor( void *aInstance, vec4 aTextColor );
    void *UITreeViewNode_Add( void *aInstance );

    void *UITreeView_Create();
    void  UITreeView_Destroy( void *aInstance );
    void  UITreeView_SetIndent( void *aInstance, float aIndent );
    void  UITreeView_SetIconSpacing( void *aInstance, float aSpacing );
    void *UITreeView_Add( void *aInstance );

    void *UIVec2Input_Create();
    void  UIVec2Input_Destroy( void *aInstance );
    void  UIVec2Input_OnChanged( void *aInstance, void *aDelegate );
    void  UIVec2Input_SetValue( void *aInstance, vec2 aValue );
    vec2  UIVec2Input_GetValue( void *aInstance );
    void  UIVec2Input_SetFormat( void *aInstance, void *aFormat );
    void  UIVec2Input_SetResetValues( void *aInstance, vec2 aValues );

    void *UIVec3Input_Create();
    void  UIVec3Input_Destroy( void *aInstance );
    void  UIVec3Input_OnChanged( void *aInstance, void *aDelegate );
    void  UIVec3Input_SetValue( void *aInstance, vec3 aValue );
    vec3  UIVec3Input_GetValue( void *aInstance );
    void  UIVec3Input_SetFormat( void *aInstance, void *aFormat );
    void  UIVec3Input_SetResetValues( void *aInstance, vec3 aValues );

    void *UIVec4Input_Create();
    void  UIVec4Input_Destroy( void *aInstance );
    void  UIVec4Input_OnChanged( void *aInstance, void *aDelegate );
    void  UIVec4Input_SetValue( void *aInstance, vec4 aValue );
    vec4  UIVec4Input_GetValue( void *aInstance );
    void  UIVec4Input_SetFormat( void *aInstance, void *aFormat );
    void  UIVec4Input_SetResetValues( void *aInstance, vec4 aValues );

    void *UIWorkspaceDocument_Create();
    void  UIWorkspaceDocument_Destroy( void *aInstance );
    void  UIWorkspaceDocument_SetName( void *aInstance, void *aName );
    void  UIWorkspaceDocument_SetContent( void *aInstance, void *aContent );
    void  UIWorkspaceDocument_Update( void *aInstance );
    bool  UIWorkspaceDocument_IsDirty( void *aInstance );
    void  UIWorkspaceDocument_MarkAsDirty( void *aInstance, bool aDirty );
    void  UIWorkspaceDocument_Open( void *aInstance );
    void  UIWorkspaceDocument_RequestClose( void *aInstance );
    void  UIWorkspaceDocument_ForceClose( void *aInstance );
    void  UIWorkspaceDocument_RegisterSaveDelegate( void *aInstance, void *aDelegate );

    void *UIWorkspace_Create();
    void  UIWorkspace_Destroy( void *aSelf );
    void  UIWorkspace_Add( void *aSelf, void *aDocument );
    void  UIWorkspace_RegisterCloseDocumentDelegate( void *aSelf, void *aDelegate );

    void *UIBoxLayout_CreateWithOrientation( eBoxLayoutOrientation aOrientation );
    void  UIBoxLayout_Destroy( void *aInstance );
    void  UIBoxLayout_AddAlignedNonFixed( void *aInstance, void *aChild, bool aExpand, bool aFill,
                                                 eHorizontalAlignment aHAlignment, eVerticalAlignment aVAlignment );
    void  UIBoxLayout_AddNonAlignedNonFixed( void *aInstance, void *aChild, bool aExpand, bool aFill );
    void  UIBoxLayout_AddAlignedFixed( void *aInstance, void *aChild, float aFixedSize, bool aExpand, bool aFill,
                                              eHorizontalAlignment aHAlignment, eVerticalAlignment aVAlignment );
    void  UIBoxLayout_AddNonAlignedFixed( void *aInstance, void *aChild, float aFixedSize, bool aExpand, bool aFill );
    void  UIBoxLayout_AddSeparator( void *aInstance );
    void  UIBoxLayout_SetItemSpacing( void *aInstance, float aItemSpacing );
    void  UIBoxLayout_Clear( void *aInstance );

    void *UIContainer_Create();
    void  UIContainer_Destroy( void *aInstance );
    void  UIContainer_SetContent( void *aInstance, void *aChild );

    void *UISplitter_Create();
    void *UISplitter_CreateWithOrientation( eBoxLayoutOrientation aOrientation );
    void  UISplitter_Destroy( void *aInstance );
    void  UISplitter_Add1( void *aInstance, void *aChild );
    void  UISplitter_Add2( void *aInstance, void *aChild );
    void  UISplitter_SetItemSpacing( void *aInstance, float aItemSpacing );

    void *UIStackLayout_Create();
    void  UIStackLayout_Destroy( void *aInstance );
    void  UIStackLayout_Add( void *aInstance, void *aChild, void *aKey );
    void  UIStackLayout_SetCurrent( void *aInstance, void *aKey );

    void *UIZLayout_Create();
    void  UIZLayout_Destroy( void *aInstance );
    void  UIZLayout_AddAlignedNonFixed( void *aInstance, void *aChild, bool aExpand, bool aFill,
                                               eHorizontalAlignment aHAlignment, eVerticalAlignment aVAlignment );
    void  UIZLayout_AddNonAlignedNonFixed( void *aInstance, void *aChild, bool aExpand, bool aFill );
    void  UIZLayout_AddAlignedFixed( void *aInstance, void *aChild, vec2 aSize, vec2 aPosition, bool aExpand,
                                            bool aFill, eHorizontalAlignment aHAlignment, eVerticalAlignment aVAlignment );
    void  UIZLayout_AddNonAlignedFixed( void *aInstance, void *aChild, vec2 aSize, vec2 aPosition, bool aExpand,
                                               bool aFill );

    void *UIFileTree_Create();
    void  UIFileTree_Destroy( void *aInstance );
    void *UIFileTree_Add( void *aInstance, void* aPath );

    void *UIDialog_Create();
    void *UIDialog_CreateWithTitleAndSize( void *aTitle, math::vec2 *aSize );
    void  UIDialog_Destroy( void *aInstance );
    void  UIDialog_SetTitle( void *aInstance, void *aTitle );
    void  UIDialog_SetSize( void *aInstance, math::vec2 aSize );
    void  UIDialog_SetContent( void *aInstance, void *aContent );
    void  UIDialog_Open( void *aInstance );
    void  UIDialog_Close( void *aInstance );
    void  UIDialog_Update( void *aInstance );

    void *UIForm_Create();
    void  UIForm_Destroy( void *aInstance );
    void  UIForm_SetTitle( void *aInstance, void *aTitle );
    void  UIForm_SetContent( void *aInstance, void *aContent );
    void  UIForm_Update( void *aInstance );
    void  UIForm_SetSize( void *aInstance, float aWidth, float aHeight );

} // namespace SE::Core::Interop
