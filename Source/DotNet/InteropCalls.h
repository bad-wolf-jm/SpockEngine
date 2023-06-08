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
    void  UIBaseImage_Destroy( void *aSelf );
    void  UIBaseImage_SetImage( void *aSelf, void *aPath );

    void UIBaseImage_SetSize( void *aSelf, vec2 aSize );
    vec2 UIBaseImage_GetSize( void *aSelf );
    void UIBaseImage_SetTopLeft( void *aSelf, vec2 aTopLeft );
    vec2 UIBaseImage_GetTopLeft( void *aSelf );
    void UIBaseImage_SetBottomRight( void *aSelf, vec2 aBottomRight );
    vec2 UIBaseImage_GetBottomRight( void *aSelf );
    void UIBaseImage_SetTintColor( void *aSelf, vec4 aColor );
    vec4 UIBaseImage_GetTintColor( void *aSelf );

    void *UIButton_Create();
    void *UIButton_CreateWithText( void *aText );
    void  UIButton_Destroy( void *aSelf );
    void  UIButton_OnClick( void *aSelf, void *aDelegate );
    void  UIButton_SetText( void *aSelf, void *aText );

    void *UICheckBox_Create();
    void  UICheckBox_Destroy( void *aSelf );
    void  UICheckBox_OnClick( void *aSelf, void *aTextColor );
    bool  UICheckBox_IsChecked( void *aSelf );
    void  UICheckBox_SetIsChecked( void *aSelf, bool aValue );

    void *UIColorButton_Create();
    void  UIColorButton_Destroy( void *aSelf );

    void *UIComboBox_Create();
    void *UIComboBox_CreateWithItems( void *aItems );
    void  UIComboBox_Destroy( void *aSelf );
    int   UIComboBox_GetCurrent( void *aSelf );
    void  UIComboBox_SetCurrent( void *aSelf, int aValue );
    void  UIComboBox_SetItemList( void *aSelf, void *aItems );
    void  UIComboBox_OnChanged( void *aSelf, void *aDelegate );

    void *UIDropdownButton_Create();
    void  UIDropdownButton_Destroy( void *aSelf );
    void  UIDropdownButton_SetContent( void *aSelf, void *aContent );
    void  UIDropdownButton_SetContentSize( void *aSelf, vec2 aSize );
    void  UIDropdownButton_SetImage( void *aSelf, void *aImage );
    void  UIDropdownButton_SetText( void *aSelf, void *aText );
    void  UIDropdownButton_SetTextColor( void *aSelf, vec4 aColor );

    void *UIImage_Create();
    void *UIImage_CreateWithPath( void *aText, vec2 aSize );
    void  UIImage_Destroy( void *aSelf );

    void *UIImageButton_Create();
    void *UIImageButton_CreateWithPath( void *aText, vec2 *aSize );
    void  UIImageButton_Destroy( void *aSelf );
    void  UIImageButton_OnClick( void *aSelf, void *aDelegate );

    void *UIImageToggleButton_Create();
    void  UIImageToggleButton_Destroy( void *aSelf );
    void  UIImageToggleButton_OnClicked( void *aSelf, void *aHandler );
    void  UIImageToggleButton_OnChanged( void *aSelf, void *aHandler );
    bool  UIImageToggleButton_IsActive( void *aSelf );
    void  UIImageToggleButton_SetActive( void *aSelf, bool aValue );
    void  UIImageToggleButton_SetActiveImage( void *aSelf, void *aImage );
    void  UIImageToggleButton_SetInactiveImage( void *aSelf, void *aImage );

    void *UILabel_Create();
    void *UILabel_CreateWithText( void *aText );
    void  UILabel_Destroy( void *aSelf );
    void  UILabel_SetText( void *aSelf, void *aText );
    void  UILabel_SetTextColor( void *aSelf, vec4 aTextColor );

    void *UIMenuItem_Create();
    void *UIMenuItem_CreateWithText( void *aText );
    void *UIMenuItem_CreateWithTextAndShortcut( void *aText, void *aShortcut );
    void  UIMenuItem_Destroy( void *aSelf );
    void  UIMenuItem_SetText( void *aSelf, void *aText );
    void  UIMenuItem_SetShortcut( void *aSelf, void *aShortcut );
    void  UIMenuItem_SetTextColor( void *aSelf, vec4 *aTextColor );
    void  UIMenuItem_OnTrigger( void *aSelf, void *aDelegate );

    void *UIMenuSeparator_Create();
    void  UIMenuSeparator_Destroy( void *aSelf );

    void *UIMenu_Create();
    void *UIMenu_CreateWithText( void *aText );
    void  UIMenu_Destroy( void *aSelf );
    void *UIMenu_AddAction( void *aSelf, void *aText, void *aShortcut );
    void *UIMenu_AddMenu( void *aSelf, void *aText );
    void *UIMenu_AddSeparator( void *aSelf );
    void  UIMenu_Update( void *aSelf );

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
    void  UIPlot_Destroy( void *aSelf );
    void  UIPlot_Clear( void *aSelf );
    void  UIPlot_ConfigureLegend( void *aSelf, vec2 *aLegendPadding, vec2 *aLegendInnerPadding, vec2 *aLegendSpacing );
    void  UIPlot_Add( void *aSelf, void *aPlot );
    void  UIPlot_SetAxisLimits( void *aSelf, int aAxis, double aMin, double aMax );
    void *UIPlot_GetAxisTitle( void *aSelf, int aAxis );
    void  UIPlot_SetAxisTitle( void *aSelf, int aAxis, void *aTitle );

    void *UIProgressBar_Create();
    void  UIProgressBar_Destroy( void *aSelf );
    void  UIProgressBar_SetProgressValue( void *aSelf, float aValue );
    void  UIProgressBar_SetProgressColor( void *aSelf, vec4 aProgressColor );
    void  UIProgressBar_SetText( void *aSelf, void *aValue );
    void  UIProgressBar_SetTextColor( void *aSelf, vec4 aTextColor );
    void  UIProgressBar_SetThickness( void *aSelf, float aValue );

    void *UIPropertyValue_Create();
    void *UIPropertyValue_CreateWithText( void *aText );
    void *UIPropertyValue_CreateWithTextAndOrientation( void *aText, eBoxLayoutOrientation aOrientation );
    void  UIPropertyValue_Destroy( void *aSelf );
    void  UIPropertyValue_SetValue( void *aSelf, void *aText );
    void  UIPropertyValue_SetValueFont( void *aSelf, FontFamilyFlags aFont );
    void  UIPropertyValue_SetNameFont( void *aSelf, FontFamilyFlags aFont );

    void *UISlider_Create();
    void  UISlider_Destroy( void *aSelf );

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
    void  UITextInput_Destroy( void *aSelf );
    void *UITextInput_GetText( void *aSelf );
    void  UITextInput_SetHintText( void *aSelf, void *aText );
    void  UITextInput_SetTextColor( void *aSelf, vec4 *aTextColor );
    void  UITextInput_SetBufferSize( void *aSelf, uint32_t aNewSize );
    void  UITextInput_OnTextChanged( void *aSelf, void *aDelegate );

    void *UITextOverlay_Create();
    void  UITextOverlay_Destroy( void *aSelf );
    void  UITextOverlay_AddText( void *aSelf, void *aText );
    void  UITextOverlay_Clear( void *aSelf );

    void *UITextToggleButton_Create();
    void *UITextToggleButton_CreateWithText( void *aText );
    void  UITextToggleButton_Destroy( void *aSelf );
    void  UITextToggleButton_OnClicked( void *aSelf, void *aHandler );
    void  UITextToggleButton_OnChanged( void *aSelf, void *aHandler );
    bool  UITextToggleButton_IsActive( void *aSelf );
    void  UITextToggleButton_SetActive( void *aSelf, bool aValue );
    void  UITextToggleButton_SetActiveColor( void *aSelf, vec4 *aColor );
    void  UITextToggleButton_SetInactiveColor( void *aSelf, vec4 *aColor );

    void *UITreeViewNode_Create();
    void  UITreeViewNode_Destroy( void *aSelf );
    void  UITreeViewNode_SetIcon( void *aSelf, void *aIcon );
    void  UITreeViewNode_SetIndicator( void *aSelf, void *aIndicator );
    void  UITreeViewNode_SetText( void *aSelf, void *aText );
    void  UITreeViewNode_SetTextColor( void *aSelf, vec4 aTextColor );
    void *UITreeViewNode_Add( void *aSelf );

    void *UITreeView_Create();
    void  UITreeView_Destroy( void *aSelf );
    void  UITreeView_SetIndent( void *aSelf, float aIndent );
    void  UITreeView_SetIconSpacing( void *aSelf, float aSpacing );
    void *UITreeView_Add( void *aSelf );

    void *UIVec2Input_Create();
    void  UIVec2Input_Destroy( void *aSelf );
    void  UIVec2Input_OnChanged( void *aSelf, void *aDelegate );
    void  UIVec2Input_SetValue( void *aSelf, vec2 aValue );
    vec2  UIVec2Input_GetValue( void *aSelf );
    void  UIVec2Input_SetFormat( void *aSelf, void *aFormat );
    void  UIVec2Input_SetResetValues( void *aSelf, vec2 aValues );

    void *UIVec3Input_Create();
    void  UIVec3Input_Destroy( void *aSelf );
    void  UIVec3Input_OnChanged( void *aSelf, void *aDelegate );
    void  UIVec3Input_SetValue( void *aSelf, vec3 aValue );
    vec3  UIVec3Input_GetValue( void *aSelf );
    void  UIVec3Input_SetFormat( void *aSelf, void *aFormat );
    void  UIVec3Input_SetResetValues( void *aSelf, vec3 aValues );

    void *UIVec4Input_Create();
    void  UIVec4Input_Destroy( void *aSelf );
    void  UIVec4Input_OnChanged( void *aSelf, void *aDelegate );
    void  UIVec4Input_SetValue( void *aSelf, vec4 aValue );
    vec4  UIVec4Input_GetValue( void *aSelf );
    void  UIVec4Input_SetFormat( void *aSelf, void *aFormat );
    void  UIVec4Input_SetResetValues( void *aSelf, vec4 aValues );

    void *UIWorkspaceDocument_Create();
    void  UIWorkspaceDocument_Destroy( void *aSelf );
    void  UIWorkspaceDocument_SetName( void *aSelf, void *aName );
    void  UIWorkspaceDocument_SetContent( void *aSelf, void *aContent );
    void  UIWorkspaceDocument_Update( void *aSelf );
    bool  UIWorkspaceDocument_IsDirty( void *aSelf );
    void  UIWorkspaceDocument_MarkAsDirty( void *aSelf, bool aDirty );
    void  UIWorkspaceDocument_Open( void *aSelf );
    void  UIWorkspaceDocument_RequestClose( void *aSelf );
    void  UIWorkspaceDocument_ForceClose( void *aSelf );
    void  UIWorkspaceDocument_RegisterSaveDelegate( void *aSelf, void *aDelegate );

    void *UIWorkspace_Create();
    void  UIWorkspace_Destroy( void *aSelf );
    void  UIWorkspace_Add( void *aSelf, void *aDocument );
    void  UIWorkspace_RegisterCloseDocumentDelegate( void *aSelf, void *aDelegate );

    void *UIBoxLayout_CreateWithOrientation( eBoxLayoutOrientation aOrientation );
    void  UIBoxLayout_Destroy( void *aSelf );
    void  UIBoxLayout_AddAlignedNonFixed( void *aSelf, void *aChild, bool aExpand, bool aFill, eHorizontalAlignment aHAlignment,
                                          eVerticalAlignment aVAlignment );
    void  UIBoxLayout_AddNonAlignedNonFixed( void *aSelf, void *aChild, bool aExpand, bool aFill );
    void  UIBoxLayout_AddAlignedFixed( void *aSelf, void *aChild, float aFixedSize, bool aExpand, bool aFill,
                                       eHorizontalAlignment aHAlignment, eVerticalAlignment aVAlignment );
    void  UIBoxLayout_AddNonAlignedFixed( void *aSelf, void *aChild, float aFixedSize, bool aExpand, bool aFill );
    void  UIBoxLayout_AddSeparator( void *aSelf );
    void  UIBoxLayout_SetItemSpacing( void *aSelf, float aItemSpacing );
    void  UIBoxLayout_Clear( void *aSelf );

    void *UIContainer_Create();
    void  UIContainer_Destroy( void *aSelf );
    void  UIContainer_SetContent( void *aSelf, void *aChild );

    void *UISplitter_Create();
    void *UISplitter_CreateWithOrientation( eBoxLayoutOrientation aOrientation );
    void  UISplitter_Destroy( void *aSelf );
    void  UISplitter_Add1( void *aSelf, void *aChild );
    void  UISplitter_Add2( void *aSelf, void *aChild );
    void  UISplitter_SetItemSpacing( void *aSelf, float aItemSpacing );

    void *UIStackLayout_Create();
    void  UIStackLayout_Destroy( void *aSelf );
    void  UIStackLayout_Add( void *aSelf, void *aChild, void *aKey );
    void  UIStackLayout_SetCurrent( void *aSelf, void *aKey );

    void *UIZLayout_Create();
    void  UIZLayout_Destroy( void *aSelf );
    void  UIZLayout_AddAlignedNonFixed( void *aSelf, void *aChild, bool aExpand, bool aFill, eHorizontalAlignment aHAlignment,
                                        eVerticalAlignment aVAlignment );
    void  UIZLayout_AddNonAlignedNonFixed( void *aSelf, void *aChild, bool aExpand, bool aFill );
    void  UIZLayout_AddAlignedFixed( void *aSelf, void *aChild, vec2 aSize, vec2 aPosition, bool aExpand, bool aFill,
                                     eHorizontalAlignment aHAlignment, eVerticalAlignment aVAlignment );
    void  UIZLayout_AddNonAlignedFixed( void *aSelf, void *aChild, vec2 aSize, vec2 aPosition, bool aExpand, bool aFill );

    void *UIFileTree_Create();
    void  UIFileTree_Destroy( void *aSelf );
    void *UIFileTree_Add( void *aSelf, void *aPath );

    void *UIDialog_Create();
    void *UIDialog_CreateWithTitleAndSize( void *aTitle, math::vec2 *aSize );
    void  UIDialog_Destroy( void *aSelf );
    void  UIDialog_SetTitle( void *aSelf, void *aTitle );
    void  UIDialog_SetSize( void *aSelf, math::vec2 aSize );
    void  UIDialog_SetContent( void *aSelf, void *aContent );
    void  UIDialog_Open( void *aSelf );
    void  UIDialog_Close( void *aSelf );
    void  UIDialog_Update( void *aSelf );

    void *UIForm_Create();
    void  UIForm_Destroy( void *aSelf );
    void  UIForm_SetTitle( void *aSelf, void *aTitle );
    void  UIForm_SetContent( void *aSelf, void *aContent );
    void  UIForm_Update( void *aSelf );
    void  UIForm_SetSize( void *aSelf, float aWidth, float aHeight );

} // namespace SE::Core::Interop
