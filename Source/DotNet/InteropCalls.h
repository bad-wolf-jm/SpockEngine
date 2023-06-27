// #include "UI/Components/Component.h"
// #include "UI/Components/Plot.h"
// #include "UI/Layouts/BoxLayout.h"

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
#include "UI/Components/CodeEditor/CodeEditor.h"
#include "UI/Components/Markdown/Markdown.h"


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
    using namespace math;

    struct CLRVec2
    {
        float x, y;
    };

   struct CLRVec3
    {
        float x, y, z;
    };

    struct CLRVec4
    {
        float x, y, z, w;
    };

    extern "C" 
    {
        wchar_t *OpenFile( wchar_t *aFilter );
        CLRVec4 UIColors_GetStyleColor( ImGuiCol aColor );

        void UIComponent_SetIsVisible( UIComponent *aSelf, bool aIsVisible );
        void UIComponent_SetIsEnabled( UIComponent *aSelf, bool aIsEnabled );
        void UIComponent_SetAllowDragDrop( UIComponent *aSelf, bool aAllowDragDrop );

        void UIComponent_SetPaddingAll( UIComponent *aSelf, float aPaddingAll );
        void UIComponent_SetPaddingPairs( UIComponent *aSelf, float aPaddingTopBottom, float aPaddingLeftRight );
        void UIComponent_SetPaddingIndividual( UIComponent *aSelf, float aPaddingTop, float aPaddingBottom, float aPaddingLeft,
                                            float aPaddingRight );
        void UIComponent_SetAlignment( UIComponent *aSelf, eHorizontalAlignment aHAlignment, eVerticalAlignment aVAlignment );
        void UIComponent_SetHorizontalAlignment( UIComponent *aSelf, eHorizontalAlignment aAlignment );
        void UIComponent_SetVerticalAlignment( UIComponent *aSelf, eVerticalAlignment aAlignment );
        void UIComponent_SetBackgroundColor( UIComponent *aSelf, CLRVec4 aColor );
        void UIComponent_SetFont( UIComponent *aSelf, FontFamilyFlags aFont );
        void UIComponent_SetTooltip( UIComponent *aSelf, void *aTooltip );

        void *UIBaseImage_Create();
        void *UIBaseImage_CreateWithPath( wchar_t *aText, CLRVec2 aSize );
        void  UIBaseImage_Destroy( UIBaseImage *aSelf );
        void  UIBaseImage_SetImage( UIBaseImage *aSelf, wchar_t *aPath );

        void UIBaseImage_SetSize( UIBaseImage *aSelf, CLRVec2 aSize );
        CLRVec2 UIBaseImage_GetSize( UIBaseImage *aSelf );
        void UIBaseImage_SetTopLeft( UIBaseImage *aSelf, CLRVec2 aTopLeft );
        CLRVec2 UIBaseImage_GetTopLeft( UIBaseImage *aSelf );
        void UIBaseImage_SetBottomRight( UIBaseImage *aSelf, CLRVec2 aBottomRight );
        CLRVec2 UIBaseImage_GetBottomRight( UIBaseImage *aSelf );
        void UIBaseImage_SetTintColor( UIBaseImage *aSelf, CLRVec4 aColor );
        CLRVec4 UIBaseImage_GetTintColor( UIBaseImage *aSelf );

        void *UIButton_Create();
        void *UIButton_CreateWithText( wchar_t *aText );
        void  UIButton_Destroy( UIButton *aSelf );
        void  UIButton_OnClick( UIButton *aSelf, void *aDelegate );
        void  UIButton_SetText( UIButton *aSelf, wchar_t *aText );

        void *UICheckBox_Create();
        void  UICheckBox_Destroy( UICheckBox *aSelf );
        void  UICheckBox_OnClick( UICheckBox *aSelf, void *aDelegate );
        bool  UICheckBox_IsChecked( UICheckBox *aSelf );
        void  UICheckBox_SetIsChecked( UICheckBox *aSelf, bool aValue );

        void *UIColorButton_Create();
        void  UIColorButton_Destroy( UIColorButton *aSelf );

        void *UIComboBox_Create();
        void *UIComboBox_CreateWithItems( wchar_t **aItems, int aLength );
        void  UIComboBox_Destroy( UIComboBox *aSelf );
        int   UIComboBox_GetCurrent( UIComboBox *aSelf );
        void  UIComboBox_SetCurrent( UIComboBox *aSelf, int aValue );
        void  UIComboBox_SetItemList( UIComboBox *aSelf, wchar_t **aItems, int aLength );
        void  UIComboBox_OnChanged( UIComboBox *aSelf, void *aDelegate );

        void *UIDropdownButton_Create();
        void  UIDropdownButton_Destroy( UIDropdownButton *aSelf );
        void  UIDropdownButton_SetContent( UIDropdownButton *aSelf, UIComponent *aContent );
        void  UIDropdownButton_SetContentSize( UIDropdownButton *aSelf, CLRVec2 aSize );
        void  UIDropdownButton_SetImage( UIDropdownButton *aSelf, UIImage *aImage );
        void  UIDropdownButton_SetText( UIDropdownButton *aSelf, wchar_t *aText );
        void  UIDropdownButton_SetTextColor( UIDropdownButton *aSelf, CLRVec4 aColor );

        void *UIImage_Create();
        void *UIImage_CreateWithPath( wchar_t *aText, CLRVec2 aSize );
        void  UIImage_Destroy( UIImage *aSelf );

        void *UIImageButton_Create();
        void *UIImageButton_CreateWithPath( wchar_t *aText, CLRVec2 aSize );
        void  UIImageButton_Destroy( UIImageButton *aSelf );
        void  UIImageButton_OnClick( UIImageButton *aSelf, void *aDelegate );

        void *UIImageToggleButton_Create();
        void  UIImageToggleButton_Destroy( UIImageToggleButton *aSelf );
        void  UIImageToggleButton_OnClicked( UIImageToggleButton *aSelf, void *aHandler );
        void  UIImageToggleButton_OnChanged( UIImageToggleButton *aSelf, void *aHandler );
        bool  UIImageToggleButton_IsActive( UIImageToggleButton *aSelf );
        void  UIImageToggleButton_SetActive( UIImageToggleButton *aSelf, bool aValue );
        void  UIImageToggleButton_SetActiveImage( UIImageToggleButton *aSelf, UIBaseImage *aImage );
        void  UIImageToggleButton_SetInactiveImage( UIImageToggleButton *aSelf, UIBaseImage *aImage );

        void *UILabel_Create();
        void *UILabel_CreateWithText( wchar_t *aText );
        void  UILabel_Destroy( UILabel *aSelf );
        void  UILabel_SetText( UILabel *aSelf, wchar_t *aText );
        void  UILabel_SetTextColor( UILabel *aSelf, CLRVec4 aTextColor );

        void *UIMenuItem_Create();
        void *UIMenuItem_CreateWithText( wchar_t *aText );
        void *UIMenuItem_CreateWithTextAndShortcut( wchar_t *aText, wchar_t *aShortcut );
        void  UIMenuItem_Destroy( UIMenuItem *aSelf );
        void  UIMenuItem_SetText( UIMenuItem *aSelf, wchar_t *aText );
        void  UIMenuItem_SetShortcut( UIMenuItem *aSelf, wchar_t *aShortcut );
        void  UIMenuItem_SetTextColor( UIMenuItem *aSelf, CLRVec4  aTextColor );
        void  UIMenuItem_OnTrigger( UIMenuItem *aSelf, void *aDelegate );

        void *UIMenuSeparator_Create();
        void  UIMenuSeparator_Destroy( UIMenuSeparator *aSelf );

        void *UIMenu_Create();
        void *UIMenu_CreateWithText( wchar_t *aText );
        void  UIMenu_Destroy( UIMenu *aSelf );
        void *UIMenu_AddAction( UIMenu *aSelf, wchar_t *aText, wchar_t *aShortcut );
        void *UIMenu_AddMenu( UIMenu *aSelf, wchar_t *aText );
        void *UIMenu_AddSeparator( UIMenu *aSelf );
        void  UIMenu_Update( UIMenu *aSelf );

        void UIPlotData_SetThickness( UIPlotData *aSelf, float aThickness );
        void UIPlotData_SetLegend( UIPlotData *aSelf, wchar_t *aText );
        void UIPlotData_SetColor( UIPlotData *aSelf, CLRVec4 aColor );
        void UIPlotData_SetXAxis( UIPlotData *aSelf, int aAxis );
        void UIPlotData_SetYAxis( UIPlotData *aSelf, int aAxis );

        void *UIFloat64LinePlot_Create();
        void  UIFloat64LinePlot_Destroy( UIFloat64LinePlot *aSelf );
        void  UIFloat64LinePlot_SetX( UIFloat64LinePlot *aSelf, double *aValue, int aLength );
        void  UIFloat64LinePlot_SetY( UIFloat64LinePlot *aSelf, double *aValue, int aLength );

        void *UIFloat64ScatterPlot_Create();
        void  UIFloat64ScatterPlot_Destroy( UIFloat64ScatterPlot *aSelf );
        void  UIFloat64ScatterPlot_SetX( UIFloat64ScatterPlot *aSelf, double *aValue, int aLength );
        void  UIFloat64ScatterPlot_SetY( UIFloat64ScatterPlot *aSelf, double *aValue, int aLength );

        void *UIVLinePlot_Create();
        void  UIVLinePlot_Destroy( UIVLinePlot *aSelf );
        void  UIVLinePlot_SetX( UIVLinePlot *aSelf, double *aValue, int aLength );

        void *UIHLinePlot_Create();
        void  UIHLinePlot_Destroy( UIHLinePlot *aSelf );
        void  UIHLinePlot_SetY( UIHLinePlot *aSelf, double *aValue, int aLength );

        void  *UIVRangePlot_Create();
        void   UIVRangePlot_Destroy( UIVRangePlot *aSelf );
        double UIVRangePlot_GetMin( UIVRangePlot *aSelf );
        void   UIVRangePlot_SetMin( UIVRangePlot *aSelf, double aValue );
        double UIVRangePlot_GetMax( UIVRangePlot *aSelf );
        void   UIVRangePlot_SetMax( UIVRangePlot *aSelf, double aValue );

        void  *UIHRangePlot_Create();
        void   UIHRangePlot_Destroy( UIHRangePlot *aSelf );
        double UIHRangePlot_GetMin( UIHRangePlot *aSelf );
        void   UIHRangePlot_SetMin( UIHRangePlot *aSelf, double aValue );
        double UIHRangePlot_GetMax( UIHRangePlot *aSelf );
        void   UIHRangePlot_SetMax( UIHRangePlot *aSelf, double aValue );

        void *UIAxisTag_Create();
        void *UIAxisTag_CreateWithTextAndColor( UIPlotAxis aAxis, double aX, wchar_t *aText, CLRVec4 aColor );
        void  UIAxisTag_Destroy( UIAxisTag *aSelf );
        void  UIAxisTag_SetX( UIAxisTag *aSelf, double aValue );
        void  UIAxisTag_SetText( UIAxisTag *aSelf, wchar_t *aText );
        CLRVec4  UIAxisTag_GetColor( UIAxisTag *aSelf );
        void  UIAxisTag_SetColor( UIAxisTag *aSelf, CLRVec4 aColor );
        int   UIAxisTag_GetAxis( UIAxisTag *aSelf );
        void  UIAxisTag_SetAxis( UIAxisTag *aSelf, int aAxis );

        void *UIPlot_Create();
        void  UIPlot_Destroy( UIPlot *aSelf );
        void  UIPlot_Clear( UIPlot *aSelf );
        void  UIPlot_ConfigureLegend( UIPlot *aSelf, CLRVec2 aLegendPadding, CLRVec2 aLegendInnerPadding, CLRVec2 aLegendSpacing );
        void  UIPlot_Add( UIPlot *aSelf, UIPlotData *aPlot );
        void  UIPlot_SetAxisLimits( UIPlot *aSelf, int aAxis, double aMin, double aMax );
        wchar_t *UIPlot_GetAxisTitle( UIPlot *aSelf, int aAxis );
        void  UIPlot_SetAxisTitle( UIPlot *aSelf, int aAxis, wchar_t *aTitle );

        void *UIProgressBar_Create();
        void  UIProgressBar_Destroy( UIProgressBar *aSelf );
        void  UIProgressBar_SetProgressValue( UIProgressBar *aSelf, float aValue );
        void  UIProgressBar_SetProgressColor( UIProgressBar *aSelf, CLRVec4 aProgressColor );
        void  UIProgressBar_SetText( UIProgressBar *aSelf, wchar_t *aValue );
        void  UIProgressBar_SetTextColor( UIProgressBar *aSelf, CLRVec4 aTextColor );
        void  UIProgressBar_SetThickness( UIProgressBar *aSelf, float aValue );

        void *UIPropertyValue_Create();
        void *UIPropertyValue_CreateWithText( wchar_t *aText );
        void *UIPropertyValue_CreateWithTextAndOrientation( wchar_t *aText, eBoxLayoutOrientation aOrientation );
        void  UIPropertyValue_Destroy( UIPropertyValue *aSelf );
        void  UIPropertyValue_SetValue( UIPropertyValue *aSelf, wchar_t *aText );
        void  UIPropertyValue_SetValueFont( UIPropertyValue *aSelf, FontFamilyFlags aFont );
        void  UIPropertyValue_SetNameFont( UIPropertyValue *aSelf, FontFamilyFlags aFont );

        void *UISlider_Create();
        void  UISlider_Destroy( UISlider *aSelf );

        void UITableColumn_SetTooltip( UITableColumn *aSelf, UIComponent **aTooptip, int aLength );
        void UITableColumn_SetForegroundColor( UITableColumn *aSelf, CLRVec4 *aForegroundColor, int aLength );
        void UITableColumn_SetBackgroundColor( UITableColumn *aSelf, CLRVec4 *aBackroundColor , int aLength);

        void *UIFloat64Column_Create();
        void *UIFloat64Column_CreateFull( wchar_t *aHeader, float aInitialSize, wchar_t *aFormat, wchar_t *aNaNFormat );
        void  UIFloat64Column_Destroy( UIFloat64Column *aSelf );
        void  UIFloat64Column_Clear( UIFloat64Column *aSelf );
        void  UIFloat64Column_SetData( UIFloat64Column *aSelf, double *aValue, int aLength );

        void *UIUint32Column_Create();
        void *UIUint32Column_CreateFull( wchar_t *aHeader, float aInitialSize );
        void  UIUint32Column_Destroy( UIUint32Column *aSelf );
        void  UIUint32Column_Clear( UIUint32Column *aSelf );
        void  UIUint32Column_SetData( UIUint32Column *aSelf, uint32_t *aValue, int aLength );

        void *UIStringColumn_Create();
        void *UIStringColumn_CreateFull( wchar_t *aHeader, float aInitialSize );
        void  UIStringColumn_Destroy( UIStringColumn *aSelf );
        void  UIStringColumn_Clear( UIStringColumn *aSelf );
        void  UIStringColumn_SetData( UIStringColumn *aSelf, wchar_t **aValue, int aLength );

        void *UITable_Create();
        void  UITable_Destroy( UITable *aSelf );
        void  UITable_OnRowClicked( UITable *aSelf, void *aHandler );
        void  UITable_AddColumn( UITable *aSelf, UITableColumn *aColumn );
        void  UITable_SetRowHeight( UITable *aSelf, float aRowHeight );
        void  UITable_SetRowBackgroundColor( UITable *aSelf, CLRVec4 *aColors, int aLength );
        void  UITable_SetDisplayedRowIndices( UITable *aSelf, int *aIndices, int aLength );
        void  UITable_ClearRowBackgroundColor( UITable *aSelf );

        void *UITextInput_Create();
        void *UITextInput_CreateWithText( wchar_t *aText );
        void  UITextInput_Destroy( UITextInput *aSelf );
        void *UITextInput_GetText( UITextInput *aSelf );
        void  UITextInput_SetHintText( UITextInput *aSelf, wchar_t *aText );
        void  UITextInput_SetTextColor( UITextInput *aSelf, CLRVec4  aTextColor );
        void  UITextInput_SetBufferSize( UITextInput *aSelf, uint32_t aNewSize );
        void  UITextInput_OnTextChanged( UITextInput *aSelf, void *aDelegate );

        void *UITextOverlay_Create();
        void  UITextOverlay_Destroy( UITextOverlay *aSelf );
        void  UITextOverlay_AddText( UITextOverlay *aSelf, wchar_t *aText );
        void  UITextOverlay_Clear( UITextOverlay *aSelf );

        void *UITextToggleButton_Create();
        void *UITextToggleButton_CreateWithText( wchar_t *aText );
        void  UITextToggleButton_Destroy( UITextToggleButton *aSelf );
        void  UITextToggleButton_OnClicked( UITextToggleButton *aSelf, void *aHandler );
        void  UITextToggleButton_OnChanged( UITextToggleButton *aSelf, void *aHandler );
        bool  UITextToggleButton_IsActive( UITextToggleButton *aSelf );
        void  UITextToggleButton_SetActive( UITextToggleButton *aSelf, bool aValue );
        void  UITextToggleButton_SetActiveColor( UITextToggleButton *aSelf, CLRVec4  aColor );
        void  UITextToggleButton_SetInactiveColor( UITextToggleButton *aSelf, CLRVec4  aColor );

        void *UITreeViewNode_Create();
        void  UITreeViewNode_Destroy( UITreeViewNode *aSelf );
        void  UITreeViewNode_SetIcon( UITreeViewNode *aSelf, UIImage *aIcon );
        void  UITreeViewNode_SetIndicator( UITreeViewNode *aSelf, UIComponent *aIndicator );
        void  UITreeViewNode_SetText( UITreeViewNode *aSelf, wchar_t *aText );
        void  UITreeViewNode_SetTextColor( UITreeViewNode *aSelf, CLRVec4 aTextColor );
        void *UITreeViewNode_Add( UITreeViewNode *aSelf );

        void *UITreeView_Create();
        void  UITreeView_Destroy( UITreeView *aSelf );
        void  UITreeView_SetIndent( UITreeView *aSelf, float aIndent );
        void  UITreeView_SetIconSpacing( UITreeView *aSelf, float aSpacing );
        void *UITreeView_Add( UITreeView *aSelf );

        void *UIVec2Input_Create();
        void  UIVec2Input_Destroy( UIVec2Input *aSelf );
        void  UIVec2Input_OnChanged( UIVec2Input *aSelf, void *aDelegate );
        void  UIVec2Input_SetValue( UIVec2Input *aSelf, CLRVec2 aValue );
        CLRVec2  UIVec2Input_GetValue( UIVec2Input *aSelf );
        void  UIVec2Input_SetFormat( UIVec2Input *aSelf, wchar_t *aFormat );
        void  UIVec2Input_SetResetValues( UIVec2Input *aSelf, CLRVec2 aValues );

        void *UIVec3Input_Create();
        void  UIVec3Input_Destroy( UIVec3Input *aSelf );
        void  UIVec3Input_OnChanged( UIVec3Input *aSelf, void *aDelegate );
        void  UIVec3Input_SetValue( UIVec3Input *aSelf, CLRVec3 aValue );
        CLRVec3  UIVec3Input_GetValue( UIVec3Input *aSelf );
        void  UIVec3Input_SetFormat( UIVec3Input *aSelf, wchar_t *aFormat );
        void  UIVec3Input_SetResetValues( UIVec3Input *aSelf, CLRVec3 aValues );

        void *UIVec4Input_Create();
        void  UIVec4Input_Destroy( UIVec4Input *aSelf );
        void  UIVec4Input_OnChanged( UIVec4Input *aSelf, void *aDelegate );
        void  UIVec4Input_SetValue( UIVec4Input *aSelf, CLRVec4 aValue );
        CLRVec4  UIVec4Input_GetValue( UIVec4Input *aSelf );
        void  UIVec4Input_SetFormat( UIVec4Input *aSelf, wchar_t *aFormat );
        void  UIVec4Input_SetResetValues( UIVec4Input *aSelf, CLRVec4 aValues );

        void *UIWorkspaceDocument_Create();
        void  UIWorkspaceDocument_Destroy( UIWorkspaceDocument *aSelf );
        void  UIWorkspaceDocument_SetName( UIWorkspaceDocument *aSelf, wchar_t *aName );
        void  UIWorkspaceDocument_SetContent( UIWorkspaceDocument *aSelf, UIComponent *aContent );
        void  UIWorkspaceDocument_Update( UIWorkspaceDocument *aSelf );
        bool  UIWorkspaceDocument_IsDirty( UIWorkspaceDocument *aSelf );
        void  UIWorkspaceDocument_MarkAsDirty( UIWorkspaceDocument *aSelf, bool aDirty );
        void  UIWorkspaceDocument_Open( UIWorkspaceDocument *aSelf );
        void  UIWorkspaceDocument_RequestClose( UIWorkspaceDocument *aSelf );
        void  UIWorkspaceDocument_ForceClose( UIWorkspaceDocument *aSelf );
        void  UIWorkspaceDocument_RegisterSaveDelegate( UIWorkspaceDocument *aSelf, void *aDelegate );

        void *UIWorkspace_Create();
        void  UIWorkspace_Destroy( UIWorkspace *aSelf );
        void  UIWorkspace_Add( UIWorkspace *aSelf, UIWorkspaceDocument *aDocument );
        void  UIWorkspace_RegisterCloseDocumentDelegate( UIWorkspace *aSelf, void *aDelegate );

        void *UIBoxLayout_CreateWithOrientation( eBoxLayoutOrientation aOrientation );
        void  UIBoxLayout_Destroy( UIBoxLayout *aSelf );
        void  UIBoxLayout_AddAlignedNonFixed( UIBoxLayout *aSelf, UIComponent *aChild, bool aExpand, bool aFill, eHorizontalAlignment aHAlignment,
                                            eVerticalAlignment aVAlignment );
        void  UIBoxLayout_AddNonAlignedNonFixed( UIBoxLayout *aSelf, UIComponent *aChild, bool aExpand, bool aFill );
        void  UIBoxLayout_AddAlignedFixed( UIBoxLayout *aSelf, UIComponent *aChild, float aFixedSize, bool aExpand, bool aFill,
                                        eHorizontalAlignment aHAlignment, eVerticalAlignment aVAlignment );
        void  UIBoxLayout_AddNonAlignedFixed( UIBoxLayout *aSelf, UIComponent *aChild, float aFixedSize, bool aExpand, bool aFill );
        void  UIBoxLayout_AddSeparator( UIBoxLayout *aSelf );
        void  UIBoxLayout_SetItemSpacing( UIBoxLayout *aSelf, float aItemSpacing );
        void  UIBoxLayout_Clear( UIBoxLayout *aSelf );

        void *UIContainer_Create();
        void  UIContainer_Destroy( UIContainer *aSelf );
        void  UIContainer_SetContent( UIContainer *aSelf, UIComponent *aChild );

        void *UISplitter_Create();
        void *UISplitter_CreateWithOrientation( eBoxLayoutOrientation aOrientation );
        void  UISplitter_Destroy( UISplitter *aSelf );
        void  UISplitter_Add1( UISplitter *aSelf, UIComponent *aChild );
        void  UISplitter_Add2( UISplitter *aSelf, UIComponent *aChild );
        void  UISplitter_SetItemSpacing( UISplitter *aSelf, float aItemSpacing );

        void *UIStackLayout_Create();
        void  UIStackLayout_Destroy( UIStackLayout *aSelf );
        void  UIStackLayout_Add( UIStackLayout *aSelf, UIComponent *aChild, wchar_t *aKey );
        void  UIStackLayout_SetCurrent( UIStackLayout *aSelf, wchar_t *aKey );

        void *UIZLayout_Create();
        void  UIZLayout_Destroy( UIZLayout *aSelf );
        void  UIZLayout_AddAlignedNonFixed( UIZLayout *aSelf, UIComponent *aChild, bool aExpand, bool aFill, eHorizontalAlignment aHAlignment,
                                            eVerticalAlignment aVAlignment );
        void  UIZLayout_AddNonAlignedNonFixed( UIZLayout *aSelf, UIComponent *aChild, bool aExpand, bool aFill );
        void  UIZLayout_AddAlignedFixed( UIZLayout *aSelf, UIComponent *aChild, CLRVec2 aSize, CLRVec2 aPosition, bool aExpand, bool aFill,
                                        eHorizontalAlignment aHAlignment, eVerticalAlignment aVAlignment );
        void  UIZLayout_AddNonAlignedFixed( UIZLayout *aSelf, UIComponent *aChild, CLRVec2 aSize, CLRVec2 aPosition, bool aExpand, bool aFill );

        void *UIFileTree_Create();
        void  UIFileTree_Destroy( UIFileTree *aSelf );
        void *UIFileTree_Add( UIFileTree *aSelf, wchar_t *aPath );

        void *UIDialog_Create();
        void *UIDialog_CreateWithTitleAndSize( wchar_t *aTitle, CLRVec2 aSize );
        void  UIDialog_Destroy( UIDialog *aSelf );
        void  UIDialog_SetTitle( UIDialog *aSelf, wchar_t *aTitle );
        void  UIDialog_SetSize( UIDialog *aSelf, CLRVec2 aSize );
        void  UIDialog_SetContent( UIDialog *aSelf, UIComponent *aContent );
        void  UIDialog_Open( UIDialog *aSelf );
        void  UIDialog_Close( UIDialog *aSelf );
        void  UIDialog_Update( UIDialog *aSelf );

        void *UIForm_Create();
        void  UIForm_Destroy( UIForm *aSelf );
        void  UIForm_SetTitle( UIForm *aSelf, wchar_t *aTitle );
        void  UIForm_SetContent( UIForm *aSelf, UIComponent *aContent );
        void  UIForm_Update( UIForm *aSelf );
        void  UIForm_SetSize( UIForm *aSelf, float aWidth, float aHeight );
    }
} // namespace SE::Core::Interop
