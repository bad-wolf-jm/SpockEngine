using System;
using System.Runtime.CompilerServices;

namespace SpockEngine
{
    public static class Interop
    {
        // #region UIBaseImage
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIBaseImage_Create();
        // #end region UIBaseImage

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIBaseImage_CreateWithPath(string aText, Math.vec2 Size);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIBaseImage_Destroy(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIBaseImage_SetImage(IntPtr aSelf, string aPath);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIBaseImage_SetSize(IntPtr aSelf, Math.vec2 aSize);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static Math.vec2 UIBaseImage_GetSize(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIBaseImage_SetTopLeft(IntPtr aSelf, Math.vec2 aTopLeft);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static Math.vec2 UIBaseImage_GetTopLeft(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIBaseImage_SetBottomRight(IntPtr aSelf, Math.vec2 aBottomRight);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static Math.vec2 UIBaseImage_GetBottomRight(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIBaseImage_SetTintColor(IntPtr aSelf, Math.vec4 aColor);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static Math.vec4 UIBaseImage_GetTintColor(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIButton_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIButton_CreateWithText(string aText);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIButton_Destroy(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIButton_SetText(IntPtr aSelf, string aText);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIButton_OnClick(IntPtr aSelf, IntPtr aDelegate);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UICheckBox_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UICheckBox_Destroy(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UICheckBox_OnClick(IntPtr aSelf, IntPtr aHandler);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static bool UICheckBox_IsChecked(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UICheckBox_SetIsChecked(IntPtr aSelf, bool aValue);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIColorButton_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIColorButton_Destroy(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIComboBox_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIComboBox_CreateWithItems(string[] aItems);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIComboBox_Destroy(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static int UIComboBox_GetCurrent(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIComboBox_SetCurrent(IntPtr aSelf, int aValue);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIComboBox_SetItemList(IntPtr aSelf, string[] aItems);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIComboBox_OnChanged(IntPtr aSelf, IntPtr aHandler);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIComponent_SetIsVisible(IntPtr aSelf, bool aIsVisible);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIComponent_SetIsEnabled(IntPtr aSelf, bool aIsEnabled);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIComponent_SetAllowDragDrop(IntPtr aSelf, bool aAllowDragDrop);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIComponent_SetPaddingAll(IntPtr aSelf, float aPaddingAll);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIComponent_SetPaddingPairs(IntPtr aSelf, float aPaddingTopBottom, float aPaddingLeftRight);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIComponent_SetPaddingIndividual(IntPtr aSelf, float aPaddingTop, float aPaddingBottom, float aPaddingLeft, float aPaddingRight);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIComponent_SetAlignment(IntPtr aSelf, eHorizontalAlignment aHAlignment, eVerticalAlignment aVAlignment);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIComponent_SetHorizontalAlignment(IntPtr aSelf, eHorizontalAlignment aAlignment);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIComponent_SetVerticalAlignment(IntPtr aSelf, eVerticalAlignment aAlignment);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIComponent_SetBackgroundColor(IntPtr aSelf, Math.vec4 aColor);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIComponent_SetFont(IntPtr aSelf, eFontFamily aFont);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIComponent_SetTooltip(IntPtr aSelf, IntPtr aTooltip);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIDropdownButton_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIDropdownButton_Destroy(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static bool UIDropdownButton_SetContent(IntPtr aSelf, IntPtr aContent);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static bool UIDropdownButton_SetContentSize(IntPtr aSelf, Math.vec2 aSize);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIDropdownButton_SetImage(IntPtr aSelf, IntPtr aImage);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIDropdownButton_SetText(IntPtr aSelf, string aText);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIDropdownButton_SetTextColor(IntPtr aSelf, Math.vec4 aColor);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIImage_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIImage_CreateWithPath(string aText, Math.vec2 Size);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIImage_Destroy(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIImageButton_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIImageButton_CreateWithPath(string aText, Math.vec2 Size);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIImageButton_Destroy(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIImageButton_OnClick(IntPtr aSelf, IntPtr aText);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIImageToggleButton_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIImageToggleButton_Destroy(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIImageToggleButton_OnClicked(IntPtr aSelf, IntPtr aHandler);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIImageToggleButton_OnChanged(IntPtr aSelf, IntPtr aHandler);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static bool UIImageToggleButton_IsActive(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIImageToggleButton_SetActive(IntPtr aSelf, bool aValue);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIImageToggleButton_SetActiveImage(IntPtr aSelf, IntPtr aImage);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIImageToggleButton_SetInactiveImage(IntPtr aSelf, IntPtr aImage);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UILabel_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UILabel_CreateWithText(string aText);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UILabel_Destroy(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UILabel_SetText(IntPtr aSelf, string aText);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UILabel_SetTextColor(IntPtr aSelf, Math.vec4 aText);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIMenuItem_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIMenuItem_CreateWithTextAndShortcut(string aText, string aShortcut);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIMenuItem_CreateWithText(string aText);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIMenuItem_Destroy(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIMenuItem_SetText(IntPtr aSelf, string aText);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIMenuItem_SetShortcut(IntPtr aSelf, string aShortcut);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIMenuItem_SetTextColor(IntPtr aSelf, Math.vec4 aColor);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIMenuItem_OnTrigger(IntPtr aSelf, IntPtr aHandler);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIMenuSeparator_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIMenuSeparator_Destroy(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIMenu_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIMenu_CreateWithText(string aText);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIMenu_Destroy(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIMenu_AddAction(IntPtr aSelf, string aName, string aShortcut);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIMenu_AddSeparator(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIMenu_AddMenu(IntPtr aSelf, string aName);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIMenu_Update(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIPlotData_SetThickness(IntPtr aSelf, float aThickness);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIPlotData_SetLegend(IntPtr aSelf, string aText);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIPlotData_SetColor(IntPtr aSelf, Math.vec4 aColor);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIPlotData_SetXAxis(IntPtr aSelf, eUIPlotAxis aAxis);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIPlotData_SetYAxis(IntPtr aSelf, eUIPlotAxis aAxis);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIVLinePlot_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIVLinePlot_Destroy(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIVLinePlot_SetX(IntPtr aSelf, double[] aValues);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIHLinePlot_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIHLinePlot_Destroy(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIHLinePlot_SetY(IntPtr aSelf, double[] aValues);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIAxisTag_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIAxisTag_CreateWithTextAndColor(eUIPlotAxis aAxis, double aX, string aText, Math.vec4 aColor);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIAxisTag_Destroy(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIAxisTag_SetX(IntPtr aSelf, double aValue);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIAxisTag_SetText(IntPtr aSelf, string aText);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static Math.vec4 UIAxisTag_GetColor(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIAxisTag_SetColor(IntPtr aSelf, Math.vec4 aColor);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static eUIPlotAxis UIAxisTag_GetAxis(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIAxisTag_SetAxis(IntPtr aSelf, eUIPlotAxis aColor);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIVRangePlot_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIVRangePlot_Destroy(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static double UIVRangePlot_GetMin(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIVRangePlot_SetMin(IntPtr aSelf, double aValue);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static double UIVRangePlot_GetMax(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIVRangePlot_SetMax(IntPtr aSelf, double aValue);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIHRangePlot_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIHRangePlot_Destroy(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static double UIHRangePlot_GetMin(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIHRangePlot_SetMin(IntPtr aSelf, double aValue);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static double UIHRangePlot_GetMax(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIHRangePlot_SetMax(IntPtr aSelf, double aValue);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIFloat64LinePlot_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIFloat64LinePlot_Destroy(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIFloat64LinePlot_SetX(IntPtr aSelf, double[] aValues);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIFloat64LinePlot_SetY(IntPtr aSelf, double[] aValues);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIFloat64ScatterPlot_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIFloat64ScatterPlot_Destroy(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIFloat64ScatterPlot_SetX(IntPtr aSelf, double[] aValues);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIFloat64ScatterPlot_SetY(IntPtr aSelf, double[] aValues);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIPlot_SetAxisLimits(IntPtr aSelf, eUIPlotAxis aAxis, double aMin, double aMax);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static string UIPlot_GetAxisTitle(IntPtr aSelf, eUIPlotAxis aAxis);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIPlot_SetAxisTitle(IntPtr aSelf, eUIPlotAxis aAxis, string aTitle);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIPlot_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIPlot_Destroy(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIPlot_Clear(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIPlot_ConfigureLegend(IntPtr aSelf, Math.vec2 aLegendPadding, Math.vec2 aLegendInnerPadding, Math.vec2 aLegendSpacing);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIPlot_Add(IntPtr aSelf, IntPtr aPlot);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIPlot_PlotVLines(IntPtr aSelf, double[] a, string aLegend, Math.vec4 aColor);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIProgressBar_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIProgressBar_Destroy(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIProgressBar_SetText(IntPtr aSelf, string aText);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIProgressBar_SetTextColor(IntPtr aSelf, Math.vec4 aText);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIProgressBar_SetProgressValue(IntPtr aSelf, float aValue);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIProgressBar_SetProgressColor(IntPtr aSelf, Math.vec4 aColor);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIProgressBar_SetThickness(IntPtr aSelf, float aThickness);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIPropertyValue_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIPropertyValue_CreateWithText(string aText);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIPropertyValue_CreateWithTextAndOrientation(string aText, eBoxLayoutOrientation aOrientation);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIPropertyValue_Destroy(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIPropertyValue_SetValue(IntPtr aSelf, string aText);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIPropertyValue_SetValueFont(IntPtr aSelf, eFontFamily aFont);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIPropertyValue_SetNameFont(IntPtr aSelf, eFontFamily aFont);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UISlider_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UISlider_Destroy(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UITableColumn_SetTooltip(IntPtr aSelf, IntPtr[] aValue);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UITableColumn_SetBackgroundColor(IntPtr aSelf, Math.vec4[] aForegroundColor);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UITableColumn_SetForegroundColor(IntPtr aSelf, Math.vec4[] aForegroundColor);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIFloat64Column_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIFloat64Column_CreateFull(string aHeader, float aInitialSize, string aFormat, string aNaNFormat);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIFloat64Column_Destroy(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIFloat64Column_Clear(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIFloat64Column_SetData(IntPtr aSelf, double[] aValue);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIUint32Column_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIUint32Column_CreateFull(string aHeader, float aInitialSize);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIUint32Column_Destroy(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIUint32Column_Clear(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIUint32Column_SetData(IntPtr aSelf, uint[] aValue);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIStringColumn_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIStringColumn_CreateFull(string aHeader, float aInitialSize);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIStringColumn_Destroy(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIStringColumn_Clear(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIStringColumn_SetData(IntPtr aSelf, string[] aValue);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UITable_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UITable_Destroy(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UITable_OnRowClicked(IntPtr aSelf, IntPtr aHandler);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UITable_AddColumn(IntPtr aSelf, IntPtr aColumnInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UITable_SetRowHeight(IntPtr aSelf, float aRowHeight);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UITable_SetRowBackgroundColor(IntPtr aSelf, Math.vec4[] aColors);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UITable_ClearRowBackgroundColor(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UITable_SetDisplayedRowIndices(IntPtr aSelf, int[] aIndices);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UITextInput_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UITextInput_CreateWithText(string aText);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UITextInput_Destroy(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static string UITextInput_GetText(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UITextInput_OnTextChanged(IntPtr aSelf, IntPtr aHandler);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UITextInput_SetHintText(IntPtr aSelf, string aText);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UITextInput_SetTextColor(IntPtr aSelf, Math.vec4 aColor);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UITextInput_SetBufferSize(IntPtr aSelf, uint aSize);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UITextOverlay_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UITextOverlay_Destroy(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UITextOverlay_AddText(IntPtr aSelf, string aText);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UITextOverlay_Clear(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UITextToggleButton_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UITextToggleButton_CreateWithText(string aText);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UITextToggleButton_Destroy(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UITextToggleButton_OnClicked(IntPtr aSelf, IntPtr aHandler);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UITextToggleButton_OnChanged(IntPtr aSelf, IntPtr aHandler);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static bool UITextToggleButton_IsActive(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UITextToggleButton_SetActive(IntPtr aSelf, bool aValue);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UITextToggleButton_SetActiveColor(IntPtr aSelf, Math.vec4 aColor);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UITextToggleButton_SetInactiveColor(IntPtr aSelf, Math.vec4 aColor);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UITreeViewNode_Destroy(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UITreeViewNode_SetText(IntPtr aSelf, string aValue);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UITreeViewNode_SetTextColor(IntPtr aSelf, Math.vec4 aColor);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UITreeViewNode_SetIcon(IntPtr aSelf, IntPtr aIcon);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UITreeViewNode_SetIndicator(IntPtr aSelf, IntPtr aIcon);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UITreeViewNode_Add(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UITreeView_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UITreeView_Destroy(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UITreeView_SetIndent(IntPtr aSelf, float aIndent);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UITreeView_SetIconSpacing(IntPtr aSelf, float aSpacing);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UITreeView_Add(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIVec2Input_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIVec2Input_Destroy(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIVec2Input_OnChanged(IntPtr aSelf, IntPtr aDelegate);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIVec2Input_SetValue(IntPtr aSelf, Math.vec2 aValue);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIVec2Input_SetResetValues(IntPtr aSelf, Math.vec2 aValue);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static Math.vec2 UIVec2Input_GetValue(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIVec2Input_SetFormat(IntPtr aSelf, string aFormat);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIVec3Input_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIVec3Input_Destroy(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIVec3Input_OnChanged(IntPtr aSelf, IntPtr aDelegate);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIVec3Input_SetValue(IntPtr aSelf, Math.vec2 aValue);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIVec3Input_SetResetValues(IntPtr aSelf, Math.vec2 aValue);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static Math.vec2 UIVec3Input_GetValue(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIVec3Input_SetFormat(IntPtr aSelf, string aFormat);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIVec4Input_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIVec4Input_Destroy(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIVec4Input_OnChanged(IntPtr aSelf, IntPtr aDelegate);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIVec4Input_SetValue(IntPtr aSelf, Math.vec2 aValue);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIVec4Input_SetResetValues(IntPtr aSelf, Math.vec2 aValue);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static Math.vec2 UIVec4Input_GetValue(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIVec4Input_SetFormat(IntPtr aSelf, string aFormat);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIWorkspaceDocument_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIWorkspaceDocument_RegisterSaveDelegate(IntPtr aSelf, IntPtr aDelegate);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIWorkspaceDocument_Destroy(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIWorkspaceDocument_SetContent(IntPtr aSelf, IntPtr aContent);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIWorkspaceDocument_SetName(IntPtr aSelf, string aName);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIWorkspaceDocument_Update(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static bool UIWorkspaceDocument_IsDirty(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIWorkspaceDocument_MarkAsDirty(IntPtr aSelf, bool aDirty);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIWorkspaceDocument_Open(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIWorkspaceDocument_RequestClose(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIWorkspaceDocument_ForceClose(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIWorkspace_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIWorkspace_RegisterCloseDocumentDelegate(IntPtr aSelf, IntPtr aDelegate);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIWorkspace_Destroy(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIWorkspace_Add(IntPtr aSelf, IntPtr aDocument);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIBoxLayout_CreateWithOrientation(eBoxLayoutOrientation aOrientation);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIBoxLayout_AddAlignedNonFixed(IntPtr aSelf, IntPtr aChild, bool aExpand, bool aFill, eHorizontalAlignment aHAlignment, eVerticalAlignment aVAlignment);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIBoxLayout_AddNonAlignedNonFixed(IntPtr aSelf, IntPtr aChild, bool aExpand, bool aFill);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIBoxLayout_AddAlignedFixed(IntPtr aSelf, IntPtr aChild, float aFixedSize, bool aExpand, bool aFill, eHorizontalAlignment aHAlignment, eVerticalAlignment aVAlignment);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIBoxLayout_AddNonAlignedFixed(IntPtr aSelf, IntPtr aChild, float aFixedSize, bool aExpand, bool aFill);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIBoxLayout_AddSeparator(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIBoxLayout_Destroy(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIBoxLayout_SetItemSpacing(IntPtr aSelf, float aItemSpacing);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIBoxLayout_Clear(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIContainer_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIContainer_Destroy(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIContainer_SetContent(IntPtr aSelf, IntPtr aChild);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UISplitter_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UISplitter_CreateWithOrientation(eBoxLayoutOrientation aOrientation);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UISplitter_Add1(IntPtr aSelf, IntPtr aChild);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UISplitter_Add2(IntPtr aSelf, IntPtr aChild);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UISplitter_Destroy(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UISplitter_SetItemSpacing(IntPtr aSelf, float aItemSpacing);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIStackLayout_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIStackLayout_Destroy(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIStackLayout_Add(IntPtr aSelf, IntPtr aChild, string aKey);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIStackLayout_SetCurrent(IntPtr aSelf, string aKey);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIZLayout_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIZLayout_AddAlignedNonFixed(IntPtr aSelf, IntPtr aChild, bool aExpand, bool aFill, eHorizontalAlignment aHAlignment, eVerticalAlignment aVAlignment);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIZLayout_AddNonAlignedNonFixed(IntPtr aSelf, IntPtr aChild, bool aExpand, bool aFill);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIZLayout_AddAlignedFixed(IntPtr aSelf, IntPtr aChild, SpockEngine.Math.vec2 aSize, SpockEngine.Math.vec2 aPosition, bool aExpand, bool aFill, eHorizontalAlignment aHAlignment, eVerticalAlignment aVAlignment);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIZLayout_AddNonAlignedFixed(IntPtr aSelf, IntPtr aChild, SpockEngine.Math.vec2 aSize, SpockEngine.Math.vec2 aPosition, bool aExpand, bool aFill);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIZLayout_Destroy(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIFileTree_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIFileTree_Destroy(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIFileTree_Add(IntPtr aSelf, string aText);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIDialog_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIDialog_CreateWithTitleAndSize(IntPtr aSelf, string aTitle, Math.vec2 aSize);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIDialog_Destroy(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIDialog_SetTitle(IntPtr aSelf, string aTitle);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIDialog_SetSize(IntPtr aSelf, Math.vec2 aSize);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIDialog_SetContent(IntPtr aSelf, IntPtr aContent);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIDialog_Update(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIDialog_Open(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIDialog_Close(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static IntPtr UIForm_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIForm_Destroy(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIForm_SetTitle(IntPtr aSelf, string aTitle);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIForm_SetContent(IntPtr aSelf, IntPtr aContent);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIForm_Update(IntPtr aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static void UIForm_SetSize(IntPtr aSelf, float aWidth, float aHeight);

    }
}