using System;
using System.Runtime.CompilerServices;

namespace SpockEngine
{
    public static class CppCall
    {
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIBaseImage_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIBaseImage_CreateWithPath(string aText, Math.vec2 Size);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIBaseImage_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIBaseImage_SetImage(ulong aInstance, string aPath);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIBaseImage_SetSize(ulong aInstance, Math.vec2 aSize);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static Math.vec2 UIBaseImage_GetSize(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIBaseImage_SetTopLeft(ulong aInstance, Math.vec2 aTopLeft);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static Math.vec2 UIBaseImage_GetTopLeft(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIBaseImage_SetBottomRight(ulong aInstance, Math.vec2 aBottomRight);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static Math.vec2 UIBaseImage_GetBottomRight(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIBaseImage_SetTintColor(ulong aInstance, Math.vec4 aColor);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static Math.vec4 UIBaseImage_GetTintColor(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIButton_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIButton_CreateWithText(string aText);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIButton_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIButton_SetText(ulong aInstance, string aText);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIButton_OnClick(ulong aInstance, IntPtr aDelegate);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UICheckBox_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UICheckBox_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UICheckBox_OnClick(ulong aInstance, IntPtr aHandler);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static bool UICheckBox_IsChecked(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UICheckBox_SetIsChecked(ulong aInstance, bool aValue);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIColorButton_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIColorButton_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIComboBox_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIComboBox_CreateWithItems(string[] aItems);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIComboBox_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static int UIComboBox_GetCurrent(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIComboBox_SetCurrent(ulong aInstance, int aValue);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIComboBox_SetItemList(ulong aInstance, string[] aItems);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIComboBox_OnChanged(ulong aInstance, IntPtr aHandler);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIComponent_SetIsVisible(ulong aSelf, bool aIsVisible);
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIComponent_SetIsEnabled(ulong aSelf, bool aIsEnabled);
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIComponent_SetAllowDragDrop(ulong aSelf, bool aAllowDragDrop);


        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIComponent_SetPaddingAll(ulong aSelf, float aPaddingAll);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIComponent_SetPaddingPairs(ulong aSelf, float aPaddingTopBottom, float aPaddingLeftRight);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIComponent_SetPaddingIndividual(ulong aSelf, float aPaddingTop, float aPaddingBottom, float aPaddingLeft, float aPaddingRight);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIComponent_SetAlignment(ulong aSelf, eHorizontalAlignment aHAlignment, eVerticalAlignment aVAlignment);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIComponent_SetHorizontalAlignment(ulong aSelf, eHorizontalAlignment aAlignment);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIComponent_SetVerticalAlignment(ulong aSelf, eVerticalAlignment aAlignment);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIComponent_SetBackgroundColor(ulong aSelf, Math.vec4 aColor);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIComponent_SetFont(ulong aSelf, eFontFamily aFont);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIComponent_SetTooltip(ulong aSelf, ulong aTooltip);


        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIDropdownButton_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIDropdownButton_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static bool UIDropdownButton_SetContent(ulong aInstance, ulong aContent);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static bool UIDropdownButton_SetContentSize(ulong aInstance, Math.vec2 aSize);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIDropdownButton_SetImage(ulong aInstance, ulong aImage);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIDropdownButton_SetText(ulong aInstance, string aText);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIDropdownButton_SetTextColor(ulong aInstance, Math.vec4 aColor);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIImage_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIImage_CreateWithPath(string aText, Math.vec2 Size);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIImage_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIImageButton_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIImageButton_CreateWithPath(string aText, Math.vec2 Size);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIImageButton_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIImageButton_OnClick(ulong aInstance, IntPtr aText);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIImageToggleButton_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIImageToggleButton_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIImageToggleButton_OnClicked(ulong aInstance, IntPtr aHandler);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIImageToggleButton_OnChanged(ulong aInstance, IntPtr aHandler);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static bool UIImageToggleButton_IsActive(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIImageToggleButton_SetActive(ulong aInstance, bool aValue);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIImageToggleButton_SetActiveImage(ulong aInstance, ulong aImage);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIImageToggleButton_SetInactiveImage(ulong aInstance, ulong aImage);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UILabel_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UILabel_CreateWithText(string aText);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UILabel_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UILabel_SetText(ulong aInstance, string aText);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UILabel_SetTextColor(ulong aInstance, Math.vec4 aText);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIMenuItem_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIMenuItem_CreateWithTextAndShortcut(string aText, string aShortcut);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIMenuItem_CreateWithText(string aText);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIMenuItem_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIMenuItem_SetText(ulong aInstance, string aText);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIMenuItem_SetShortcut(ulong aInstance, string aShortcut);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIMenuItem_SetTextColor(ulong aInstance, Math.vec4 aColor);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIMenuItem_OnTrigger(ulong aInstance, IntPtr aHandler);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIMenuSeparator_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIMenuSeparator_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIMenu_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIMenu_CreateWithText(string aText);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIMenu_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIMenu_AddAction(ulong aInstance, string aName, string aShortcut);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIMenu_AddSeparator(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIMenu_AddMenu(ulong aInstance, string aName);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIMenu_Update(ulong aInstance);


        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIPlotData_SetThickness(ulong aInstance, float aThickness);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIPlotData_SetLegend(ulong aInstance, string aText);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIPlotData_SetColor(ulong aInstance, Math.vec4 aColor);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIPlotData_SetXAxis(ulong aInstance, eUIPlotAxis aAxis);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIPlotData_SetYAxis(ulong aInstance, eUIPlotAxis aAxis);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIVLinePlot_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIVLinePlot_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIVLinePlot_SetX(ulong aInstance, double[] aValues);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIHLinePlot_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIHLinePlot_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIHLinePlot_SetY(ulong aInstance, double[] aValues);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIAxisTag_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIAxisTag_CreateWithTextAndColor(eUIPlotAxis aAxis, double aX, string aText, Math.vec4 aColor);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIAxisTag_Destroy(ulong aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIAxisTag_SetX(ulong aSelf, double aValue);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIAxisTag_SetText(ulong aSelf, string aText);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static Math.vec4 UIAxisTag_GetColor(ulong aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIAxisTag_SetColor(ulong aSelf, Math.vec4 aColor);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static eUIPlotAxis UIAxisTag_GetAxis(ulong aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIAxisTag_SetAxis(ulong aSelf, eUIPlotAxis aColor);


        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIVRangePlot_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIVRangePlot_Destroy(ulong aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static double UIVRangePlot_GetMin(ulong aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIVRangePlot_SetMin(ulong aSelf, double aValue);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static double UIVRangePlot_GetMax(ulong aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIVRangePlot_SetMax(ulong aSelf, double aValue);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIHRangePlot_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIHRangePlot_Destroy(ulong aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static double UIHRangePlot_GetMin(ulong aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIHRangePlot_SetMin(ulong aSelf, double aValue);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static double UIHRangePlot_GetMax(ulong aSelf);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIHRangePlot_SetMax(ulong aSelf, double aValue);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIFloat64LinePlot_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIFloat64LinePlot_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIFloat64LinePlot_SetX(ulong aInstance, double[] aValues);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIFloat64LinePlot_SetY(ulong aInstance, double[] aValues);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIFloat64ScatterPlot_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIFloat64ScatterPlot_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIFloat64ScatterPlot_SetX(ulong aInstance, double[] aValues);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIFloat64ScatterPlot_SetY(ulong aInstance, double[] aValues);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIPlot_SetAxisLimits(ulong aInstance, eUIPlotAxis aAxis, double aMin, double aMax);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static string UIPlot_GetAxisTitle(ulong aInstance, eUIPlotAxis aAxis);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIPlot_SetAxisTitle(ulong aInstance, eUIPlotAxis aAxis, string aTitle);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIPlot_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIPlot_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIPlot_Clear(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIPlot_ConfigureLegend(ulong aInstance, Math.vec2 aLegendPadding, Math.vec2 aLegendInnerPadding, Math.vec2 aLegendSpacing);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIPlot_Add(ulong aInstance, ulong aPlot);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIPlot_PlotVLines(ulong aInstance, double[] a, string aLegend, Math.vec4 aColor);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIProgressBar_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIProgressBar_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIProgressBar_SetText(ulong aInstance, string aText);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIProgressBar_SetTextColor(ulong aInstance, Math.vec4 aText);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIProgressBar_SetProgressValue(ulong aInstance, float aValue);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIProgressBar_SetProgressColor(ulong aInstance, Math.vec4 aColor);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIProgressBar_SetThickness(ulong aInstance, float aThickness);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIPropertyValue_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIPropertyValue_CreateWithText(string aText);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIPropertyValue_CreateWithTextAndOrientation(string aText, eBoxLayoutOrientation aOrientation);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIPropertyValue_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIPropertyValue_SetValue(ulong aInstance, string aText);


        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIPropertyValue_SetValueFont(ulong aInstance, eFontFamily aFont);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIPropertyValue_SetNameFont(ulong aInstance, eFontFamily aFont);


        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UISlider_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UISlider_Destroy(ulong aInstance);

                [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UITableColumn_SetTooltip(ulong aInstance, ulong[] aValue);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UITableColumn_SetBackgroundColor(ulong aInstance, Math.vec4[] aForegroundColor);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UITableColumn_SetForegroundColor(ulong aInstance, Math.vec4[] aForegroundColor);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIFloat64Column_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIFloat64Column_CreateFull(string aHeader, float aInitialSize, string aFormat, string aNaNFormat);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIFloat64Column_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIFloat64Column_Clear(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIFloat64Column_SetData(ulong aInstance, double[] aValue);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIUint32Column_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIUint32Column_CreateFull(string aHeader, float aInitialSize);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIUint32Column_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIUint32Column_Clear(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIUint32Column_SetData(ulong aInstance, uint[] aValue);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIStringColumn_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIStringColumn_CreateFull(string aHeader, float aInitialSize);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIStringColumn_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIStringColumn_Clear(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIStringColumn_SetData(ulong aInstance, string[] aValue);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UITable_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UITable_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UITable_OnRowClicked(ulong aInstance, IntPtr aHandler);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UITable_AddColumn(ulong aInstance, ulong aColumnInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UITable_SetRowHeight(ulong aInstance, float aRowHeight);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UITable_SetRowBackgroundColor(ulong aInstance, Math.vec4[] aColors);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UITable_ClearRowBackgroundColor(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UITable_SetDisplayedRowIndices(ulong aInstance, int[] aIndices);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UITextInput_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UITextInput_CreateWithText(string aText);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UITextInput_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static string UITextInput_GetText(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UITextInput_OnTextChanged(ulong aInstance, OnChangeDelegate aHandler);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UITextInput_SetHintText(ulong aInstance, string aText);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UITextInput_SetTextColor(ulong aInstance, Math.vec4 aColor);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UITextInput_SetBufferSize(ulong aInstance, uint aSize);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UITextOverlay_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UITextOverlay_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UITextOverlay_AddText(ulong aInstance, string aText);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UITextOverlay_Clear(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UITextToggleButton_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UITextToggleButton_CreateWithText(string aText);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UITextToggleButton_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UITextToggleButton_OnClicked(ulong aInstance, IntPtr aHandler);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UITextToggleButton_OnChanged(ulong aInstance, IntPtr aHandler);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static bool UITextToggleButton_IsActive(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UITextToggleButton_SetActive(ulong aInstance, bool aValue);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UITextToggleButton_SetActiveColor(ulong aInstance, Math.vec4 aColor);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UITextToggleButton_SetInactiveColor(ulong aInstance, Math.vec4 aColor);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UITreeViewNode_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UITreeViewNode_SetText(ulong aInstance, string aValue);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UITreeViewNode_SetTextColor(ulong aInstance, Math.vec4 aColor);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UITreeViewNode_SetIcon(ulong aInstance, ulong aIcon);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UITreeViewNode_SetIndicator(ulong aInstance, ulong aIcon);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UITreeViewNode_Add(ulong aInstance);


        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UITreeView_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UITreeView_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UITreeView_SetIndent(ulong aInstance, float aIndent);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UITreeView_SetIconSpacing(ulong aInstance, float aSpacing);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UITreeView_Add(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIVec2Input_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIVec2Input_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIVec2Input_OnChanged(ulong aInstance, IntPtr aDelegate);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIVec2Input_SetValue(ulong aInstance, Math.vec2 aValue);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIVec2Input_SetResetValues(ulong aInstance, Math.vec2 aValue);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static Math.vec2 UIVec2Input_GetValue(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIVec2Input_SetFormat(ulong aInstance, string aFormat);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIVec3Input_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIVec3Input_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIVec3Input_OnChanged(ulong aInstance, IntPtr aDelegate);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIVec3Input_SetValue(ulong aInstance, Math.vec2 aValue);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIVec3Input_SetResetValues(ulong aInstance, Math.vec2 aValue);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static Math.vec2 UIVec3Input_GetValue(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIVec3Input_SetFormat(ulong aInstance, string aFormat);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIVec4Input_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIVec4Input_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIVec4Input_OnChanged(ulong aInstance, IntPtr aDelegate);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIVec4Input_SetValue(ulong aInstance, Math.vec2 aValue);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIVec4Input_SetResetValues(ulong aInstance, Math.vec2 aValue);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static Math.vec2 UIVec4Input_GetValue(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIVec4Input_SetFormat(ulong aInstance, string aFormat);


        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIWorkspaceDocument_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIWorkspaceDocument_RegisterSaveDelegate(ulong aInstance, DocumentSaveDelegate aDelegate);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIWorkspaceDocument_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIWorkspaceDocument_SetContent(ulong aInstance, ulong aContent);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIWorkspaceDocument_SetName(ulong aInstance, string aName);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIWorkspaceDocument_Update(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static bool UIWorkspaceDocument_IsDirty(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIWorkspaceDocument_MarkAsDirty(ulong aInstance, bool aDirty);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIWorkspaceDocument_Open(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIWorkspaceDocument_ForceClose(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIWorkspace_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIWorkspace_RegisterCloseDocumentDelegate(ulong aInstance, DocumentCloseDelegate aDelegate);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIWorkspace_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIWorkspace_Add(ulong aInstance, ulong aDocument);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIBoxLayout_CreateWithOrientation(eBoxLayoutOrientation aOrientation);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIBoxLayout_AddAlignedNonFixed(ulong aInstance, ulong aChild, bool aExpand, bool aFill, eHorizontalAlignment aHAlignment, eVerticalAlignment aVAlignment);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIBoxLayout_AddNonAlignedNonFixed(ulong aInstance, ulong aChild, bool aExpand, bool aFill);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIBoxLayout_AddAlignedFixed(ulong aInstance, ulong aChild, float aFixedSize, bool aExpand, bool aFill, eHorizontalAlignment aHAlignment, eVerticalAlignment aVAlignment);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIBoxLayout_AddNonAlignedFixed(ulong aInstance, ulong aChild, float aFixedSize, bool aExpand, bool aFill);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIBoxLayout_AddSeparator(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIBoxLayout_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIBoxLayout_SetItemSpacing(ulong aInstance, float aItemSpacing);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIBoxLayout_Clear(ulong aInstance);


        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIContainer_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIContainer_Destroy(ulong aInstance);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIContainer_SetContent(ulong aInstance, ulong aChild);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UISplitter_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UISplitter_CreateWithOrientation(eBoxLayoutOrientation aOrientation);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UISplitter_Add1(ulong aInstance, ulong aChild);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UISplitter_Add2(ulong aInstance, ulong aChild);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UISplitter_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UISplitter_SetItemSpacing(ulong aInstance, float aItemSpacing);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIStackLayout_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIStackLayout_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIStackLayout_Add(ulong aInstance, ulong aChild, string aKey);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIStackLayout_SetCurrent(ulong aInstance, string aKey);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIZLayout_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIZLayout_AddAlignedNonFixed(ulong aInstance, ulong aChild, bool aExpand, bool aFill, eHorizontalAlignment aHAlignment, eVerticalAlignment aVAlignment);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIZLayout_AddNonAlignedNonFixed(ulong aInstance, ulong aChild, bool aExpand, bool aFill);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIZLayout_AddAlignedFixed(ulong aInstance, ulong aChild, SpockEngine.Math.vec2 aSize, SpockEngine.Math.vec2 aPosition, bool aExpand, bool aFill, eHorizontalAlignment aHAlignment, eVerticalAlignment aVAlignment);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIZLayout_AddNonAlignedFixed(ulong aInstance, ulong aChild, SpockEngine.Math.vec2 aSize, SpockEngine.Math.vec2 aPosition, bool aExpand, bool aFill);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIZLayout_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIFileTree_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIFileTree_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIFileTree_Add(ulong aInstance, string aText);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIDialog_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIDialog_CreateWithTitleAndSize(ulong aInstance, string aTitle, Math.vec2 aSize);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIDialog_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIDialog_SetTitle(ulong aInstance, string aTitle);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIDialog_SetSize(ulong aInstance, Math.vec2 aSize);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIDialog_SetContent(ulong aInstance, ulong aContent);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIDialog_Update(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIDialog_Open(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIDialog_Close(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIForm_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIForm_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIForm_SetTitle(ulong aInstance, string aTitle);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIForm_SetContent(ulong aInstance, ulong aContent);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIForm_Update(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIForm_SetSize(ulong aInstance, float aWidth, float aHeight);

    }
}