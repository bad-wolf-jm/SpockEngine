using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace SpockEngine
{
    public static class Interop
    {
        const string SE_RUNTIME = "LTSimulationEngineRuntime.dll";

        #region UIBaseImage
        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIBaseImage_Create();

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIBaseImage_CreateWithPath([MarshalAs(UnmanagedType.LPTStr)] string aText, Math.vec2 Size);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIBaseImage_Destroy(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIBaseImage_SetImage(IntPtr aSelf, [MarshalAs(UnmanagedType.LPTStr)] string aPath);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIBaseImage_SetSize(IntPtr aSelf, Math.vec2 aSize);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static Math.vec2 UIBaseImage_GetSize(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIBaseImage_SetTopLeft(IntPtr aSelf, Math.vec2 aTopLeft);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static Math.vec2 UIBaseImage_GetTopLeft(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIBaseImage_SetBottomRight(IntPtr aSelf, Math.vec2 aBottomRight);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static Math.vec2 UIBaseImage_GetBottomRight(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIBaseImage_SetTintColor(IntPtr aSelf, Math.vec4 aColor);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static Math.vec4 UIBaseImage_GetTintColor(IntPtr aSelf);
        #endregion

        #region UIButton
        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIButton_Create();

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIButton_CreateWithText([MarshalAs(UnmanagedType.LPTStr)] string aText);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIButton_Destroy(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIButton_SetText(IntPtr aSelf, [MarshalAs(UnmanagedType.LPTStr)] string aText);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIButton_OnClick(IntPtr aSelf, IntPtr aDelegate);
        #endregion

        #region UICheckbox
        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UICheckBox_Create();

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UICheckBox_Destroy(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UICheckBox_OnClick(IntPtr aSelf, IntPtr aHandler);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static bool UICheckBox_IsChecked(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UICheckBox_SetIsChecked(IntPtr aSelf, bool aValue);
        #endregion

        #region UIColorButton
        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIColorButton_Create();

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIColorButton_Destroy(IntPtr aSelf);
        #endregion

        #region UIComboBox
        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIComboBox_Create();

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIComboBox_CreateWithItems(string[] aItems, int aLength);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIComboBox_Destroy(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static int UIComboBox_GetCurrent(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIComboBox_SetCurrent(IntPtr aSelf, int aValue);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIComboBox_SetItemList(IntPtr aSelf, string[] aItems, int aLength);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIComboBox_OnChanged(IntPtr aSelf, IntPtr aHandler);
        #endregion

        #region UIComponent
        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIComponent_SetIsVisible(IntPtr aSelf, bool aIsVisible);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIComponent_SetIsEnabled(IntPtr aSelf, bool aIsEnabled);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIComponent_SetAllowDragDrop(IntPtr aSelf, bool aAllowDragDrop);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIComponent_SetPaddingAll(IntPtr aSelf, float aPaddingAll);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIComponent_SetPaddingPairs(IntPtr aSelf, float aPaddingTopBottom, float aPaddingLeftRight);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIComponent_SetPaddingIndividual(IntPtr aSelf, float aPaddingTop, float aPaddingBottom, float aPaddingLeft, float aPaddingRight);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIComponent_SetAlignment(IntPtr aSelf, eHorizontalAlignment aHAlignment, eVerticalAlignment aVAlignment);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIComponent_SetHorizontalAlignment(IntPtr aSelf, eHorizontalAlignment aAlignment);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIComponent_SetVerticalAlignment(IntPtr aSelf, eVerticalAlignment aAlignment);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIComponent_SetBackgroundColor(IntPtr aSelf, Math.vec4 aColor);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIComponent_SetFont(IntPtr aSelf, eFontFamily aFont);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIComponent_SetTooltip(IntPtr aSelf, IntPtr aTooltip);
        #endregion

        #region UIDropdownButton
        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIDropdownButton_Create();

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIDropdownButton_Destroy(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static bool UIDropdownButton_SetContent(IntPtr aSelf, IntPtr aContent);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static bool UIDropdownButton_SetContentSize(IntPtr aSelf, Math.vec2 aSize);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIDropdownButton_SetImage(IntPtr aSelf, IntPtr aImage);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIDropdownButton_SetText(IntPtr aSelf, string aText);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIDropdownButton_SetTextColor(IntPtr aSelf, Math.vec4 aColor);
        #endregion

        #region UIImage
        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIImage_Create();

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIImage_CreateWithPath(string aText, Math.vec2 Size);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIImage_Destroy(IntPtr aSelf);
        #endregion

        #region UIImageButton
        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIImageButton_Create();

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIImageButton_CreateWithPath(string aText, Math.vec2 Size);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIImageButton_Destroy(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIImageButton_OnClick(IntPtr aSelf, IntPtr aText);
        #endregion

        #region UIImageToggleButton
        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIImageToggleButton_Create();

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIImageToggleButton_Destroy(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIImageToggleButton_OnClicked(IntPtr aSelf, IntPtr aHandler);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIImageToggleButton_OnChanged(IntPtr aSelf, IntPtr aHandler);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static bool UIImageToggleButton_IsActive(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIImageToggleButton_SetActive(IntPtr aSelf, bool aValue);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIImageToggleButton_SetActiveImage(IntPtr aSelf, IntPtr aImage);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIImageToggleButton_SetInactiveImage(IntPtr aSelf, IntPtr aImage);
        #endregion

        #region UILabel
        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UILabel_Create();

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UILabel_CreateWithText(string aText);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UILabel_Destroy(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UILabel_SetText(IntPtr aSelf, string aText);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UILabel_SetTextColor(IntPtr aSelf, Math.vec4 aText);
        #endregion

        #region UIMenuItem
        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIMenuItem_Create();

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIMenuItem_CreateWithTextAndShortcut(string aText, string aShortcut);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIMenuItem_CreateWithText(string aText);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIMenuItem_Destroy(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIMenuItem_SetText(IntPtr aSelf, string aText);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIMenuItem_SetShortcut(IntPtr aSelf, string aShortcut);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIMenuItem_SetTextColor(IntPtr aSelf, Math.vec4 aColor);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIMenuItem_OnTrigger(IntPtr aSelf, IntPtr aHandler);
        #endregion

        #region UIMenuSeparator
        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIMenuSeparator_Create();

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIMenuSeparator_Destroy(IntPtr aSelf);
        #endregion

        #region UIMenu
        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIMenu_Create();

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIMenu_CreateWithText(string aText);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIMenu_Destroy(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIMenu_AddAction(IntPtr aSelf, string aName, string aShortcut);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIMenu_AddSeparator(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIMenu_AddMenu(IntPtr aSelf, string aName);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIMenu_Update(IntPtr aSelf);
        #endregion

        #region UIPlotData
        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIPlotData_SetThickness(IntPtr aSelf, float aThickness);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIPlotData_SetLegend(IntPtr aSelf, string aText);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIPlotData_SetColor(IntPtr aSelf, Math.vec4 aColor);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIPlotData_SetXAxis(IntPtr aSelf, eUIPlotAxis aAxis);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIPlotData_SetYAxis(IntPtr aSelf, eUIPlotAxis aAxis);
        #endregion

        #region UIVLinePlot
        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIVLinePlot_Create();

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIVLinePlot_Destroy(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIVLinePlot_SetX(IntPtr aSelf, double[] aValues, int aLength);
        #endregion

        #region UIHLinPlot
        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIHLinePlot_Create();

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIHLinePlot_Destroy(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIHLinePlot_SetY(IntPtr aSelf, double[] aValues, int aLength);
        #endregion

        #region UIAxisTag
        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIAxisTag_Create();

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIAxisTag_CreateWithTextAndColor(eUIPlotAxis aAxis, double aX, string aText, Math.vec4 aColor);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIAxisTag_Destroy(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIAxisTag_SetX(IntPtr aSelf, double aValue);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIAxisTag_SetText(IntPtr aSelf, string aText);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static Math.vec4 UIAxisTag_GetColor(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIAxisTag_SetColor(IntPtr aSelf, Math.vec4 aColor);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static eUIPlotAxis UIAxisTag_GetAxis(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIAxisTag_SetAxis(IntPtr aSelf, eUIPlotAxis aColor);
        #endregion

        #region UIVRangePlot
        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIVRangePlot_Create();

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIVRangePlot_Destroy(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static double UIVRangePlot_GetMin(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIVRangePlot_SetMin(IntPtr aSelf, double aValue);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static double UIVRangePlot_GetMax(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIVRangePlot_SetMax(IntPtr aSelf, double aValue);
        #endregion

        #region UIHRangePlot
        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIHRangePlot_Create();

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIHRangePlot_Destroy(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static double UIHRangePlot_GetMin(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIHRangePlot_SetMin(IntPtr aSelf, double aValue);
        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static double UIHRangePlot_GetMax(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIHRangePlot_SetMax(IntPtr aSelf, double aValue);
        #endregion

        #region UIFloat64LinePlot
        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIFloat64LinePlot_Create();

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIFloat64LinePlot_Destroy(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIFloat64LinePlot_SetX(IntPtr aSelf, double[] aValues, int aLength);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIFloat64LinePlot_SetY(IntPtr aSelf, double[] aValues, int aLength);
        #endregion

        #region UIFloatScatterPlot
        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIFloat64ScatterPlot_Create();

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIFloat64ScatterPlot_Destroy(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIFloat64ScatterPlot_SetX(IntPtr aSelf, double[] aValues, int aLength);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIFloat64ScatterPlot_SetY(IntPtr aSelf, double[] aValues, int aLength);
        #endregion

        #region UIPlot
        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIPlot_SetAxisLimits(IntPtr aSelf, eUIPlotAxis aAxis, double aMin, double aMax);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        [return: MarshalAs(UnmanagedType.LPWStr)]
        public extern static string UIPlot_GetAxisTitle(IntPtr aSelf, eUIPlotAxis aAxis);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIPlot_SetAxisTitle(IntPtr aSelf, eUIPlotAxis aAxis, string aTitle);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIPlot_Create();

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIPlot_Destroy(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIPlot_Clear(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIPlot_ConfigureLegend(IntPtr aSelf, Math.vec2 aLegendPadding, Math.vec2 aLegendInnerPadding, Math.vec2 aLegendSpacing);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIPlot_Add(IntPtr aSelf, IntPtr aPlot);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIPlot_PlotVLines(IntPtr aSelf, double[] a, string aLegend, Math.vec4 aColor);
        #endregion

        #region UIProgressBar
        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIProgressBar_Create();

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIProgressBar_Destroy(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIProgressBar_SetText(IntPtr aSelf, string aText);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIProgressBar_SetTextColor(IntPtr aSelf, Math.vec4 aText);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIProgressBar_SetProgressValue(IntPtr aSelf, float aValue);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIProgressBar_SetProgressColor(IntPtr aSelf, Math.vec4 aColor);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIProgressBar_SetThickness(IntPtr aSelf, float aThickness);
        #endregion

        #region UIPropertyValue
        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIPropertyValue_Create();

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIPropertyValue_CreateWithText(string aText);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIPropertyValue_CreateWithTextAndOrientation(string aText, eBoxLayoutOrientation aOrientation);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIPropertyValue_Destroy(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIPropertyValue_SetValue(IntPtr aSelf, string aText);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIPropertyValue_SetValueFont(IntPtr aSelf, eFontFamily aFont);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIPropertyValue_SetNameFont(IntPtr aSelf, eFontFamily aFont);
        #endregion

        #region UISlider
        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UISlider_Create();

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UISlider_Destroy(IntPtr aSelf);
        #endregion

        #region UITableColumn
        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UITableColumn_SetTooltip(IntPtr aSelf, IntPtr[] aValue, int aLength);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UITableColumn_SetBackgroundColor(IntPtr aSelf, Math.vec4[] aForegroundColor, int aLength);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UITableColumn_SetForegroundColor(IntPtr aSelf, Math.vec4[] aForegroundColor, int aLength);
        #endregion

        #region UIFloat64Column
        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIFloat64Column_Create();

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIFloat64Column_CreateFull(string aHeader, float aInitialSize, string aFormat, string aNaNFormat);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIFloat64Column_Destroy(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIFloat64Column_Clear(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIFloat64Column_SetData(IntPtr aSelf, double[] aValue, int aLength);
        #endregion

        #region UIUint32Column
        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIUint32Column_Create();

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIUint32Column_CreateFull(string aHeader, float aInitialSize);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIUint32Column_Destroy(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIUint32Column_Clear(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIUint32Column_SetData(IntPtr aSelf, uint[] aValue, int aLength);
        #endregion

        #region UIStringColumn
        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIStringColumn_Create();

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIStringColumn_CreateFull(string aHeader, float aInitialSize);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIStringColumn_Destroy(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIStringColumn_Clear(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIStringColumn_SetData(IntPtr aSelf, string[] aValue, int aLength);
        #endregion

        #region UITable
        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UITable_Create();

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UITable_Destroy(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UITable_OnRowClicked(IntPtr aSelf, IntPtr aHandler);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UITable_AddColumn(IntPtr aSelf, IntPtr aColumnInstance);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UITable_SetRowHeight(IntPtr aSelf, float aRowHeight);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UITable_SetRowBackgroundColor(IntPtr aSelf, Math.vec4[] aColors, int aLength);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UITable_ClearRowBackgroundColor(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UITable_SetDisplayedRowIndices(IntPtr aSelf, int[] aIndices, int aLength);
        #endregion

        #region UITextInput
        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UITextInput_Create();

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UITextInput_CreateWithText(string aText);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UITextInput_Destroy(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        [return: MarshalAs(UnmanagedType.LPWStr)]
        public extern static string UITextInput_GetText(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UITextInput_OnTextChanged(IntPtr aSelf, IntPtr aHandler);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UITextInput_SetHintText(IntPtr aSelf, string aText);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UITextInput_SetTextColor(IntPtr aSelf, Math.vec4 aColor);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UITextInput_SetBufferSize(IntPtr aSelf, uint aSize);
        #endregion

        #region UITextOverlay
        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UITextOverlay_Create();

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UITextOverlay_Destroy(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UITextOverlay_AddText(IntPtr aSelf, string aText);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UITextOverlay_Clear(IntPtr aSelf);
        #endregion

        #region UITextToggleButton
        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UITextToggleButton_Create();

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UITextToggleButton_CreateWithText(string aText);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UITextToggleButton_Destroy(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UITextToggleButton_OnClicked(IntPtr aSelf, IntPtr aHandler);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UITextToggleButton_OnChanged(IntPtr aSelf, IntPtr aHandler);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static bool UITextToggleButton_IsActive(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UITextToggleButton_SetActive(IntPtr aSelf, bool aValue);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UITextToggleButton_SetActiveColor(IntPtr aSelf, Math.vec4 aColor);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UITextToggleButton_SetInactiveColor(IntPtr aSelf, Math.vec4 aColor);
        #endregion

        #region UITreeViewNode
        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UITreeViewNode_Destroy(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UITreeViewNode_SetText(IntPtr aSelf, string aValue);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UITreeViewNode_SetTextColor(IntPtr aSelf, Math.vec4 aColor);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UITreeViewNode_SetIcon(IntPtr aSelf, IntPtr aIcon);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UITreeViewNode_SetIndicator(IntPtr aSelf, IntPtr aIcon);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UITreeViewNode_Add(IntPtr aSelf);
        #endregion

        #region UITreeView
        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UITreeView_Create();

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UITreeView_Destroy(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UITreeView_SetIndent(IntPtr aSelf, float aIndent);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UITreeView_SetIconSpacing(IntPtr aSelf, float aSpacing);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UITreeView_Add(IntPtr aSelf);
        #endregion

        #region UIVec2Input
        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIVec2Input_Create();

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIVec2Input_Destroy(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIVec2Input_OnChanged(IntPtr aSelf, IntPtr aDelegate);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIVec2Input_SetValue(IntPtr aSelf, Math.vec2 aValue);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIVec2Input_SetResetValues(IntPtr aSelf, Math.vec2 aValue);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static Math.vec2 UIVec2Input_GetValue(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIVec2Input_SetFormat(IntPtr aSelf, string aFormat);
        #endregion

        #region UIVec3Input
        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIVec3Input_Create();

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIVec3Input_Destroy(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIVec3Input_OnChanged(IntPtr aSelf, IntPtr aDelegate);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIVec3Input_SetValue(IntPtr aSelf, Math.vec2 aValue);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIVec3Input_SetResetValues(IntPtr aSelf, Math.vec2 aValue);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static Math.vec2 UIVec3Input_GetValue(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIVec3Input_SetFormat(IntPtr aSelf, string aFormat);
        #endregion

        #region UIVec4Input
        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIVec4Input_Create();

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIVec4Input_Destroy(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIVec4Input_OnChanged(IntPtr aSelf, IntPtr aDelegate);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIVec4Input_SetValue(IntPtr aSelf, Math.vec2 aValue);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIVec4Input_SetResetValues(IntPtr aSelf, Math.vec2 aValue);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static Math.vec2 UIVec4Input_GetValue(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIVec4Input_SetFormat(IntPtr aSelf, string aFormat);
        #endregion

        #region UIWorkspaceDocument
        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIWorkspaceDocument_Create();

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIWorkspaceDocument_RegisterSaveDelegate(IntPtr aSelf, IntPtr aDelegate);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIWorkspaceDocument_Destroy(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIWorkspaceDocument_SetContent(IntPtr aSelf, IntPtr aContent);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIWorkspaceDocument_SetName(IntPtr aSelf, string aName);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIWorkspaceDocument_Update(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static bool UIWorkspaceDocument_IsDirty(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIWorkspaceDocument_MarkAsDirty(IntPtr aSelf, bool aDirty);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIWorkspaceDocument_Open(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIWorkspaceDocument_RequestClose(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIWorkspaceDocument_ForceClose(IntPtr aSelf);
        #endregion

        #region UIWorkspace
        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIWorkspace_Create();

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIWorkspace_RegisterCloseDocumentDelegate(IntPtr aSelf, IntPtr aDelegate);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIWorkspace_Destroy(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIWorkspace_Add(IntPtr aSelf, IntPtr aDocument);
        #endregion

        #region UIBoxLayout
        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIBoxLayout_CreateWithOrientation(eBoxLayoutOrientation aOrientation);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIBoxLayout_AddAlignedNonFixed(IntPtr aSelf, IntPtr aChild, bool aExpand, bool aFill, eHorizontalAlignment aHAlignment, eVerticalAlignment aVAlignment);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIBoxLayout_AddNonAlignedNonFixed(IntPtr aSelf, IntPtr aChild, bool aExpand, bool aFill);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIBoxLayout_AddAlignedFixed(IntPtr aSelf, IntPtr aChild, float aFixedSize, bool aExpand, bool aFill, eHorizontalAlignment aHAlignment, eVerticalAlignment aVAlignment);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIBoxLayout_AddNonAlignedFixed(IntPtr aSelf, IntPtr aChild, float aFixedSize, bool aExpand, bool aFill);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIBoxLayout_AddSeparator(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIBoxLayout_Destroy(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIBoxLayout_SetItemSpacing(IntPtr aSelf, float aItemSpacing);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIBoxLayout_Clear(IntPtr aSelf);
        #endregion

        #region UIOContainer
        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIContainer_Create();

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIContainer_Destroy(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIContainer_SetContent(IntPtr aSelf, IntPtr aChild);
        #endregion

        #region UISplitter
        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UISplitter_Create();

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UISplitter_CreateWithOrientation(eBoxLayoutOrientation aOrientation);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UISplitter_Add1(IntPtr aSelf, IntPtr aChild);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UISplitter_Add2(IntPtr aSelf, IntPtr aChild);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UISplitter_Destroy(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UISplitter_SetItemSpacing(IntPtr aSelf, float aItemSpacing);
        #endregion

        #region UIStackLayout
        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIStackLayout_Create();

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIStackLayout_Destroy(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIStackLayout_Add(IntPtr aSelf, IntPtr aChild, string aKey);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIStackLayout_SetCurrent(IntPtr aSelf, string aKey);
        #endregion

        #region UIZLayout
        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIZLayout_Create();

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIZLayout_AddAlignedNonFixed(IntPtr aSelf, IntPtr aChild, bool aExpand, bool aFill, eHorizontalAlignment aHAlignment, eVerticalAlignment aVAlignment);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIZLayout_AddNonAlignedNonFixed(IntPtr aSelf, IntPtr aChild, bool aExpand, bool aFill);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIZLayout_AddAlignedFixed(IntPtr aSelf, IntPtr aChild, SpockEngine.Math.vec2 aSize, SpockEngine.Math.vec2 aPosition, bool aExpand, bool aFill, eHorizontalAlignment aHAlignment, eVerticalAlignment aVAlignment);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIZLayout_AddNonAlignedFixed(IntPtr aSelf, IntPtr aChild, SpockEngine.Math.vec2 aSize, SpockEngine.Math.vec2 aPosition, bool aExpand, bool aFill);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIZLayout_Destroy(IntPtr aSelf);
        #endregion

        #region UIFileTree
        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIFileTree_Create();

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIFileTree_Destroy(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIFileTree_Add(IntPtr aSelf, string aText);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIFileTree_Remove(IntPtr aSelf, string aText);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIFileTree_OnSelected(IntPtr aSelf, IntPtr aDelegate);
        #endregion

        #region UIDialog
        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIDialog_Create();

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIDialog_CreateWithTitleAndSize(IntPtr aSelf, string aTitle, Math.vec2 aSize);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIDialog_Destroy(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIDialog_SetTitle(IntPtr aSelf, string aTitle);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIDialog_SetSize(IntPtr aSelf, Math.vec2 aSize);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIDialog_SetContent(IntPtr aSelf, IntPtr aContent);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIDialog_Update(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIDialog_Open(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIDialog_Close(IntPtr aSelf);
        #endregion

        #region UIForm
        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIForm_Create();

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIForm_Destroy(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIForm_SetTitle(IntPtr aSelf, string aTitle);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIForm_SetContent(IntPtr aSelf, IntPtr aContent);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIForm_Update(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIForm_SetSize(IntPtr aSelf, float aWidth, float aHeight);
        #endregion

        #region UICodeEditor
        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UICodeEditor_Create();

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UICodeEditor_Destroy(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UICodeEditor_SetText(IntPtr aSelf, string aTitle);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        [return: MarshalAs(UnmanagedType.LPWStr)]
        public extern static string UICodeEditor_GetText(IntPtr aSelf);


        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UICodeEditor_MoveUp(IntPtr aSelf, int aAmount, bool aSelect);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UICodeEditor_MoveDown(IntPtr aSelf, int aAmount, bool aSelect);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UICodeEditor_MoveLeft(IntPtr aSelf, int aAmount, bool aSelect, bool aWordMode);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UICodeEditor_MoveRight(IntPtr aSelf, int aAmount, bool aSelect, bool aWordMode);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UICodeEditor_MoveTop(IntPtr aSelf, bool aSelect);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UICodeEditor_MoveBottom(IntPtr aSelf, bool aSelect);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UICodeEditor_MoveHome(IntPtr aSelf, bool aSelect);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UICodeEditor_MoveEnd(IntPtr aSelf, bool aSelect);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UICodeEditor_SetSelectionStart(IntPtr aSelf, UICodeEditor.Coordinates aPosition);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UICodeEditor_SetSelectionEnd(IntPtr aSelf, UICodeEditor.Coordinates aPosition);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UICodeEditor_SetSelection(IntPtr aSelf, UICodeEditor.Coordinates aStart, UICodeEditor.Coordinates aEnd, int aMode);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UICodeEditor_SelectWordUnderCursor(IntPtr aSelf);
        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]

        public extern static void UICodeEditor_SelectAll(IntPtr aSelf);
        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static bool UICodeEditor_HasSelection(IntPtr aSelf);


        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UICodeEditor_Cut(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]

        public extern static void UICodeEditor_Copy(IntPtr aSelf);
        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]

        public extern static void UICodeEditor_Paste(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UICodeEditor_Delete(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static bool UICodeEditor_CanUndo(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static bool UICodeEditor_CanRedo(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UICodeEditor_Undo(IntPtr aSelf, int aSteps);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UICodeEditor_Redo(IntPtr aSelf, int aSteps);



        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UICodeEditor_InsertText(IntPtr aSelf, string aText);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        [return: MarshalAs(UnmanagedType.LPWStr)]
        public extern static string UICodeEditor_GetSelectedText(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        [return: MarshalAs(UnmanagedType.LPWStr)]
        public extern static string UICodeEditor_GetCurrentLineText(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static bool UICodeEditor_GetReadOnly(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UICodeEditor_SetReadOnly(IntPtr aSelf, bool aValue);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static UICodeEditor.Coordinates UICodeEditor_GetCursorPosition(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UICodeEditor_SetCursorPosition(IntPtr aSelf, UICodeEditor.Coordinates aValue);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static bool UICodeEditor_GetShowWhitespace(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UICodeEditor_SetShowWhitespace(IntPtr aSelf, bool aValue);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static int UICodeEditor_GetTabSize(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UICodeEditor_SetTabSize(IntPtr aSelf, int aValue);

        #endregion

        #region UIMarkdown
        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static IntPtr UIMarkdown_Create();

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIMarkdown_Destroy(IntPtr aSelf);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIMarkdown_SetText(IntPtr aSelf, string aText);

        [DllImport(SE_RUNTIME, CharSet = CharSet.Unicode)]
        public extern static void UIMarkdown_SetTextColor(IntPtr aSelf, Math.vec4 aText);
        #endregion


    }
}