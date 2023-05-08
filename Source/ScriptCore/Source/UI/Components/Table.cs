using System;
using System.Linq;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

using Math = SpockEngine.Math;

namespace SpockEngine
{
    public class UITableColumn
    {
        protected ulong mInstance;
        public ulong Instance { get { return mInstance; } }

        public UITableColumn() { mInstance = 0; }
        public UITableColumn(ulong aInstance) { mInstance = aInstance; }

        private UIComponent[] mToolTips;
        private void SetTooltip(ulong[] aTooltips)
        {
            UITableColumn_SetTooltip(mInstance, aTooltips);
        }

        public void SetTooltip(UIComponent[] aTooltips)
        {
            mToolTips = aTooltips;

            var lElements = aTooltips.Select(i => i.Instance).ToArray();
            SetTooltip(lElements);
        }

        public void SetTooltip(List<UIComponent> aTooltips)
        {
            mToolTips = aTooltips.ToArray();
            
            var lElements = aTooltips.Select(i => i.Instance).ToArray();
            SetTooltip(lElements);
        }

        public void SetBackgroundColor(Math.vec4[] aBackgroundColor)
        {
            UITableColumn_SetBackgroundColor(mInstance, aBackgroundColor);
        }

        public void SetBackgroundColor(List<Math.vec4> aBackroundColor)
        {
            SetBackgroundColor(aBackroundColor.ToArray());
        }

        public void SetForegroundColor(Math.vec4[] aForegroundColor)
        {
            UITableColumn_SetForegroundColor(mInstance, aForegroundColor);
        }

        public void SetForegroundColor(List<Math.vec4> aForegroundColor)
        {
            SetForegroundColor(aForegroundColor.ToArray());
        }

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UITableColumn_SetTooltip(ulong aInstance, ulong[] aValue);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UITableColumn_SetBackgroundColor(ulong aInstance, Math.vec4[] aForegroundColor);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UITableColumn_SetForegroundColor(ulong aInstance, Math.vec4[] aForegroundColor);
    }

    public class UIFloat64Column : UITableColumn
    {
        public UIFloat64Column() : base(UIFloat64Column_Create()) { }
        public UIFloat64Column(string aHeader, float aInitialSize, string aFormat, string aNaNFormat)
            : base(UIFloat64Column_CreateFull(aHeader, aInitialSize, aFormat, aNaNFormat)) { }

        ~UIFloat64Column() { UIFloat64Column_Destroy(mInstance); }

        public void Clear()
        {
            UIFloat64Column_Clear(mInstance);
        }

        public void SetData(double[] aValue)
        {
            UIFloat64Column_SetData(mInstance, aValue);
        }

        public void SetData(List<double> aValue)
        {
            SetData(aValue.ToArray());
        }

        public void SetData(double[] aValue, Math.vec4[] aForegroundColor)
        {
            SetData(aValue);
            SetForegroundColor(aForegroundColor);
        }

        public void SetData(List<double> aValue, List<Math.vec4> aForegroundColor)
        {
            SetData(aValue);
            SetForegroundColor(aForegroundColor);
        }

        public void SetData(double[] aValue, Math.vec4[] aForegroundColor, Math.vec4[] aBackroundColor)
        {
            SetData(aValue);
            SetForegroundColor(aForegroundColor);
            SetBackgroundColor(aBackroundColor);
        }

        public void SetData(List<double> aValue, List<Math.vec4> aForegroundColor, List<Math.vec4> aBackroundColor)
        {
            SetData(aValue);
            SetForegroundColor(aForegroundColor);
            SetBackgroundColor(aBackroundColor);
        }

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
    }

    public class UIUint32Column : UITableColumn
    {
        public UIUint32Column() : base(UIUint32Column_Create()) { }
        public UIUint32Column(string aHeader, float aInitialSize)
            : base(UIUint32Column_CreateFull(aHeader, aInitialSize)) { }

        ~UIUint32Column() { UIUint32Column_Destroy(mInstance); }

        public void Clear()
        {
            UIUint32Column_Clear(mInstance);
        }

        public void SetData(uint[] aValue)
        {
            UIUint32Column_SetData(mInstance, aValue);
        }
        public void SetData(List<uint> aValue)
        {
            SetData(aValue.ToArray());
        }

        public void SetData(uint[] aValue, Math.vec4[] aForegroundColor)
        {
            SetData(aValue);
            SetForegroundColor(aForegroundColor);
        }
        public void SetData(List<uint> aValue, List<Math.vec4> aForegroundColor)
        {
            SetData(aValue);
            SetForegroundColor(aForegroundColor);
        }

        public void SetData(uint[] aValue, Math.vec4[] aForegroundColor, Math.vec4[] aBackroundColor)
        {
            SetData(aValue);
            SetForegroundColor(aForegroundColor);
            SetBackgroundColor(aBackroundColor);
        }

        public void SetData(List<uint> aValue, List<Math.vec4> aForegroundColor, List<Math.vec4> aBackroundColor)
        {
            SetData(aValue);
            SetForegroundColor(aForegroundColor);
            SetBackgroundColor(aBackroundColor);
        }

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
    }

    public class UIStringColumn : UITableColumn
    {
        public UIStringColumn() : base(UIStringColumn_Create()) { }
        public UIStringColumn(string aHeader, float aInitialSize)
            : base(UIStringColumn_CreateFull(aHeader, aInitialSize)) { }

        ~UIStringColumn() { UIStringColumn_Destroy(mInstance); }

        public void Clear()
        {
            UIStringColumn_Clear(mInstance);
        }

        public void SetData(string[] aValue)
        {
            UIStringColumn_SetData(mInstance, aValue);
        }

        public void SetData(List<string> aValue)
        {
            SetData(aValue.ToArray());
        }

        public void SetData(string[] aValue, Math.vec4[] aForegroundColor)
        {
            SetData(aValue);
            SetForegroundColor(aForegroundColor);
        }

        public void SetData(List<string> aValue, List<Math.vec4> aForegroundColor)
        {
            SetData(aValue);
            SetForegroundColor(aForegroundColor);
        }

        public void SetData(string[] aValue, Math.vec4[] aForegroundColor, Math.vec4[] aBackroundColor)
        {
            SetData(aValue);
            SetForegroundColor(aForegroundColor);
            SetBackgroundColor(aBackroundColor);
        }

        public void SetData(List<string> aValue, List<Math.vec4> aForegroundColor, List<Math.vec4> aBackroundColor)
        {
            SetData(aValue);
            SetForegroundColor(aForegroundColor);
            SetBackgroundColor(aBackroundColor);
        }

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
    }

    public class UITable : UIComponent
    {
        public UITable() : base(UITable_Create()) { }

        ~UITable() { UITable_Destroy(mInstance); }

        public void SetRowHeight(float aRowHeight) { UITable_SetRowHeight(mInstance, aRowHeight); }

        public void AddColumn(UITableColumn aColumn) { UITable_AddColumn(mInstance, aColumn.Instance); }

        public void SetDisplayedRowIndices(int[] aIndices)
        {
            UITable_SetDisplayedRowIndices(mInstance, aIndices);
        }

        public void SetDisplayedRowIndices(List<int> aIndices)
        {
            SetDisplayedRowIndices(aIndices.ToArray());
        }

        public void SetRowBackgroundColor(Math.vec4[] aColors)
        {
            UITable_SetRowBackgroundColor(mInstance, aColors);
        }

        public void SetRowBackgroundColor(List<Math.vec4> aColors)
        {
            SetRowBackgroundColor(aColors.ToArray());
        }

        public void ClearRowBackgroundColors()
        {
            UITable_ClearRowBackgroundColor(mInstance);
        }

        public delegate void RowClickedDelegate(int aRow);
        RowClickedDelegate onRowClickedDelegate;
        public void OnRowClicked(RowClickedDelegate aHandler)
        {
            onRowClickedDelegate = aHandler;

            UITable_OnRowClicked(mInstance, onRowClickedDelegate);
        }

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UITable_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UITable_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UITable_OnRowClicked(ulong aInstance, RowClickedDelegate aHandler);

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
    }
}
