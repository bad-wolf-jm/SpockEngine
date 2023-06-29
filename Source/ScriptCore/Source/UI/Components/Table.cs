using System;
using System.Linq;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

using Math = SpockEngine.Math;

namespace SpockEngine
{
    public class UITableColumn : UIComponent
    {
        public UITableColumn() { mInstance = IntPtr.Zero; }
        public UITableColumn(IntPtr aInstance) { mInstance = aInstance; }

        public void SetHeader(string aHeader)
        {
            Interop.UITableColumn_SetHeader(mInstance, aHeader);
        }

        public void SetInitialSize(float aValue)
        {
            Interop.UITableColumn_SetInitialSize(mInstance, aValue);
        }

        private UIComponent[] mToolTips;
        private void SetTooltip(IntPtr[] aTooltips)
        {
            Interop.UITableColumn_SetTooltip(mInstance, aTooltips, aTooltips.Length);
        }

        public void SetTooltip(UIComponent[] aTooltips)
        {
            mToolTips = aTooltips;

            var lElements = aTooltips.Select(i => i.Instance).ToArray();
            SetTooltip(lElements);
        }

        public void SetTooltip(IEnumerable<UIComponent> aTooltips)
        {
            mToolTips = aTooltips.ToArray();

            var lElements = aTooltips.Select(i => i.Instance).ToArray();
            SetTooltip(lElements);
        }

        public void SetBackgroundColor(Math.vec4[] aBackgroundColor)
        {
            Interop.UITableColumn_SetBackgroundColor(mInstance, aBackgroundColor, aBackgroundColor.Length);
        }

        public void SetBackgroundColor(IEnumerable<Math.vec4> aBackroundColor)
        {
            SetBackgroundColor(aBackroundColor.ToArray());
        }

        public void SetForegroundColor(Math.vec4[] aForegroundColor)
        {
            Interop.UITableColumn_SetForegroundColor(mInstance, aForegroundColor, aForegroundColor.Length);
        }

        public void SetForegroundColor(IEnumerable<Math.vec4> aForegroundColor)
        {
            SetForegroundColor(aForegroundColor.ToArray());
        }
    }

    // public class UIFloat64Column : UITableColumn
    // {
    //     public UIFloat64Column() : base(Interop.UIFloat64Column_Create()) { }
    //     public UIFloat64Column(string aHeader, float aInitialSize, string aFormat, string aNaNFormat)
    //         : this()
    //     {
    //         SetHeader(aHeader);
    //         SetInitialSize(aInitialSize);
    //         SetFormat(aFormat);
    //         SetNanFormat(aNaNFormat);
    //     }

    //     ~UIFloat64Column() { Interop.UIFloat64Column_Destroy(mInstance); }

    //     public void Clear()
    //     {
    //         Interop.UIFloat64Column_Clear(mInstance);
    //     }

    //     public void SetFormat(string aValue)
    //     {
    //         Interop.UIFloat64Column_SetFormat(mInstance, aValue);
    //     }

    //     public void SetNanFormat(string aValue)
    //     {
    //         Interop.UIFloat64Column_SetNanFormat(mInstance, aValue);
    //     }

    //     public void SetData(double[] aValue)
    //     {
    //         Interop.UIFloat64Column_SetData(mInstance, aValue, aValue.Length);
    //     }

    //     public void SetData(List<double> aValue)
    //     {
    //         SetData(aValue.ToArray());
    //     }

    //     public void SetData(double[] aValue, Math.vec4[] aForegroundColor)
    //     {
    //         SetData(aValue);
    //         SetForegroundColor(aForegroundColor);
    //     }

    //     public void SetData(List<double> aValue, List<Math.vec4> aForegroundColor)
    //     {
    //         SetData(aValue);
    //         SetForegroundColor(aForegroundColor);
    //     }

    //     public void SetData(double[] aValue, Math.vec4[] aForegroundColor, Math.vec4[] aBackroundColor)
    //     {
    //         SetData(aValue);
    //         SetForegroundColor(aForegroundColor);
    //         SetBackgroundColor(aBackroundColor);
    //     }

    //     public void SetData(List<double> aValue, List<Math.vec4> aForegroundColor, List<Math.vec4> aBackroundColor)
    //     {
    //         SetData(aValue);
    //         SetForegroundColor(aForegroundColor);
    //         SetBackgroundColor(aBackroundColor);
    //     }

    // }

    // public class UIUint32Column : UITableColumn
    // {
    //     public UIUint32Column() : base(Interop.UIUint32Column_Create()) { }
    //     public UIUint32Column(string aHeader, float aInitialSize) : this()
    //     {
    //         SetHeader(aHeader);
    //         SetInitialSize(aInitialSize);
    //     }

    //     ~UIUint32Column() { Interop.UIUint32Column_Destroy(mInstance); }

    //     public void Clear()
    //     {
    //         Interop.UIUint32Column_Clear(mInstance);
    //     }

    //     public void SetData(uint[] aValue)
    //     {
    //         Interop.UIUint32Column_SetData(mInstance, aValue, aValue.Length);
    //     }
    //     public void SetData(List<uint> aValue)
    //     {
    //         SetData(aValue.ToArray());
    //     }

    //     public void SetData(uint[] aValue, Math.vec4[] aForegroundColor)
    //     {
    //         SetData(aValue);
    //         SetForegroundColor(aForegroundColor);
    //     }
    //     public void SetData(List<uint> aValue, List<Math.vec4> aForegroundColor)
    //     {
    //         SetData(aValue);
    //         SetForegroundColor(aForegroundColor);
    //     }

    //     public void SetData(uint[] aValue, Math.vec4[] aForegroundColor, Math.vec4[] aBackroundColor)
    //     {
    //         SetData(aValue);
    //         SetForegroundColor(aForegroundColor);
    //         SetBackgroundColor(aBackroundColor);
    //     }

    //     public void SetData(List<uint> aValue, List<Math.vec4> aForegroundColor, List<Math.vec4> aBackroundColor)
    //     {
    //         SetData(aValue);
    //         SetForegroundColor(aForegroundColor);
    //         SetBackgroundColor(aBackroundColor);
    //     }

    // }

    public class UIStringColumn : UITableColumn
    {
        public UIStringColumn() : base(Interop.UIStringColumn_Create()) { }
        public UIStringColumn(string aHeader, float aInitialSize) : this()
        {
            SetHeader(aHeader);
            SetInitialSize(aInitialSize);
            SetAlignment(eHorizontalAlignment.LEFT, eVerticalAlignment.CENTER);
        }

        ~UIStringColumn() { Interop.UIStringColumn_Destroy(mInstance); }

        public void Clear()
        {
            Interop.UIStringColumn_Clear(mInstance);
        }

        public void SetData(string[] aValue)
        {
            Interop.UIStringColumn_SetData(mInstance, aValue, aValue.Length);
        }

        public void SetData(IEnumerable<string> aValue)
        {
            SetData(aValue.ToArray());
        }

        public void SetData(string[] aValue, Math.vec4[] aForegroundColor)
        {
            SetData(aValue);
            SetForegroundColor(aForegroundColor);
        }

        public void SetData(IEnumerable<string> aValue, IEnumerable<Math.vec4> aForegroundColor)
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

        public void SetData(IEnumerable<string> aValue, IEnumerable<Math.vec4> aForegroundColor, IEnumerable<Math.vec4> aBackroundColor)
        {
            SetData(aValue);
            SetForegroundColor(aForegroundColor);
            SetBackgroundColor(aBackroundColor);
        }

    }

    public class UITable : UIComponent
    {
        public UITable() : base(Interop.UITable_Create()) { }

        ~UITable() { Interop.UITable_Destroy(mInstance); }

        public void SetRowHeight(float aRowHeight) { Interop.UITable_SetRowHeight(mInstance, aRowHeight); }

        public void AddColumn(UITableColumn aColumn) { Interop.UITable_AddColumn(mInstance, aColumn.Instance); }

        public void SetDisplayedRowIndices(int[] aIndices)
        {
            Interop.UITable_SetDisplayedRowIndices(mInstance, aIndices, aIndices.Length);
        }

        public void SetDisplayedRowIndices(List<int> aIndices)
        {
            SetDisplayedRowIndices(aIndices.ToArray());
        }

        public void SetRowBackgroundColor(Math.vec4[] aColors)
        {
            Interop.UITable_SetRowBackgroundColor(mInstance, aColors, aColors.Length);
        }

        public void SetRowBackgroundColor(List<Math.vec4> aColors)
        {
            SetRowBackgroundColor(aColors.ToArray());
        }

        public void ClearRowBackgroundColors()
        {
            Interop.UITable_ClearRowBackgroundColor(mInstance);
        }

        public delegate void RowClickedDelegate(int aRow);
        RowClickedDelegate onRowClickedDelegate;
        public void OnRowClicked(RowClickedDelegate aHandler)
        {
            onRowClickedDelegate = aHandler;

            Interop.UITable_OnRowClicked(mInstance, Marshal.GetFunctionPointerForDelegate(onRowClickedDelegate));
        }

    }
}
