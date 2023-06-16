using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

namespace SpockEngine
{

    public enum eUIPlotAxis
    {
        X1,
        X2,
        X3,

        Y1,
        Y2,
        Y3
    };

    public class UIPlotData
    {
        protected IntPtr mInstance;
        public IntPtr Instance { get { return mInstance; } }

        public UIPlotData() { mInstance = IntPtr.Zero; }
        public UIPlotData(IntPtr aInstance) { mInstance = aInstance; }

        private string mLegend;

        private Math.vec4 mColor;

        private eUIPlotAxis mXAxis;

        private eUIPlotAxis mYAxis;
        private float mThickness;

        public float Thickness
        {
            get { return mThickness; }
            set { mThickness = value; Interop.UIPlotData_SetThickness(mInstance, value); }
        }

        public string Legend
        {
            get { return mLegend; }
            set { mLegend = value; Interop.UIPlotData_SetLegend(mInstance, value); }
        }

        public Math.vec4 Color
        {
            get { return mColor; }
            set { mColor = value; Interop.UIPlotData_SetColor(mInstance, value); }
        }

        public eUIPlotAxis XAxis
        {
            get { return mXAxis; }
            set { mXAxis = value; Interop.UIPlotData_SetXAxis(mInstance, value); }
        }

        public eUIPlotAxis YAxis
        {
            get { return mYAxis; }
            set { mYAxis = value; Interop.UIPlotData_SetYAxis(mInstance, value); }
        }
    }

    public class UIVLinePlot : UIPlotData
    {
        public UIVLinePlot() : base(Interop.UIVLinePlot_Create()) { }

        public UIVLinePlot(IntPtr aInstance) : base(aInstance) { }

        ~UIVLinePlot() { Interop.UIVLinePlot_Destroy(mInstance); }

        private double[] mX;
        public double[] X
        {
            get { return mX; }
            set { mX = value; Interop.UIVLinePlot_SetX(mInstance, value, value.Length); }
        }

    }

    public class UIHLinePlot : UIPlotData
    {
        public UIHLinePlot() : base(Interop.UIHLinePlot_Create()) { }

        public UIHLinePlot(IntPtr aInstance) : base(aInstance) { }

        ~UIHLinePlot() { Interop.UIHLinePlot_Destroy(mInstance); }

        private double[] mY;
        public double[] Y
        {
            get { return mY; }
            set { mY = value; Interop.UIHLinePlot_SetY(mInstance, value, value.Length); }
        }

    }

    public class UIAxisTag : UIPlotData
    {
        public UIAxisTag() : base(Interop.UIAxisTag_Create()) { }
        public UIAxisTag(eUIPlotAxis aAxis, double aX, string aText, Math.vec4 aColor) : this()
        {
            Text = aText;
            Color = aColor;
        }

        public UIAxisTag(IntPtr aInstance) : base(aInstance) { }

        ~UIAxisTag() { Interop.UIAxisTag_Destroy(mInstance); }

        private double mX;
        public double Y
        {
            get { return mX; }
            set { mX = value; Interop.UIAxisTag_SetX(mInstance, value); }
        }

        private string mText;
        public string Text
        {
            get { return mText; }
            set { mText = value; Interop.UIAxisTag_SetText(mInstance, value); }
        }

        new public Math.vec4 Color
        {
            get { return Interop.UIAxisTag_GetColor(mInstance); }
            set { Interop.UIAxisTag_SetColor(mInstance, value); }
        }

        new public eUIPlotAxis Axis
        {
            get { return Interop.UIAxisTag_GetAxis(mInstance); }
            set { Interop.UIAxisTag_SetAxis(mInstance, value); }
        }

    }

    public class UIVRange : UIPlotData
    {
        public UIVRange() : base(Interop.UIVRangePlot_Create()) { }

        public UIVRange(IntPtr aInstance) : base(aInstance) { }

        ~UIVRange() { Interop.UIVRangePlot_Destroy(mInstance); }

        public double X0
        {
            get { return Interop.UIVRangePlot_GetMin(mInstance); }
            set { Interop.UIVRangePlot_SetMin(mInstance, value); }
        }

        public double X1
        {
            get { return Interop.UIVRangePlot_GetMax(mInstance); }
            set { Interop.UIVRangePlot_SetMax(mInstance, value); }
        }
    }

    public class UIHRange : UIPlotData
    {
        public UIHRange() : base(Interop.UIHRangePlot_Create()) { }

        public UIHRange(IntPtr aInstance) : base(aInstance) { }

        ~UIHRange() { Interop.UIHRangePlot_Destroy(mInstance); }

        public double Y0
        {
            get { return Interop.UIHRangePlot_GetMin(mInstance); }
            set { Interop.UIHRangePlot_SetMin(mInstance, value); }
        }

        public double Y1
        {
            get { return Interop.UIHRangePlot_GetMax(mInstance); }
            set { Interop.UIHRangePlot_SetMax(mInstance, value); }
        }
    }

    public class UIFloat64LinePlot : UIPlotData
    {
        public UIFloat64LinePlot() : base(Interop.UIFloat64LinePlot_Create()) { }

        public UIFloat64LinePlot(IntPtr aInstance) : base(aInstance) { }

        ~UIFloat64LinePlot() { Interop.UIFloat64LinePlot_Destroy(mInstance); }

        private double[] mX;

        private double[] mY;

        public double[] X
        {
            get { return mX; }
            set { mX = value; Interop.UIFloat64LinePlot_SetX(mInstance, value, value.Length); }
        }

        public double[] Y
        {
            get { return mY; }
            set { mY = value; Interop.UIFloat64LinePlot_SetY(mInstance, value, value.Length); }
        }
    }

    public class UIFloat64ScatterPlot : UIPlotData
    {
        public UIFloat64ScatterPlot() : base(Interop.UIFloat64ScatterPlot_Create()) { }

        public UIFloat64ScatterPlot(IntPtr aInstance) : base(aInstance) { }

        ~UIFloat64ScatterPlot() { Interop.UIFloat64ScatterPlot_Destroy(mInstance); }

        private double[] mX;

        private double[] mY;

        public double[] X
        {
            get { return mX; }
            set { mX = value; Interop.UIFloat64ScatterPlot_SetX(mInstance, value, value.Length); }
        }

        public double[] Y
        {
            get { return mY; }
            set { mY = value; Interop.UIFloat64ScatterPlot_SetY(mInstance, value, value.Length); }
        }
    }

    public class UIPlotAxis
    {
        private IntPtr mPlotInstance;
        private eUIPlotAxis mAxis;

        public UIPlotAxis(IntPtr aPlotInstance, eUIPlotAxis aAxis)
        {
            mPlotInstance = aPlotInstance;
            mAxis = aAxis;
        }

        public string Title
        {
            get { return Interop.UIPlot_GetAxisTitle(mPlotInstance, mAxis); }
            set { Interop.UIPlot_SetAxisTitle(mPlotInstance, mAxis, value); }
        }

        public void SetLimits(double aMin, double aMax)
        {
            Interop.UIPlot_SetAxisLimits(mPlotInstance, mAxis, aMin, aMax);
        }
    }



    public class UIPlot : UIComponent
    {
        List<UIPlotData> mPlots;

        private UIPlotAxis x1;
        public UIPlotAxis X1 { get { return x1; } }

        private UIPlotAxis x2;
        public UIPlotAxis X2 { get { return x2; } }

        private UIPlotAxis x3;
        public UIPlotAxis X3 { get { return x3; } }

        private UIPlotAxis y1;
        public UIPlotAxis Y1 { get { return y1; } }

        private UIPlotAxis y2;
        public UIPlotAxis Y2 { get { return y2; } }

        private UIPlotAxis y3;
        public UIPlotAxis Y3 { get { return y3; } }

        public UIPlot() : base(Interop.UIPlot_Create())
        {
            mPlots = new List<UIPlotData>();

            x1 = new UIPlotAxis(mInstance, eUIPlotAxis.X1);
            x2 = new UIPlotAxis(mInstance, eUIPlotAxis.X2);
            x3 = new UIPlotAxis(mInstance, eUIPlotAxis.X3);
            y1 = new UIPlotAxis(mInstance, eUIPlotAxis.Y1);
            y2 = new UIPlotAxis(mInstance, eUIPlotAxis.Y2);
            y3 = new UIPlotAxis(mInstance, eUIPlotAxis.Y3);
        }

        public UIPlot(IntPtr aSelf) : base(aSelf)
        {
            mPlots = new List<UIPlotData>();

            x1 = new UIPlotAxis(aSelf, eUIPlotAxis.X1);
            x2 = new UIPlotAxis(aSelf, eUIPlotAxis.X2);
            x3 = new UIPlotAxis(aSelf, eUIPlotAxis.X3);
            y1 = new UIPlotAxis(aSelf, eUIPlotAxis.Y1);
            y2 = new UIPlotAxis(aSelf, eUIPlotAxis.Y2);
            y3 = new UIPlotAxis(aSelf, eUIPlotAxis.Y3);
        }

        ~UIPlot() { Interop.UIPlot_Destroy(mInstance); }

        public void Clear()
        {
            mPlots.Clear();

            Interop.UIPlot_Clear(mInstance);
        }

        public void ConfigureLegend(Math.vec2 aLegendPadding, Math.vec2 aLegendInnerPadding, Math.vec2 aLegendSpacing)
        {
            Interop.UIPlot_ConfigureLegend(mInstance, aLegendPadding, aLegendInnerPadding, aLegendSpacing);
        }

        public void Add(UIPlotData aPlot)
        {
            mPlots.Add(aPlot);

            Interop.UIPlot_Add(mInstance, aPlot.Instance);
        }

        public UIFloat64LinePlot Plot(double[] aX, double[] aY, string aLegend = "", float aThickness = 1.0f, Math.vec4? aColor = null)
        {
            var lNewPlot = new UIFloat64LinePlot();
            lNewPlot.X = aX;
            lNewPlot.Y = aY;
            lNewPlot.Legend = aLegend;
            lNewPlot.Thickness = aThickness;
            lNewPlot.Color = aColor.HasValue ? aColor.Value : new Math.vec4(0.0f, 0.0f, 0.0f, -1.0f);

            Add(lNewPlot);

            return lNewPlot;
        }

        public UIFloat64ScatterPlot PlotScatter(double[] aX, double[] aY, string aLegend = "", Math.vec4? aColor = null)
        {
            var lNewPlot = new UIFloat64ScatterPlot();
            lNewPlot.X = aX;
            lNewPlot.Y = aY;
            lNewPlot.Legend = aLegend;
            lNewPlot.Color = aColor.HasValue ? aColor.Value : new Math.vec4(0.0f, 0.0f, 0.0f, -1.0f);

            Add(lNewPlot);

            return lNewPlot;
        }

        public UIVLinePlot VLines(double[] aX, string aLegend = "", Math.vec4? aColor = null)
        {
            var lNewPlot = new UIVLinePlot();
            lNewPlot.X = aX;
            lNewPlot.Legend = aLegend;
            lNewPlot.Color = aColor.HasValue ? aColor.Value : new Math.vec4(0.0f, 0.0f, 0.0f, -1.0f);

            Add(lNewPlot);

            return lNewPlot;
        }

        public UIHLinePlot HLines(double[] aY, string aLegend = "", Math.vec4? aColor = null)
        {
            var lNewPlot = new UIHLinePlot();
            lNewPlot.Y = aY;
            lNewPlot.Legend = aLegend;
            lNewPlot.Color = aColor.HasValue ? aColor.Value : new Math.vec4(0.0f, 0.0f, 0.0f, -1.0f);

            Add(lNewPlot);

            return lNewPlot;
        }

        public UIVRange VRange(double aX0, double aX1, Math.vec4? aColor = null)
        {
            var lNewPlot = new UIVRange();
            lNewPlot.X0 = aX0;
            lNewPlot.X1 = aX1;
            lNewPlot.Color = aColor.HasValue ? aColor.Value : new Math.vec4(0.0f, 0.0f, 0.0f, -1.0f);

            Add(lNewPlot);

            return lNewPlot;
        }
    }
}
