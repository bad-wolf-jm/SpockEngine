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
        protected ulong mInstance;
        public ulong Instance { get { return mInstance; } }

        public UIPlotData() { mInstance = 0; }
        public UIPlotData(ulong aInstance) { mInstance = aInstance; }

        private string mLegend;

        private Math.vec4 mColor;

        private eUIPlotAxis mXAxis;

        private eUIPlotAxis mYAxis;
        private float mThickness;

        public float Thickness
        {
            get { return mThickness; }
            set { mThickness = value; UIPlotData_SetThickness(mInstance, value); }
        }

        public string Legend
        {
            get { return mLegend; }
            set { mLegend = value; UIPlotData_SetLegend(mInstance, value); }
        }

        public Math.vec4 Color
        {
            get { return mColor; }
            set { mColor = value; UIPlotData_SetColor(mInstance, value); }
        }

        public eUIPlotAxis XAxis
        {
            get { return mXAxis; }
            set { mXAxis = value; UIPlotData_SetXAxis(mInstance, value); }
        }

        public eUIPlotAxis YAxis
        {
            get { return mYAxis; }
            set { mYAxis = value; UIPlotData_SetYAxis(mInstance, value); }
        }

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
    }

    public class UIVLinePlot : UIPlotData
    {
        public UIVLinePlot() : base(UIVLinePlot_Create()) { }

        public UIVLinePlot(ulong aInstance) : base(aInstance) { }

        ~UIVLinePlot() { UIVLinePlot_Destroy(mInstance); }

        private double[] mX;
        public double[] X
        {
            get { return mX; }
            set { mX = value; UIVLinePlot_SetX(mInstance, value); }
        }

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIVLinePlot_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIVLinePlot_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIVLinePlot_SetX(ulong aInstance, double[] aValues);
    }

    public class UIFloat64LinePlot : UIPlotData
    {
        public UIFloat64LinePlot() : base(UIFloat64LinePlot_Create()) { }

        public UIFloat64LinePlot(ulong aInstance) : base(aInstance) { }

        ~UIFloat64LinePlot() { UIFloat64LinePlot_Destroy(mInstance); }

        private double[] mX;

        private double[] mY;

        public double[] X
        {
            get { return mX; }
            set { mX = value; UIFloat64LinePlot_SetX(mInstance, value); }
        }

        public double[] Y
        {
            get { return mY; }
            set { mY = value; UIFloat64LinePlot_SetY(mInstance, value); }
        }

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIFloat64LinePlot_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIFloat64LinePlot_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIFloat64LinePlot_SetX(ulong aInstance, double[] aValues);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIFloat64LinePlot_SetY(ulong aInstance, double[] aValues);
    }

    public class UIFloat64ScatterPlot : UIPlotData
    {
        public UIFloat64ScatterPlot() : base(UIFloat64ScatterPlot_Create()) { }

        public UIFloat64ScatterPlot(ulong aInstance) : base(aInstance) { }

        ~UIFloat64ScatterPlot() { UIFloat64ScatterPlot_Destroy(mInstance); }

        private double[] mX;

        private double[] mY;

        public double[] X
        {
            get { return mX; }
            set { mX = value; UIFloat64ScatterPlot_SetX(mInstance, value); }
        }

        public double[] Y
        {
            get { return mY; }
            set { mY = value; UIFloat64ScatterPlot_SetY(mInstance, value); }
        }

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIFloat64ScatterPlot_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIFloat64ScatterPlot_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIFloat64ScatterPlot_SetX(ulong aInstance, double[] aValues);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIFloat64ScatterPlot_SetY(ulong aInstance, double[] aValues);
    }

    public class UIPlotAxis
    {
        private ulong mPlotInstance;
        private eUIPlotAxis mAxis;

        public UIPlotAxis(ulong aPlotInstance, eUIPlotAxis aAxis)
        {
            mPlotInstance = aPlotInstance;
            mAxis = aAxis;
        }

        public string Title 
        {
            get { return UIPlot_GetAxisTitle(mPlotInstance, mAxis);}
            set { UIPlot_SetAxisTitle(mPlotInstance, mAxis, value);}
        }

        public void SetLimits(double aMin, double aMax)
        {
            UIPlot_SetAxisLimits(mPlotInstance, mAxis, aMin, aMax);
        }

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIPlot_SetAxisLimits(ulong aInstance, eUIPlotAxis aAxis, double aMin, double aMax);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static string UIPlot_GetAxisTitle(ulong aInstance, eUIPlotAxis aAxis);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIPlot_SetAxisTitle(ulong aInstance, eUIPlotAxis aAxis, string aTitle);
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

        public UIPlot() : base(UIPlot_Create())
        {
            mPlots = new List<UIPlotData>();

            x1 = new UIPlotAxis(mInstance, eUIPlotAxis.X1);
            x2 = new UIPlotAxis(mInstance, eUIPlotAxis.X2);
            x3 = new UIPlotAxis(mInstance, eUIPlotAxis.X3);
            y1 = new UIPlotAxis(mInstance, eUIPlotAxis.Y1);
            y2 = new UIPlotAxis(mInstance, eUIPlotAxis.Y2);
            y3 = new UIPlotAxis(mInstance, eUIPlotAxis.Y3);
        }

        public UIPlot(ulong aSelf) : base(aSelf)
        {
            mPlots = new List<UIPlotData>();

            x1 = new UIPlotAxis(aSelf, eUIPlotAxis.X1);
            x2 = new UIPlotAxis(aSelf, eUIPlotAxis.X2);
            x3 = new UIPlotAxis(aSelf, eUIPlotAxis.X3);
            y1 = new UIPlotAxis(aSelf, eUIPlotAxis.Y1);
            y2 = new UIPlotAxis(aSelf, eUIPlotAxis.Y2);
            y3 = new UIPlotAxis(aSelf, eUIPlotAxis.Y3);
        }

        ~UIPlot() { UIPlot_Destroy(mInstance); }

        public void Clear()
        {
            mPlots.Clear();

            UIPlot_Clear(mInstance);
        }

        public void ConfigureLegend(Math.vec2 aLegendPadding, Math.vec2 aLegendInnerPadding, Math.vec2 aLegendSpacing)
        {
            UIPlot_ConfigureLegend(mInstance, aLegendPadding, aLegendInnerPadding, aLegendSpacing);
        }

        public UIFloat64LinePlot Plot(double[] aX, double[] aY, string aLegend = "", float aThickness = 1.0f, Math.vec4? aColor = null)
        {
            var lNewPlot = new UIFloat64LinePlot();
            lNewPlot.X = aX;
            lNewPlot.Y = aY;
            lNewPlot.Legend = aLegend;
            lNewPlot.Thickness = aThickness;
            lNewPlot.Color = aColor.HasValue ? aColor.Value : new Math.vec4(0.0f, 0.0f, 0.0f, -1.0f);

            mPlots.Add(lNewPlot);

            UIPlot_Add(mInstance, lNewPlot.Instance);

            return lNewPlot;
        }

        public UIFloat64ScatterPlot PlotScatter(double[] aX, double[] aY, string aLegend = "", Math.vec4? aColor = null)
        {
            var lNewPlot = new UIFloat64ScatterPlot();
            lNewPlot.X = aX;
            lNewPlot.Y = aY;
            lNewPlot.Legend = aLegend;
            lNewPlot.Color = aColor.HasValue ? aColor.Value : new Math.vec4(0.0f, 0.0f, 0.0f, -1.0f);

            mPlots.Add(lNewPlot);

            UIPlot_Add(mInstance, lNewPlot.Instance);

            return lNewPlot;
        }

        public UIVLinePlot VLines(double[] aX, string aLegend = "", Math.vec4? aColor = null)
        {
            var lNewPlot = new UIVLinePlot();
            lNewPlot.X = aX;
            lNewPlot.Legend = aLegend;
            lNewPlot.Color = aColor.HasValue ? aColor.Value : new Math.vec4(0.0f, 0.0f, 0.0f, -1.0f);

            mPlots.Add(lNewPlot);

            UIPlot_Add(mInstance, lNewPlot.Instance);

            return lNewPlot;
        }

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
    }
}
