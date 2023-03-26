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

    public class UIPlot : UIComponent
    {
        List<UIPlotData> mPlots;

        public UIPlot() : base(UIPlot_Create())
        {
            mPlots = new List<UIPlotData>();
        }

        public UIPlot(ulong aSelf) : base(aSelf)
        {
            mPlots = new List<UIPlotData>();
        }

        ~UIPlot() { UIPlot_Destroy(mInstance); }

        void CLear()
        {
            mPlots.Clear();

            UIPlot_Clear(mInstance);
        }

        void ConfigureLegend(Math.vec2 aLegendPadding, Math.vec2 aLegendInnerPadding, Math.vec2 aLegendSpacing)
        {
            UIPlot_ConfigureLegend(mInstance, aLegendPadding, aLegendInnerPadding, aLegendSpacing);
        }

        UIFloat64LinePlot Plot(double[] aX, double[] aY, string aLegend = "", Math.vec4? aColor = null)
        {
            var lNewPlot = new UIFloat64LinePlot();
            lNewPlot.X = aX;
            lNewPlot.Y = aY;
            lNewPlot.Legend = aLegend;
            lNewPlot.Color = aColor.HasValue ? aColor.Value : new Math.vec4(0.0f, 0.0f, 0.0f, -1.0f);

            mPlots.Add(lNewPlot);

            UIPlot_Add(mInstance, lNewPlot.Instance);

            return lNewPlot;
        }

        UIVLinePlot VLines(double[] aX, string aLegend = "", Math.vec4? aColor = null)
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
