#pragma once

#include "Component.h"

namespace SE::Core
{
    class UIPlot;

    enum class UIPlotAxis : int32_t
    {
        X1,
        X2,
        X3,

        Y1,
        Y2,
        Y3
    };

    struct UIPlotAxisConfiguration
    {
        string_t mTitle;
        float    mMin = 0.0f;
        float    mMax = 0.0f;

        UIPlotAxis mAxis            = UIPlotAxis::X1;
        bool       mInUse           = false;
        bool       mShowGrid        = true;
        bool       mOppositeSide    = false;
        bool       mSetLimitRequest = false;
    };

    struct UIPlotData
    {
        string_t   mLegend;
        math::vec4 mColor     = { 0.0f, 0.0f, 0.0f, -1.0f };
        float      mThickness = -1.0f;

        UIPlotAxis mXAxis = UIPlotAxis::X1;
        UIPlotAxis mYAxis = UIPlotAxis::Y1;

        virtual void Render( UIPlot *aParentPlot ) = 0;
    };

    template <typename _Ty>
    struct sXYPlot : public UIPlotData
    {
        vector_t<_Ty> mX;
        vector_t<_Ty> mY;
        int32_t       mOffset = 0;
        int32_t       mStride = 1;
    };

    struct UIFloat64LinePlot : public sXYPlot<double>
    {
        void Render( UIPlot *aParentPlot );
    };

    struct UIFloat64ScatterPlot : public sXYPlot<double>
    {
        void Render( UIPlot *aParentPlot );
    };

    struct UIVLinePlot : public UIPlotData
    {
        vector_t<double> mX;

        UIVLinePlot() = default;

        UIVLinePlot( vector_t<double> const &x )
            : mX{ x }
        {
        }

        void Render( UIPlot *aParentPlot );
    };

    struct UIHLinePlot : public UIPlotData
    {
        vector_t<double> mY;

        UIHLinePlot() = default;

        UIHLinePlot( vector_t<double> const &y )
            : mY{ y }
        {
        }

        void Render( UIPlot *aParentPlot );
    };

    struct UIVRangePlot : public UIPlotData
    {
        double mX0;
        double mX1;

        UIVRangePlot() = default;

        UIVRangePlot( double aX0, double aX1 )
            : mX0{ aX0 }
            , mX1{ aX1 }
        {
        }

        void Render( UIPlot *aParentPlot );
    };

    struct UIHRangePlot : public UIPlotData
    {
        double mY0;
        double mY1;

        UIHRangePlot() = default;

        UIHRangePlot( double aY0, double aY1 )
            : mY0{ aY0 }
            , mY1{ aY1 }
        {
        }

        void Render( UIPlot *aParentPlot );
    };

    struct UIAxisTag : public UIPlotData
    {
        double     mX;
        string_t   mText;
        math::vec4 mColor;
        UIPlotAxis mAxis = UIPlotAxis::X1;

        UIAxisTag() = default;

        UIAxisTag( UIPlotAxis aAxis, double aX, string_t const &aText, math::vec4 aColor )
            : mX{ aX }
            , mText{ aText }
            , mColor{ aColor }
            , mAxis{ aAxis }
        {
        }

        void Render( UIPlot *aParentPlot );
    };

    class UIPlot : public UIComponent
    {
      public:
        UIPlot();

        void Add( ref_t<UIPlotData> aPlot );
        void Add( UIPlotData *aPlot );
        void Clear();

        void ConfigureLegend( math::vec2 aLegendPadding, math::vec2 aLegendInnerPadding, math::vec2 aLegendSpacing );

        std::array<UIPlotAxisConfiguration, 6> mAxisConfiguration;

      protected:
        vector_t<UIPlotData *> mElements;

        ImPlotLocation mLegendPosition = ImPlotLocation_NorthEast;

        ImVec2 mLegendPadding{ 5.0f, 5.0f };
        ImVec2 mLegendInnerPadding{ 5.0f, 5.0f };
        ImVec2 mLegendSpacing{ 2.0f, 2.0f };

      protected:
        void PushStyles();
        void PopStyles();

        ImVec2 RequiredSize();
        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );
    };
} // namespace SE::Core