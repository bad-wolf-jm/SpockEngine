#pragma once

#include "Component.h"

namespace SE::Core
{
    class UIPlot;

    enum class UIPlotAxis : uint8_t
    {
        X1,
        X2,
        X3,

        Y1,
        Y2,
        Y3
    };

    struct sPlotAxisConfiguration
    {
        std::string mTitle;

        UIPlotAxis mAxis         = UIPlotAxis::X1;
        bool       mInUse        = false;
        bool       mShowGrid     = true;
        bool       mOppositeSide = false;
    };

    template <typename _Ty>
    struct sXYPlot
    {
        std::string mLegend;

        UIPlotAxis mXAxis = UIPlotAxis::X1;
        UIPlotAxis mYAxis = UIPlotAxis::Y1;

        std::vector<_Ty> mX;
        std::vector<_Ty> mY;
        int32_t          mOffset = 0;
        int32_t          mStride = 1;

        virtual void Render( UIPlot *aParentPlot ) = 0;
    };

    struct sFloat64LinePlot : public sXYPlot<double>
    {
        void Render( UIPlot *aParentPlot );
    };

    class UIPlot : public UIComponent
    {
      public:
        UIPlot();

        void Add( Ref<sFloat64LinePlot> aPlot );
        void Clear();

        void ConfigureLegend( math::vec2 aLegendPadding, math::vec2 aLegendInnerPadding, math::vec2 aLegendSpacing );

      protected:
        std::vector<Ref<sFloat64LinePlot>> mElements;

        ImPlotLocation mLegendPosition = ImPlotLocation_NorthEast;

        std::array<sPlotAxisConfiguration, 6> mAxisConfiguration;

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