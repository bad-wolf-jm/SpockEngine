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

    struct sPlotAxisConfiguration
    {
        std::string mTitle;
        float       mMin = 0.0f;
        float       mMax = 0.0f;

        UIPlotAxis mAxis         = UIPlotAxis::X1;
        bool       mInUse        = false;
        bool       mShowGrid     = true;
        bool       mOppositeSide = false;
    };

    struct sPlotData
    {
        std::string mLegend;
        math::vec4  mColor     = { 0.0f, 0.0f, 0.0f, -1.0f };
        float       mThickness = -1.0f;

        UIPlotAxis mXAxis = UIPlotAxis::X1;
        UIPlotAxis mYAxis = UIPlotAxis::Y1;

        virtual void Render( UIPlot *aParentPlot ) = 0;

      public:
        static void UIPlotData_SetThickness( void *aSelf, float aThickness );
        static void UIPlotData_SetLegend( void *aSelf, void *aText );
        static void UIPlotData_SetColor( void *aSelf, math::vec4 *aColor );
        static void UIPlotData_SetXAxis( void *aSelf, int aAxis );
        static void UIPlotData_SetYAxis( void *aSelf, int aAxis );
    };

    template <typename _Ty>
    struct sXYPlot : public sPlotData
    {
        std::vector<_Ty> mX;
        std::vector<_Ty> mY;
        int32_t          mOffset = 0;
        int32_t          mStride = 1;
    };

    struct sFloat64LinePlot : public sXYPlot<double>
    {
        void Render( UIPlot *aParentPlot );

        static void *UIFloat64LinePlot_Create();
        static void  UIFloat64LinePlot_Destroy( void *aSelf );
        static void  UIFloat64LinePlot_SetX( void *aSelf, void *aValue );
        static void  UIFloat64LinePlot_SetY( void *aSelf, void *aValue );
    };

    struct sFloat64ScatterPlot : public sXYPlot<double>
    {
        void Render( UIPlot *aParentPlot );

        static void *UIFloat64ScatterPlot_Create();
        static void  UIFloat64ScatterPlot_Destroy( void *aSelf );
        static void  UIFloat64ScatterPlot_SetX( void *aSelf, void *aValue );
        static void  UIFloat64ScatterPlot_SetY( void *aSelf, void *aValue );
    };

    struct sVLine : public sPlotData
    {
        std::vector<double> mX;

        sVLine() = default;

        sVLine( std::vector<double> const &x )
            : mX{ x }
        {
        }

        void Render( UIPlot *aParentPlot );

        static void *UIVLinePlot_Create();
        static void  UIVLinePlot_Destroy( void *aSelf );
        static void  UIVLinePlot_SetX( void *aSelf, void *aValue );
    };

    struct sHLine : public sPlotData
    {
        std::vector<double> mY;

        sHLine() = default;

        sHLine( std::vector<double> const &y )
            : mY{ y }
        {
        }

        void Render( UIPlot *aParentPlot );

        static void *UIHLinePlot_Create();
        static void  UIHLinePlot_Destroy( void *aSelf );
        static void  UIHLinePlot_SetY( void *aSelf, void *aValue );
    };

    struct sVRange : public sPlotData
    {
        double mX0;
        double mX1;

        sVRange() = default;

        sVRange( double aX0, double aX1 )
            : mX0{ aX0 }
            , mX1{ aX1 }
        {
        }

        void Render( UIPlot *aParentPlot );

        static void  *UIVRAngePlot_Create();
        static void   UIVRAngePlot_Destroy( void *aSelf );
        static double UIVRAngePlot_GetMin( void *aSelf );
        static void   UIVRAngePlot_SetMin( void *aSelf, double aValue );
        static double UIVRAngePlot_GetMax( void *aSelf );
        static void   UIVRAngePlot_SetMax( void *aSelf, double aValue );
    };

    struct sAxisTag : public sPlotData
    {
        double      mX;
        std::string mText;
        math::vec4  mColor;
        UIPlotAxis  mAxis = UIPlotAxis::X1;

        sAxisTag() = default;

        sAxisTag( UIPlotAxis aAxis, double aX, std::string const &aText, math::vec4 aColor )
            : mX{ aX }
            , mText{ aText }
            , mColor{ aColor }
            , mAxis{ aAxis }
        {
        }

        void Render( UIPlot *aParentPlot );

        static void      *UIAxisTag_Create();
        static void      *UIAxisTag_CreateWithTextAndColor( UIPlotAxis aAxis, double aX, void *aText, math::vec4 aColor );
        static void       UIAxisTag_Destroy( void *aSelf );
        static void       UIAxisTag_SetX( void *aSelf, double aValue );
        static void       UIAxisTag_SetText( void *aSelf, void *aText );
        static math::vec4 UIAxisTag_GetColor( void *aSelf );
        static void       UIAxisTag_SetColor( void *aSelf, math::vec4 aColor );
        static int        UIAxisTag_GetAxis( void *aSelf );
        static void       UIAxisTag_SetAxis( void *aSelf, int aAxis );
    };

    class UIPlot : public UIComponent
    {
      public:
        UIPlot();

        void Add( Ref<sPlotData> aPlot );
        void Add( sPlotData *aPlot );
        void Clear();

        void ConfigureLegend( math::vec2 aLegendPadding, math::vec2 aLegendInnerPadding, math::vec2 aLegendSpacing );

        std::array<sPlotAxisConfiguration, 6> mAxisConfiguration;

      protected:
        std::vector<sPlotData *> mElements;

        ImPlotLocation mLegendPosition = ImPlotLocation_NorthEast;

        ImVec2 mLegendPadding{ 5.0f, 5.0f };
        ImVec2 mLegendInnerPadding{ 5.0f, 5.0f };
        ImVec2 mLegendSpacing{ 2.0f, 2.0f };

      protected:
        void PushStyles();
        void PopStyles();

        ImVec2 RequiredSize();
        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );

      public:
        static void *UIPlot_Create();
        static void  UIPlot_Destroy( void *aInstance );
        static void  UIPlot_Clear( void *aInstance );
        static void  UIPlot_ConfigureLegend( void *aInstance, math::vec2 *aLegendPadding, math::vec2 *aLegendInnerPadding,
                                             math::vec2 *aLegendSpacing );
        static void  UIPlot_Add( void *aInstance, void *aPlot );
        static void  UIPlot_SetAxisLimits( void *aInstance, int aAxis, double aMin, double aMax );
        static void *UIPlot_GetAxisTitle( void *aInstance, int aAxis );
        static void  UIPlot_SetAxisTitle( void *aInstance, int aAxis, void *aTitle );
    };
} // namespace SE::Core