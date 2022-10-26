#pragma once

#include "Core/Math/Types.h"
#include "UI.h"
#include <functional>

#include <imgui_canvas.h>

namespace LTSE::Core::UI
{

    using namespace math;

    class CanvasView
    {
      public:
        vec2  XAxisBounds = { -1.0f, 1.0f };
        vec2  YAxisBounds = { -1.0f, 1.0f };
        float MajorUnit   = 0.2f;
        float MinorUnit   = 0.04f;

      public:
        CanvasView() = default;
        CanvasView( std::string m_Title );
        ~CanvasView() = default;

        void Display( std::function<void( CanvasView * )> a_DisplayElements );
        vec2 GetScale();

        void AddImageRect(
            ImageHandle a_Image, math::vec2 a_TopLeft, math::vec2 a_BottomRight, math::vec2 a_TopLeftUV, math::vec2 a_BottomRightUV );
        void AddRect( math::vec2 a_TopLeft, math::vec2 a_BottomRight, math::vec4 a_Color, float a_Thickness );
        void AddFilledRect( math::vec2 a_TopLeft, math::vec2 a_BottomRight, math::vec4 a_Color );

      private:
        void DrawGrid();
        void DrawAxis( float a_ViewSize, math::vec2 a_AxisBound, const math::vec2 &from, const math::vec2 &to, float labelAlignment,
            float sign = 1.0f );

      private:
        std::string     m_Title;
        ImGuiEx::Canvas canvas;
        ImVec2          drawStartPoint;
        ImRect          viewRect;
        ImVec2          viewOrigin;
        ImRect          panelRect;
        bool            m_IsDragging = false;
    };

} // namespace LTSE::Core::UI