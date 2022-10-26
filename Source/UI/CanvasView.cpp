#include "CanvasView.h"

#include "Core/Logging.h"

#include "UI.h"
#include "Widgets.h"

namespace LTSE::Core::UI
{

    static inline ImVec2 to_ImVec( const math::vec2 &v ) { return ImVec2{ v.x, v.y }; }
    struct CanvasCoordinates
    {
        math::vec2 XAxisBounds;
        math::vec2 YAxisBounds;
    };

    void CanvasView::DrawAxis(
        float a_ViewSize, math::vec2 a_AxisBound, const math::vec2 &from, const math::vec2 &to, float labelAlignment, float sign )
    {
        auto drawList     = ImGui::GetWindowDrawList();
        auto direction    = ( to - from ) / math::length( to - from );
        auto normal       = math::vec2( -direction.y, direction.x );
        auto l_AxisLength = math::length( to - from );

        if( math::dot( direction, direction ) < FLT_EPSILON ) return;

        auto minorSize     = 5.0f;
        auto majorSize     = 10.0f;
        auto labelDistance = 15.0f;

        drawList->AddLine( to_ImVec( from ), to_ImVec( to ), IM_COL32( 255, 255, 255, 255 ) );

        float l_Scale = ( a_AxisBound.y - a_AxisBound.x ) / a_ViewSize;

        auto p = from;
        for( auto d = 0.0f; d <= l_AxisLength * l_Scale; d += MinorUnit * l_Scale, p += direction * MinorUnit / l_Scale )
            drawList->AddLine(
                to_ImVec( p - normal * minorSize ), to_ImVec( p + normal * minorSize ), IM_COL32( 128, 128, 128, 255 ) );

        for( auto d = 0.0f; d <= ( l_AxisLength * l_Scale + MajorUnit ); d += MajorUnit )
        {
            p = from + ( direction * d ) / l_Scale;

            drawList->AddLine(
                to_ImVec( p - normal * majorSize ), to_ImVec( p + normal * majorSize ), IM_COL32( 255, 255, 255, 255 ) );

            if( d == 0.0f ) continue;

            char label[16];
            snprintf( label, 15, "%g", d * sign );
            auto labelSize = ImGui::CalcTextSize( label );

            auto labelPosition    = to_ImVec( p ) + ImVec2( fabsf( normal.x ), fabsf( normal.y ) ) * labelDistance;
            auto labelAlignedSize = ImDot( labelSize, to_ImVec( direction ) );
            labelPosition -= to_ImVec( direction * labelAlignedSize * labelAlignment );
            labelPosition = ImFloor( labelPosition + ImVec2( 0.5f, 0.5f ) );
            drawList->AddText( labelPosition, IM_COL32( 200, 200, 200, 255 ), label );
        }
    }

    void CanvasView::DrawGrid()
    {
        auto       l_DrawList = ImGui::GetWindowDrawList();
        math::vec2 l_Scale    = GetScale();
        for( auto d = XAxisBounds.x; d <= XAxisBounds.y; d += MinorUnit )
            l_DrawList->AddLine(
                ImVec2{ d * l_Scale.x, viewRect.Min.y }, ImVec2{ d * l_Scale.x, viewRect.Max.y }, IM_COL32( 255, 255, 255, 15 ) );

        for( auto d = YAxisBounds.x; d <= YAxisBounds.y; d += MinorUnit )
            l_DrawList->AddLine(
                ImVec2{ viewRect.Min.x, -d * l_Scale.y }, ImVec2{ viewRect.Max.x, -d * l_Scale.y }, IM_COL32( 255, 255, 255, 15 ) );

        // for (auto d = 0.0f; d <= (l_AxisLength * l_Scale + MajorUnit); d += MajorUnit)
        // {
        //     drawList->AddLine(to_ImVec(p - normal * majorSize), to_ImVec(p + normal * majorSize), IM_COL32(255, 255, 255, 255));
        // }
    }

    vec2 CanvasView::GetScale()
    {
        return math::vec2{ ( viewRect.Max.x - viewRect.Min.x ) / ( XAxisBounds.y - XAxisBounds.x ),
            ( viewRect.Max.y - viewRect.Min.y ) / ( YAxisBounds.y - YAxisBounds.x ) };
    }

    void CanvasView::AddRect( math::vec2 a_TopLeft, math::vec2 a_BottomRight, math::vec4 a_Color, float a_Thickness )
    {
        auto       l_DrawList = ImGui::GetWindowDrawList();
        math::vec2 l_Scale    = GetScale();
        a_TopLeft             = a_TopLeft * l_Scale;
        a_BottomRight         = a_BottomRight * l_Scale;
        l_DrawList->AddRect( ImVec2{ a_TopLeft.x, -a_TopLeft.y }, ImVec2{ a_BottomRight.x, -a_BottomRight.y },
            IM_COL32( static_cast<uint8_t>( a_Color.x * 255.0f ), static_cast<uint8_t>( a_Color.y * 255.0f ),
                static_cast<uint8_t>( a_Color.z * 255.0f ), static_cast<uint8_t>( a_Color.w * 255.0f ) ),
            0, 0, a_Thickness );
    }

    void CanvasView::AddFilledRect( math::vec2 a_TopLeft, math::vec2 a_BottomRight, math::vec4 a_Color )
    {
        auto       l_DrawList = ImGui::GetWindowDrawList();
        math::vec2 l_Scale    = GetScale();
        a_TopLeft             = a_TopLeft * l_Scale;
        a_BottomRight         = a_BottomRight * l_Scale;
        l_DrawList->AddRectFilled( ImVec2{ a_TopLeft.x, -a_TopLeft.y }, ImVec2{ a_BottomRight.x, -a_BottomRight.y },
            IM_COL32( static_cast<uint8_t>( a_Color.x * 255.0f ), static_cast<uint8_t>( a_Color.y * 255.0f ),
                static_cast<uint8_t>( a_Color.z * 255.0f ), static_cast<uint8_t>( a_Color.w * 255.0f ) ),
            0, 0 );
    }

    void CanvasView::AddImageRect(
        ImageHandle a_Image, math::vec2 a_TopLeft, math::vec2 a_BottomRight, math::vec2 a_TopLeftUV, math::vec2 a_BottomRightUV )
    {
        auto       l_DrawList = ImGui::GetWindowDrawList();
        math::vec2 l_Scale    = GetScale();
        a_TopLeft             = a_TopLeft * l_Scale;
        a_BottomRight         = a_BottomRight * l_Scale;
        l_DrawList->AddImage( (ImTextureID)a_Image.Handle->GetVkDescriptorSet(), ImVec2{ a_TopLeft.x, -a_TopLeft.y },
            ImVec2{ a_BottomRight.x, -a_BottomRight.y }, ImVec2{ a_TopLeftUV.x, a_TopLeftUV.y },
            ImVec2{ a_BottomRightUV.x, a_BottomRightUV.y },
            IM_COL32( static_cast<uint8_t>( 255.0f ), static_cast<uint8_t>( 255.0f ), static_cast<uint8_t>( 255.0f ),
                static_cast<uint8_t>( 255.0f ) ) );
    }

    void CanvasView::Display( std::function<void( CanvasView * )> a_DisplayElements )
    {
        auto canvasRect = canvas.Rect();
        // viewRect   = canvas.ViewRect();
        viewOrigin = canvas.ViewOrigin();

        ImVec2       canvas_p0 = ImGui::GetCursorScreenPos();
        ImVec2       canvas_sz = ImGui::GetContentRegionAvail();
        static float viewScale = 1.0;
        // static CanvasCoordinates l_CanvasAxes = { { -100.0f, 100.0f }, { -80.0f, 80.0f } };

        math::vec2 l_Scale = GetScale();
        ImVec2     l_View  = ImVec2{ -XAxisBounds.x * l_Scale.x, ( canvas_sz.y + YAxisBounds.x * l_Scale.y ) };

        canvas.SetView( l_View, viewScale );

        if( canvas.Begin( "##mycanvas", ImVec2( canvas_sz.x, 0.0f ) ) )
        {
            auto draw_list = ImGui::GetWindowDrawList();

            if( ImGui::IsItemHovered() && ImGui::IsMouseClicked( ImGuiMouseButton_Left ) )
            {
                auto l_MousePos = ImGui::GetMousePos();
            }
            else if( ( m_IsDragging || ImGui::IsItemHovered() ) && ImGui::IsMouseDragging( ImGuiMouseButton_Right, 0.0f ) )
            {
                if( !m_IsDragging )
                {
                    m_IsDragging   = true;
                    drawStartPoint = viewOrigin;
                }
                canvas.SetView( drawStartPoint + ImGui::GetMouseDragDelta( 1, 0.0f ) * viewScale, viewScale );
            }
            else if( ( m_IsDragging || ImGui::IsItemHovered() ) && ImGui::IsMouseDragging( ImGuiMouseButton_Left, 0.0f ) )
            {
                auto l_MousePos = ImGui::GetMousePos();
            }
            else if( m_IsDragging )
            {
                m_IsDragging = false;
            }

            auto l_ViewRect = canvas.ViewRect();
            viewRect        = l_ViewRect;

            DrawGrid();
            if( viewRect.Max.x > 0.0f )
                DrawAxis( ( l_ViewRect.Max.x - l_ViewRect.Min.x ), XAxisBounds, math::vec2( 0.0f, 0.0f ),
                    math::vec2( viewRect.Max.x, 0.0f ), .5f );
            if( viewRect.Min.x < 0.0f )
                DrawAxis( ( l_ViewRect.Max.x - l_ViewRect.Min.x ), XAxisBounds, math::vec2( 0.0f, 0.0f ),
                    math::vec2( viewRect.Min.x, 0.0f ), .5f, -1.0f );
            if( viewRect.Max.y > 0.0f )
                DrawAxis( ( l_ViewRect.Max.y - l_ViewRect.Min.y ), YAxisBounds, math::vec2( 0.0f, 0.0f ),
                    math::vec2( 0.0f, viewRect.Max.y ), .5f, -1.0f );
            if( viewRect.Min.y < 0.0f )
                DrawAxis( ( l_ViewRect.Max.y - l_ViewRect.Min.y ), YAxisBounds, math::vec2( 0.0f, 0.0f ),
                    math::vec2( 0.0f, viewRect.Min.y ), .5f );

            a_DisplayElements( this );

            panelRect.Min = ImGui::GetItemRectMin();
            panelRect.Max = ImGui::GetItemRectMax();

            canvas.End();
        }
        // });
    }

} // namespace LTSE::Core::UI