#include "TilePropertyEditor.h"

#include "Developer/UI/CanvasView.h"
#include "Developer/UI/UI.h"
#include "Developer/UI/Widgets.h"

#include "Developer/Scene/Components.h"

#include "TileFlashEditor.h"

using namespace LTSE::Core::EntityComponentSystem::Components;

namespace LTSE::Editor
{

    class SamplerChooser
    {
      public:
        Ref<SensorDeviceBase> SensorModel = nullptr;
        std::string ID                    = "";
        UI::ComboBox<Entity> Dropdown;

      public:
        SamplerChooser() = default;
        SamplerChooser( std::string a_ID )
            : ID{ a_ID }
            , Dropdown{ UI::ComboBox<Entity>( a_ID ) } {};

        ~SamplerChooser() = default;

        Entity GetValue()
        {
            if( Dropdown.Values.size() > 0 )
                return Dropdown.Values[Dropdown.CurrentItem];
            return Entity{};
        }

        void Display( Entity &a_TargetEntity )
        {
            Dropdown.Labels = { "None" };
            Dropdown.Values = { Entity{} };

            uint32_t n = 1;

            SensorModel->mSensorDefinition->ForEach<sSampler>(
                [&]( auto aEntity, auto &aComponent )
                {
                    Dropdown.Labels.push_back( aEntity.Get<sTag>().mValue );
                    Dropdown.Values.push_back( aEntity );

                    if( (uint32_t)aEntity == (uint32_t)a_TargetEntity )
                        Dropdown.CurrentItem = n;
                    n++;
                } );

            Dropdown.Display();

            if( Dropdown.Changed )
            {
                a_TargetEntity = Dropdown.GetValue();
            }
        }
    };

    static bool EditComponent( Ref<SensorDeviceBase> aSensorModel, sTileSpecificationComponent &a_Component )
    {
        auto l_WindowSize  = UI::GetAvailableContentSpace();
        auto l_TextSize0   = ImGui::CalcTextSize( "Field of view hint:" );
        auto l_SliderSize0 = static_cast<float>( l_WindowSize.x ) - l_TextSize0.x - 35.0f;

        UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 5.0f ) );
        UI::Text( "ID: {}", a_Component.mID );

        UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 5.0f ) );
        UI::VectorComponentEditor( "Field of view hint:", 0, a_Component.mPosition, -10.0f, 10.0f, 0.01f, -10.0f, 10.0f, 0.01f, 0.0f, l_TextSize0.x );

        UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 10.0f ) );
        auto l_TextSize5 = ImGui::CalcTextSize( "Sampler:" );
        UI::Text( "Sampler:" );
        UI::SameLine();
        SamplerChooser lSamplerChooser( "##FOO" );
        lSamplerChooser.SensorModel = aSensorModel;
        ImGui::SetNextItemWidth( l_SliderSize0 );
        UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( ( l_TextSize0.x - l_TextSize5.x ) + 10.0f, -5.0f ) );
        lSamplerChooser.Display( Entity{} );

        return false;
    }

    static bool EditComponent( sLaserFlashSpecificationComponent &a_Component )
    {
        auto l_WindowSize  = UI::GetAvailableContentSpace();
        auto l_TextSize0   = ImGui::CalcTextSize( "Field of view hint:" );
        auto l_SliderSize0 = static_cast<float>( l_WindowSize.x ) - l_TextSize0.x - 35.0f;

        UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 10.0f ) );
        UI::Text( "ID: {}", a_Component.mFlashID );

        UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 10.0f ) );
        UI::VectorComponentEditor( "Position:", 0, a_Component.mPosition, -10.0f, 10.0f, 0.01f, -10.0f, 10.0f, 0.01f, 0.0f, l_TextSize0.x );

        UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 5.0f ) );
        UI::VectorComponentEditor( "Extent", 0, a_Component.mExtent, -10.0f, 10.0f, 0.01f, -10.0f, 10.0f, 0.01f, 0.0f, l_TextSize0.x );

        auto l_TextSize6 = ImGui::CalcTextSize( "Laser:" );
        UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 10.0f ) );
        UI::Text( "Laser:" );
        UI::SameLine();
        SamplerChooser lLaserChooser( "##FOO2" );
        ImGui::SetNextItemWidth( l_SliderSize0 );
        UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( ( l_TextSize0.x - l_TextSize6.x ) + 10.0f, -5.0f ) );
        lLaserChooser.Display( Entity{} );

        auto l_TextSize7 = ImGui::CalcTextSize( "Photodetector:" );
        UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 5.0f ) );
        UI::Text( "Photodetector:" );
        UI::SameLine();
        SamplerChooser lPhotodetectorChooser( "##FOO3" );
        ImGui::SetNextItemWidth( l_SliderSize0 );
        UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( ( l_TextSize0.x - l_TextSize7.x ) + 10.0f, -5.0f ) );
        lPhotodetectorChooser.Display( Entity{} );

        auto l_TextSize8 = ImGui::CalcTextSize( "Diffusion:" );
        UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 5.0f ) );
        UI::Text( "Diffusion:" );
        UI::SameLine();
        SamplerChooser lDiffusionChooser( "##FOO4" );
        ImGui::SetNextItemWidth( l_SliderSize0 );
        UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( ( l_TextSize0.x - l_TextSize8.x ) + 10.0f, -5.0f ) );
        lDiffusionChooser.Display( Entity{} );

        auto l_TextSize9 = ImGui::CalcTextSize( "Reduction:" );
        UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 5.0f ) );
        UI::Text( "Reduction:" );
        UI::SameLine();
        SamplerChooser lReductionChooser( "##FOO5" );
        ImGui::SetNextItemWidth( l_SliderSize0 );
        UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( ( l_TextSize0.x - l_TextSize9.x ) + 10.0f, -5.0f ) );
        lReductionChooser.Display( Entity{} );

        return false;
    }

    TilePropertyEditor::TilePropertyEditor() { FlashEditor = FlashAttenuationBindingPopup( "EDIT FLASHES...", { 1600.0f, 1200.0f } ); }

    void TilePropertyEditor::DisplayTileFieldOfView( int32_t width )
    {
#if 0
        float l_Width  = static_cast<float>( width );
        float l_Height = math::min( l_Width / 2.0f, 300.0f );

        auto l_DrawList = ImGui::GetWindowDrawList();

        math::vec2 l_ExtendedFoVPosition{ 0.0f, 0.0f };
        float l_ExtendedFoVRectWidth  = 0.0f;
        float l_ExtendedFoVRectHeight = 0.0f;
        if( TileToEdit.Has<ExtendedFieldOfView>() )
        {
            auto &l_Component = TileToEdit.Get<ExtendedFieldOfView>();
            if( ( l_Component.Value.x > 0.0f ) && ( l_Component.Value.y > 0.0f ) )
            {
                l_ExtendedFoVRectWidth  = l_Component.Value.x;
                l_ExtendedFoVRectHeight = l_Component.Value.y;
                // l_ExtendedFoVPosition   = l_Component.Position;
            }
        }

        float l_FoVRectWidth  = 0.0f;
        float l_FoVRectHeight = 0.0f;
        if( TileToEdit.Has<FieldOfView>() )
        {
            auto &l_Component = TileToEdit.Get<FieldOfView>();
            if( ( l_Component.Value.x > 0.0f ) && ( l_Component.Value.y > 0.0f ) )
            {
                l_FoVRectWidth  = l_Component.Value.x;
                l_FoVRectHeight = l_Component.Value.y;
            }
        }

        math::vec2 l_FoVBoundingBox{ math::max( l_FoVRectWidth, l_ExtendedFoVRectWidth ), math::max( l_FoVRectHeight, l_ExtendedFoVRectHeight ) };
        ImVec2 l_DisplayTopLeft     = ImGui::GetCursorScreenPos();
        ImVec2 l_DisplayBottomRight = l_DisplayTopLeft + ImVec2{ l_Width - 25.0f, l_Height };
        ImVec2 l_Center             = ImVec2{ ( l_DisplayTopLeft.x + l_DisplayBottomRight.x ) / 2.0f, ( l_DisplayTopLeft.y + l_DisplayBottomRight.y ) / 2.0f };
        ImVec2 l_CanvasSize         = ImVec2{ ( l_DisplayBottomRight.x - l_DisplayTopLeft.x ), ( l_DisplayBottomRight.y - l_DisplayTopLeft.y ) };

        ImVec2 l_FoVDisplayHBounds{ math::min( -l_FoVRectWidth, -l_ExtendedFoVRectWidth + l_ExtendedFoVPosition.x ),
                                    math::max( l_FoVRectWidth, l_ExtendedFoVRectWidth + l_ExtendedFoVPosition.x ) };
        ImVec2 l_FoVDisplayVBounds{ math::min( -l_FoVRectHeight, -l_ExtendedFoVRectHeight + l_ExtendedFoVPosition.y ),
                                    math::max( l_FoVRectHeight, l_ExtendedFoVRectHeight + l_ExtendedFoVPosition.y ) };

        float l_AngleScaling  = l_CanvasSize.x / ( l_FoVDisplayHBounds.y - l_FoVDisplayHBounds.x );
        ImVec2 l_FoVBBTopLeft = l_DisplayTopLeft;
        ImVec2 l_FoVBBBottomRight =
            l_FoVBBTopLeft + ImVec2{ ( l_FoVDisplayHBounds.y - l_FoVDisplayHBounds.x ) * l_AngleScaling, ( l_FoVDisplayVBounds.y - l_FoVDisplayVBounds.x ) * l_AngleScaling };

        ImVec2 l_FoVCenter = l_Center;

        l_DrawList->AddRect( l_DisplayTopLeft, l_DisplayBottomRight, IM_COL32( 255, 255, 255, 100 ) );
        l_DrawList->AddRect( l_FoVBBTopLeft, l_FoVBBBottomRight, IM_COL32( 255, 0, 255, 100 ) );

        TileToEdit.IfExists<ExtendedFieldOfView>(
            [&]( auto &l_Component )
            {
                ImVec2 l_TopLeft     = l_Center - ( ImVec2{ l_Component.Value.x, l_Component.Value.y } / 2.0f ) * l_AngleScaling;
                ImVec2 l_BottomRight = l_Center + ( ImVec2{ l_Component.Value.x, l_Component.Value.y } / 2.0f ) * l_AngleScaling;
                l_DrawList->AddRectFilled( l_TopLeft, l_BottomRight, IM_COL32( 255, 0, 255, 100 ) );
                l_DrawList->AddRect( l_TopLeft, l_BottomRight, IM_COL32( 255, 255, 0, 100 ) );
                l_Height = std::max( l_Height, l_BottomRight.y - l_TopLeft.y );

                auto l_HFoV         = fmt::format( "{:.2f}", l_Component.Value.x );
                auto l_HFoVTextSize = ImGui::CalcTextSize( l_HFoV.c_str() );

                auto l_VFoV         = fmt::format( "{:.2f}", l_Component.Value.y );
                auto l_VFoVTextSize = ImGui::CalcTextSize( l_VFoV.c_str() );
                ImVec2 l_HFoVTextPos{ ( l_BottomRight.x + l_TopLeft.x - l_HFoVTextSize.x ) / 2.0f, l_BottomRight.y };
                l_DrawList->AddText( l_HFoVTextPos, IM_COL32_WHITE, l_HFoV.c_str() );

                ImVec2 l_VFoVTextPos{ l_BottomRight.x, ( l_BottomRight.y + l_TopLeft.y - l_VFoVTextSize.y ) / 2.0f };
                l_DrawList->AddText( l_VFoVTextPos, IM_COL32_WHITE, l_VFoV.c_str() );
            } );

        TileToEdit.IfExists<FieldOfView>(
            [&]( auto &l_Component )
            {
                ImVec2 l_CursorScreenPosition = ImGui::GetCursorScreenPos();
                if( ( l_Component.Value.x == 0.0f ) || ( l_Component.Value.y == 0.0f ) )
                    return;

                float l_Aspect     = l_Component.Value.x / l_Component.Value.y;
                float l_RectWidth  = l_Component.Value.x * l_AngleScaling;
                float l_RectHeight = l_Component.Value.y * l_AngleScaling;

                ImVec2 l_TopLeft     = l_Center - ( ImVec2{ l_Component.Value.x, l_Component.Value.y } / 2.0f ) * l_AngleScaling;
                ImVec2 l_BottomRight = l_Center + ( ImVec2{ l_Component.Value.x, l_Component.Value.y } / 2.0f ) * l_AngleScaling;
                l_DrawList->AddRectFilled( l_TopLeft, l_BottomRight, IM_COL32( 16, 0, 255, 220 ) );
                l_DrawList->AddRect( l_TopLeft, l_BottomRight, IM_COL32( 16, 255, 0, 220 ) );
                l_Height = std::max( l_Height, l_BottomRight.y - l_TopLeft.y );

                auto l_HFoV         = fmt::format( "{:.2f}", l_Component.Value.x );
                auto l_HFoVTextSize = ImGui::CalcTextSize( l_HFoV.c_str() );

                auto l_VFoV         = fmt::format( "{:.2f}", l_Component.Value.y );
                auto l_VFoVTextSize = ImGui::CalcTextSize( l_VFoV.c_str() );
                ImVec2 l_HFoVTextPos{ ( l_BottomRight.x + l_TopLeft.x - l_HFoVTextSize.x ) / 2.0f, l_BottomRight.y };
                l_DrawList->AddText( l_HFoVTextPos, IM_COL32_WHITE, l_HFoV.c_str() );

                ImVec2 l_VFoVTextPos{ l_BottomRight.x, ( l_BottomRight.y + l_TopLeft.y - l_VFoVTextSize.y ) / 2.0f };
                l_DrawList->AddText( l_VFoVTextPos, IM_COL32_WHITE, l_VFoV.c_str() );
            } );

        UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( 0.0f, l_Height + 15.0f ) );
#endif
    }

    void TilePropertyEditor::Display( int32_t width, int32_t height )
    {
        auto l_DrawList               = ImGui::GetWindowDrawList();
        auto l_WindowSize             = UI::GetAvailableContentSpace();
        constexpr float lFlashTagSize = 40.0f;
        constexpr float lFlashPadding = 3.0f;

        PropertiesPanel::Display( width, height );
        if( TileToEdit.Has<sTileSpecificationComponent>() )
        {
            auto &l_TileSpecificationComponent = TileToEdit.Get<sTileSpecificationComponent>();
            EditComponent( SensorModel, l_TileSpecificationComponent );

            UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 25.0f ) );
            auto l_TopLeft     = ImGui::GetCursorScreenPos() + ImVec2{ -10.0f, -10.0f };
            auto l_BottomRight = ImGui::GetCursorScreenPos() + ImVec2{ static_cast<float>( l_WindowSize.x ), 25.0f };
            l_DrawList->AddRectFilled( l_TopLeft, l_BottomRight, IM_COL32( 5, 5, 5, 255 ) );
            UI::Text( "FLASHES" );
            UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 0.0f, 10.0f ) );

            auto &lFlashes = TileToEdit.Get<sRelationshipComponent>().mChildren;

            uint32_t lFlasherPerRow = static_cast<uint32_t>( std::floor( ( l_WindowSize.x - 20.0f - ( lFlashes.size() - 1 ) * lFlashPadding ) / lFlashTagSize ) );

            math::vec2 lFlashGroupCursorPosition = UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 10.0f );
            float lLeftMargin                    = lFlashGroupCursorPosition.x;
            uint32_t lCurrentHPosition           = 0;
            math::vec2 lCursorPosition           = lFlashGroupCursorPosition;

            if( UI::Button( "View flashes", { 100.0f, 35.0f } ) )
            {
                FlashEditor.TileToEdit  = TileToEdit;
                FlashEditor.SensorModel = SensorModel;
                FlashEditor.Visible     = true;
            }

            if( FlashEditor.Visible )
                ImGui::OpenPopup( "EDIT FLASHES..." );
            FlashEditor.Display();
        }
    }

    TileLayoutProperties::TileLayoutProperties() { LayoutEditor = TileLayoutEditor( "EDIT TILE LAYOUT...", { 1800.0f, 1000.0f } ); }

    void TileLayoutProperties::Display( int32_t width, int32_t height )
    {
        auto l_WindowSize = UI::GetAvailableContentSpace();
        auto l_TextSize0  = ImGui::CalcTextSize( "Field of view:" );

        auto l_TextSize1 = ImGui::CalcTextSize( "ID:" );
        UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 10.0f ) );
        UI::Text( "ID:" );
        UI::SameLine();
        UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( ( l_TextSize0.x - l_TextSize1.x ) + 10.0f, 0.0f ) );
        UI::Text( "layout_id_goes_here" );

        auto l_TextSize2 = ImGui::CalcTextSize( "Name:" );
        UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 10.0f ) );
        UI::Text( "Name:" );
        UI::SameLine();
        UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( ( l_TextSize0.x - l_TextSize2.x ) + 10.0f, 0.0f ) );
        char buf[128] = { 0 };
        std::strncpy( buf, LayoutToEdit.Get<sTag>().mValue.c_str(), std::min( LayoutToEdit.Get<sTag>().mValue.size(), std::size_t( 128 ) ) );
        if( ImGui::InputText( "##TAG_INPUT", buf, ARRAYSIZE( buf ), ImGuiInputTextFlags_EnterReturnsTrue ) )
        {
            LayoutToEdit.Get<sTag>().mValue = std::string( buf );
        }

        UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 10.0f ) );
        math::vec2 x{0.0f, 0.0f};
        UI::VectorComponentEditor( "Field of view:", 0, x, 0.0, 2.0, 0.01, 0.1, 2.0, 0.01, 0.0, l_TextSize0.x );

        if( UI::Button( "Edit layout", { 100.0f, 35.0f } ) )
        {
            LayoutEditor.LayoutToEdit = LayoutToEdit;
            LayoutEditor.SensorModel  = SensorModel;
            LayoutEditor.Visible      = true;
        }

        if( LayoutEditor.Visible )
            ImGui::OpenPopup( "EDIT TILE LAYOUT..." );
        LayoutEditor.Display();
    }
} // namespace LTSE::Editor