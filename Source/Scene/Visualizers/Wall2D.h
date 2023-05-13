#pragma once

#include <optional>
#include <string>

// #include <Magnum/Math/Angle.h>

#include "Core/Math/Types.h"
#include "Core/Types.h"

#include "Scene/Primitives/Primitives.h"

using namespace math::literals;

struct Wall2D
{
    float Width                     = 60.0_degf;
    float Height                    = 15.0_degf;
    float Depth                     = 1.0f;
    uint32_t HorizontalSubdivisions = 16;
    uint32_t VerticalSubdivisions   = 4;
    uint32_t Segments               = 32;

    Wall2D() = default;
    Wall2D( const Wall2D &other ) { *this = other; };
    Wall2D( float width, float height, float depth, uint32_t segments )
        : Width( width )
        , Height( height )
        , Depth( depth )
        , Segments( segments )
    {
        UpdatePositions();
    }

    std::vector<math::vec3> &GetVertexData() { return m_WireframeGrid.Vertices; }
    std::vector<uint32_t> &GetIndices() { return m_WireframeGrid.Indices; }
    void UpdatePositions();

  private:
    SE::Core::Primitives::WireframeVertexBufferData m_WireframeGrid;
};
