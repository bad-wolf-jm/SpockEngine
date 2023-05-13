#pragma once

#include <optional>
#include <string>

#include "Core/Math/Types.h"
#include "Core/Types.h"
#include "Renderer/Buffer.h"
#include "Scene3D/VertexData.h"

// #include <Corrade/Containers/std::vector.h>

using namespace math::literals;

struct FOVVisualizer
{
    float    Width    = 60.0_degf;
    float    Height   = 15.0_degf;
    float    Depth    = 1.0f;
    uint32_t Segments = 32;

    FOVVisualizer() = default;
    FOVVisualizer( const FOVVisualizer& other ) { *this = other; };

    FOVVisualizer( float width, float height, float depth, uint32_t segments )
        : Width( width )
        , Height( height )
        , Depth( depth )
        , Segments( segments )
    {
        UpdatePositions();
    }

    uint32_t IndexCount() { return m_Indices.size(); }

    FOVVisualizer& operator=( const FOVVisualizer& other )
    {
        Width    = other.Width;
        Height   = other.Height;
        Depth    = other.Depth;
        Segments = other.Segments;

        // ObjectId;
        SE::Core::Utilities::CopyArray( other.m_Positions, m_Positions );

        // Positions;
        SE::Core::Utilities::CopyArray( other.m_Indices, m_Indices );
        return *this;
    }

    std::vector<SE::Scene::VertexData>& GetVertexData() { return m_VertexData; }
    std::vector<uint32_t>&                GetIndices() { return m_Indices; }
    void                                  UpdatePositions();

  private:
    std::vector<SE::Scene::VertexData> m_VertexData;
    std::vector<uint32_t>                m_Indices;
    std::vector<math::vec3>              m_Positions;
};
