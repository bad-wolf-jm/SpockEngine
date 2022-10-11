#pragma once

#include "Core/Math/Types.h"
#include <vector>


namespace LTSE::Core
{
    class Delaunator
    {

      public:
        std::vector<math::vec2> const &Points;
        std::vector<uint32_t> Triangles;
        std::vector<uint32_t> HalfEdges;
        std::vector<uint32_t> hull_prev;
        std::vector<uint32_t> hull_next;
        std::vector<uint32_t> hull_tri;
        uint32_t hull_start;

        Delaunator( std::vector<math::vec2> const &in_coords );

        float GetHullArea();

      private:
        std::vector<uint32_t> m_hash;
        math::vec2 mCenter = { 0.0f, 0.0f };

        uint32_t m_hash_size;
        std::vector<uint32_t> m_edge_stack;

        uint32_t Legalize( uint32_t a );
        uint32_t HashKey( float x, float y ) const;
        uint32_t AddTriangle( uint32_t i0, uint32_t i1, uint32_t i2, uint32_t a, uint32_t b, uint32_t c );
        void Link( uint32_t a, uint32_t b );
    };

} // namespace LTSE::Core
