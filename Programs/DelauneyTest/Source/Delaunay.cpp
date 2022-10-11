#include "Delaunay.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>

namespace LTSE::Core
{
    constexpr float EPSILON          = std::numeric_limits<float>::epsilon();
    constexpr uint32_t INVALID_INDEX = std::numeric_limits<uint32_t>::max();

    //@see https://stackoverflow.com/questions/33333363/built-in-mod-vs-custom-mod-function-improve-the-performance-of-modulus-op/33333636#33333636
    inline size_t FastModulo( const size_t i, const size_t c ) { return i >= c ? i % c : i; }

    // Kahan and Babuska summation, Neumaier variant; accumulates less FP error
    inline float Sum( const std::vector<float> &x )
    {
        float sum = x[0];
        float err = 0.0;

        for( size_t i = 1; i < x.size(); i++ )
        {
            const float k = x[i];
            const float m = sum + k;
            err += std::fabs( sum ) >= std::fabs( k ) ? sum - m + k : k - m + sum;
            sum = m;
        }
        return sum + err;
    }

    // inline float Dist( const float ax, const float ay, const float bx, const float by )
    // {
    //     const float dx = ax - bx;
    //     const float dy = ay - by;
    //     return dx * dx + dy * dy;
    // }

    inline float CircumRadius( math::vec2 A, math::vec2 B, math::vec2 C )
    {
        const math::vec2 D = B - A;
        const math::vec2 E = C - A;

        const float bl = math::length2( D );
        const float cl = math::length2( E );
        const float d  = math::det(D, E);

        const float x = ( E.y * bl - D.y * cl ) * 0.5 / d;
        const float y = ( D.x * cl - E.x * bl ) * 0.5 / d;

        if( ( bl > 0.0 || bl < 0.0 ) && ( cl > 0.0 || cl < 0.0 ) && ( d > 0.0 || d < 0.0 ) )
        {
            return x * x + y * y;
        }
        else
        {
            return std::numeric_limits<float>::max();
        }
    }

    inline bool Orient2D( const float px, const float py, const float qx, const float qy, const float rx, const float ry )
    {
        return ( qy - py ) * ( rx - qx ) - ( qx - px ) * ( ry - qy ) < 0.0;
    }

    inline math::vec2 CircumCenter( math::vec2 A, math::vec2 B, math::vec2 C )
    {
        const math::vec2 D = B - A;
        const math::vec2 E = C - A;

        const float bl = math::length2( D );
        const float cl = math::length2( E );
        const float d  = math::det(D, E);

        const float x = ( E.y * bl - D.y * cl ) * 0.5 / d;
        const float y = ( D.x * cl - E.x * bl ) * 0.5 / d;

        return A + math::vec2{ x, y };
    }

    struct compare
    {

        std::vector<math::vec2> const &Points;
        math::vec2 c = { 0.0f, 0.0f };

        bool operator()( uint32_t i, uint32_t j )
        {
            const float d1    = math::dist2( Points[i], c );
            const float d2    = math::dist2( Points[j], c );
            const float diff1 = d1 - d2;
            const float diff2 = Points[i].x - Points[j].x;
            const float diff3 = Points[i].y - Points[j].y;

            if( diff1 > 0.0 || diff1 < 0.0 )
            {
                return diff1 < 0;
            }
            else if( diff2 > 0.0 || diff2 < 0.0 )
            {
                return diff2 < 0;
            }
            else
            {
                return diff3 < 0;
            }
        }
    };

    inline bool InCircle( const float ax, const float ay, const float bx, const float by, const float cx, const float cy, const float px, const float py )
    {
        const float dx = ax - px;
        const float dy = ay - py;
        const float ex = bx - px;
        const float ey = by - py;
        const float fx = cx - px;
        const float fy = cy - py;

        const float ap = dx * dx + dy * dy;
        const float bp = ex * ex + ey * ey;
        const float cp = fx * fx + fy * fy;

        return ( dx * ( ey * cp - bp * fy ) - dy * ( ex * cp - bp * fx ) + ap * ( ex * fy - ey * fx ) ) < 0.0;
    }

    inline bool check_pts_equal( float x1, float y1, float x2, float y2 ) { return std::fabs( x1 - x2 ) <= EPSILON && std::fabs( y1 - y2 ) <= EPSILON; }

    // monotonically increases with real angle, but doesn't need expensive trigonometry
    inline float PseudoAngle( const float dx, const float dy )
    {
        const float p = dx / ( std::abs( dx ) + std::abs( dy ) );
        return ( dy > 0.0 ? 3.0 - p : 1.0 + p ) / 4.0; // [0..1)
    }

    Delaunator::Delaunator( std::vector<math::vec2> const &in_coords )
        : Points( in_coords )
        , Triangles()
        , HalfEdges()
        , hull_prev()
        , hull_next()
        , hull_tri()
        , hull_start()
        , m_hash()
        , m_hash_size()
        , m_edge_stack()
    {
        uint32_t n = Points.size();

        float max_x = std::numeric_limits<float>::min();
        float max_y = std::numeric_limits<float>::min();
        float min_x = std::numeric_limits<float>::max();
        float min_y = std::numeric_limits<float>::max();
        std::vector<uint32_t> ids;
        ids.reserve( n );

        for( uint32_t i = 0; i < n; i++ )
        {
            const float x = Points[i].x;
            const float y = Points[i].y;

            if( x < min_x )
                min_x = x;
            if( y < min_y )
                min_y = y;
            if( x > max_x )
                max_x = x;
            if( y > max_y )
                max_y = y;

            ids.push_back( i );
        }
        const float cx = ( min_x + max_x ) / 2;
        const float cy = ( min_y + max_y ) / 2;

        uint32_t i0 = INVALID_INDEX;
        uint32_t i1 = INVALID_INDEX;
        uint32_t i2 = INVALID_INDEX;

        // pick a seed point close to the centroid
        float min_dist = std::numeric_limits<float>::max();
        for( uint32_t i = 0; i < n; i++ )
        {
            const float d = math::dist2( math::vec2{cx, cy}, Points[i] );
            if( d < min_dist )
            {
                i0       = i;
                min_dist = d;
            }
        }

        math::vec2 P0 = Points[i0];


        // find the point closest to the seed
        min_dist = std::numeric_limits<float>::max();
        for( uint32_t i = 0; i < n; i++ )
        {
            if( i == i0 )
                continue;
            const float d = math::dist2( P0, Points[i] );
            if( d < min_dist && d > 0.0 )
            {
                i1       = i;
                min_dist = d;
            }
        }

        math::vec2 P1 = Points[i1];


        // find the third point which forms the smallest circumcircle with the first two
        float min_radius = std::numeric_limits<float>::max();
        for( uint32_t i = 0; i < n; i++ )
        {
            if( i == i0 || i == i1 )
                continue;

            const float r = CircumRadius( P0, P1, Points[i] );

            if( r < min_radius )
            {
                i2         = i;
                min_radius = r;
            }
        }

        if( !( min_radius < std::numeric_limits<float>::max() ) )
        {
            throw std::runtime_error( "not triangulation" );
        }

        math::vec2 P2 = Points[i2];

        if( Orient2D( P0.x, P0.y, P1.x, P1.y, P2.x, P2.y ) )
        {
            std::swap( i1, i2 );
            std::swap( P1, P2 );
        }

        mCenter = CircumCenter( P0, P1, P2 );

        // sort the points by distance from the seed triangle circumcenter
        std::sort( ids.begin(), ids.end(), compare{ Points, mCenter } );

        // initialize a hash table for storing edges of the advancing convex hull
        m_hash_size = static_cast<uint32_t>( std::llround( std::ceil( std::sqrt( n ) ) ) );
        m_hash.resize( m_hash_size );
        std::fill( m_hash.begin(), m_hash.end(), INVALID_INDEX );

        // initialize arrays for tracking the edges of the advancing convex hull
        hull_prev.resize( n );
        hull_next.resize( n );
        hull_tri.resize( n );

        hull_start = i0;

        size_t hull_size = 3;

        hull_next[i0] = hull_prev[i2] = i1;
        hull_next[i1] = hull_prev[i0] = i2;
        hull_next[i2] = hull_prev[i1] = i0;

        hull_tri[i0] = 0;
        hull_tri[i1] = 1;
        hull_tri[i2] = 2;

        m_hash[HashKey( P0.x, P0.y )] = i0;
        m_hash[HashKey( P1.x, P1.y )] = i1;
        m_hash[HashKey( P2.x, P2.y )] = i2;

        uint32_t max_triangles = n < 3 ? 1 : 2 * n - 5;
        Triangles.reserve( max_triangles * 3 );
        HalfEdges.reserve( max_triangles * 3 );
        AddTriangle( i0, i1, i2, INVALID_INDEX, INVALID_INDEX, INVALID_INDEX );

        float xp = std::numeric_limits<float>::quiet_NaN();
        float yp = std::numeric_limits<float>::quiet_NaN();
        for( uint32_t k = 0; k < n; k++ )
        {
            const uint32_t i = ids[k];
            const float x    = Points[i].x;
            const float y    = Points[i].y;

            // skip near-duplicate points
            if( k > 0 && check_pts_equal( x, y, xp, yp ) )
                continue;
            xp = x;
            yp = y;

            // skip seed triangle points
            if( check_pts_equal( x, y, P0.x, P0.y ) || check_pts_equal( x, y, P1.x, P1.y ) || check_pts_equal( x, y, P2.x, P2.y ) )
                continue;

            // find a visible edge on the convex hull using edge hash
            uint32_t start = 0;

            size_t key = HashKey( x, y );
            for( size_t j = 0; j < m_hash_size; j++ )
            {
                start = m_hash[FastModulo( key + j, m_hash_size )];
                if( start != INVALID_INDEX && start != hull_next[start] )
                    break;
            }

            start    = hull_prev[start];
            size_t e = start;
            size_t q;

            while( q = hull_next[e], !Orient2D( x, y, Points[e].x, Points[e].y, Points[q].x, Points[q].y ) )
            { // TODO: does it works in a same way as in JS
                e = q;
                if( e == start )
                {
                    e = INVALID_INDEX;
                    break;
                }
            }

            if( e == INVALID_INDEX )
                continue; // likely a near-duplicate point; skip it

            // add the first triangle from the point
            uint32_t t = AddTriangle( e, i, hull_next[e], INVALID_INDEX, INVALID_INDEX, hull_tri[e] );

            hull_tri[i] = Legalize( t + 2 );
            hull_tri[e] = t;
            hull_size++;

            // walk forward through the hull, adding more Triangles and flipping recursively
            uint32_t next = hull_next[e];
            while( q = hull_next[next], Orient2D( x, y, Points[next].x, Points[next].y, Points[q].x, Points[q].y ) )
            {
                t               = AddTriangle( next, i, q, hull_tri[i], INVALID_INDEX, hull_tri[next] );
                hull_tri[i]     = Legalize( t + 2 );
                hull_next[next] = next; // mark as removed
                hull_size--;
                next = q;
            }

            // walk backward from the other side, adding more Triangles and flipping
            if( e == start )
            {
                while( q = hull_prev[e], Orient2D( x, y, Points[q].x, Points[q].y, Points[e].x, Points[e].y ) )
                {
                    t = AddTriangle( q, i, e, INVALID_INDEX, hull_tri[e], hull_tri[q] );
                    Legalize( t + 2 );
                    hull_tri[q]  = t;
                    hull_next[e] = e; // mark as removed
                    hull_size--;
                    e = q;
                }
            }

            // update the hull indices
            hull_prev[i]    = e;
            hull_start      = e;
            hull_prev[next] = i;
            hull_next[e]    = i;
            hull_next[i]    = next;

            m_hash[HashKey( x, y )]                     = i;
            m_hash[HashKey( Points[e].x, Points[e].y )] = e;
        }
    }

    float Delaunator::GetHullArea()
    {
        std::vector<float> hull_area;
        size_t e = hull_start;
        do
        {
            hull_area.push_back( ( Points[e].x - Points[hull_prev[e]].x ) * ( Points[e].y + Points[hull_prev[e]].y ) );
            e = hull_next[e];
        } while( e != hull_start );
        return Sum( hull_area );
    }

    uint32_t Delaunator::Legalize( uint32_t a )
    {
        uint32_t i  = 0;
        uint32_t ar = 0;
        m_edge_stack.clear();

        // recursion eliminated with a fixed-size stack
        while( true )
        {
            const size_t b = HalfEdges[a];

            /* if the pair of Triangles doesn't satisfy the Delaunay condition
             * (p1 is inside the circumcircle of [p0, pl, pr]), flip them,
             * then do the same check/flip recursively for the new pair of Triangles
             *
             *           pl                    pl
             *          /||\                  /  \
             *       al/ || \bl            al/    \a
             *        /  ||  \              /      \
             *       /  a||b  \    flip    /___ar___\
             *     p0\   ||   /p1   =>   p0\---bl---/p1
             *        \  ||  /              \      /
             *       ar\ || /br             b\    /br
             *          \||/                  \  /
             *           pr                    pr
             */
            const size_t a0 = 3 * ( a / 3 );
            ar              = a0 + ( a + 2 ) % 3;

            if( b == INVALID_INDEX )
            {
                if( i > 0 )
                {
                    i--;
                    a = m_edge_stack[i];
                    continue;
                }
                else
                {
                    // i = INVALID_INDEX;
                    break;
                }
            }

            const size_t b0 = 3 * ( b / 3 );
            const size_t al = a0 + ( a + 1 ) % 3;
            const size_t bl = b0 + ( b + 2 ) % 3;

            const uint32_t p0 = Triangles[ar];
            const uint32_t pr = Triangles[a];
            const uint32_t pl = Triangles[al];
            const uint32_t p1 = Triangles[bl];

            const bool illegal = InCircle( Points[p0].x, Points[p0].y, Points[pr].x, Points[pr].y, Points[pl].x, Points[pl].y, Points[p1].x, Points[p1].y );

            if( illegal )
            {
                Triangles[a] = p1;
                Triangles[b] = p0;

                auto hbl = HalfEdges[bl];

                // edge swapped on the other side of the hull (rare); fix the halfedge reference
                if( hbl == INVALID_INDEX )
                {
                    uint32_t e = hull_start;
                    do
                    {
                        if( hull_tri[e] == bl )
                        {
                            hull_tri[e] = a;
                            break;
                        }
                        e = hull_next[e];
                    } while( e != hull_start );
                }
                Link( a, hbl );
                Link( b, HalfEdges[ar] );
                Link( ar, bl );
                uint32_t br = b0 + ( b + 1 ) % 3;

                if( i < m_edge_stack.size() )
                {
                    m_edge_stack[i] = br;
                }
                else
                {
                    m_edge_stack.push_back( br );
                }
                i++;
            }
            else
            {
                if( i > 0 )
                {
                    i--;
                    a = m_edge_stack[i];
                    continue;
                }
                else
                {
                    break;
                }
            }
        }
        return ar;
    }

    inline uint32_t Delaunator::HashKey( const float x, const float y ) const
    {
        const float dx = x - mCenter.x;
        const float dy = y - mCenter.y;
        return FastModulo( static_cast<uint32_t>( std::llround( std::floor( PseudoAngle( dx, dy ) * static_cast<float>( m_hash_size ) ) ) ), m_hash_size );
    }

    uint32_t Delaunator::AddTriangle( uint32_t i0, uint32_t i1, uint32_t i2, uint32_t a, uint32_t b, uint32_t c )
    {
        uint32_t t = Triangles.size();
        Triangles.push_back( i0 );
        Triangles.push_back( i1 );
        Triangles.push_back( i2 );
        Link( t, a );
        Link( t + 1, b );
        Link( t + 2, c );
        return t;
    }

    void Delaunator::Link( const uint32_t a, const uint32_t b )
    {
        uint32_t s = HalfEdges.size();
        if( a == s )
        {
            HalfEdges.push_back( b );
        }
        else if( a < s )
        {
            HalfEdges[a] = b;
        }
        else
        {
            throw std::runtime_error( "Cannot link edge" );
        }
        if( b != INVALID_INDEX )
        {
            uint32_t s2 = HalfEdges.size();
            if( b == s2 )
            {
                HalfEdges.push_back( a );
            }
            else if( b < s2 )
            {
                HalfEdges[b] = a;
            }
            else
            {
                throw std::runtime_error( "Cannot link edge" );
            }
        }
    }
} // namespace LTSE::Core