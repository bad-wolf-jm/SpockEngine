#include "Delaunay.h"
#include "DelaunaySource.h"
#include <cstdio>


int main()
{
    /* x0, y0, x1, y1, ... */
    std::vector<double> coords      = { -1, 1, 1, 1, 1, -1, -1, -1 };
    std::vector<math::vec2> coords2 = { math::vec2{ -1, 1 }, math::vec2{ 1, 1 }, math::vec2{ 1, -1 }, math::vec2{ -1, -1 } };

    // triangulation happens here
    delaunator::Delaunator d( coords );

    // triangulation happens here
    LTSE::Core::Delaunator d1( coords2 );

    for( std::size_t i = 0; i < d.triangles.size(); i += 3 )
    {
        printf( "Triangle points: [[%f, %f], [%f, %f], [%f, %f]]\n",
                d.coords[2 * d.triangles[i]],         // tx0
                d.coords[2 * d.triangles[i] + 1],     // ty0
                d.coords[2 * d.triangles[i + 1]],     // tx1
                d.coords[2 * d.triangles[i + 1] + 1], // ty1
                d.coords[2 * d.triangles[i + 2]],     // tx2
                d.coords[2 * d.triangles[i + 2] + 1]  // ty2
        );
    }

    for( std::size_t i = 0; i < d1.triangles.size(); i += 3 )
    {
        printf( "Triangle points (NEW): [[%f, %f], [%f, %f], [%f, %f]]\n",
                d1.Points[d1.triangles[i]].x,     // tx0
                d1.Points[d1.triangles[i]].y,     // ty0
                d1.Points[d1.triangles[i + 1]].x, // tx1
                d1.Points[d1.triangles[i + 1]].y, // ty1
                d1.Points[d1.triangles[i + 2]].x, // tx2
                d1.Points[d1.triangles[i + 2]].y  // ty2
        );
    }
}