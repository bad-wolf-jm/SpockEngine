#include "Arrow.h"

namespace LTSE::Core::Primitives
{

    using namespace math::literals;

    VertexBufferData CreateArrow()
    {
        VertexBufferData l_Shaft    = CreateCylinder( 3, 32 );
        math::mat4 l_ShaftTransform = math::Translation( math::vec3{ 0.0f, 0.46f, 0.0f } ) * math::Scale( math::mat4( 1.0f ), math::vec3{ .0075f, 0.5 * 0.95f, .0075f } );
        for( uint32_t i = 0; i < l_Shaft.Vertices.size(); i++ )
        {
            l_Shaft.Vertices[i].Position = l_ShaftTransform * l_Shaft.Vertices[i].Position;
        }

        VertexBufferData l_Tip    = CreateCone( 32 );
        math::mat4 l_TipTransform = math::Translation( math::vec3{ .0f, .90f, 0.0f } ) * math::Scale( math::mat4( 1.0f ), math::vec3{ .025f, .1f, .025f } );
        for( uint32_t i = 0; i < l_Tip.Vertices.size(); i++ )
        {
            l_Tip.Vertices[i].Position = l_TipTransform * l_Tip.Vertices[i].Position;
        }

        uint32_t l_ShaftOffset = l_Shaft.Vertices.size();
        for( uint32_t i = 0; i < l_Tip.Indices.size(); i++ )
        {
            l_Tip.Indices[i] += l_ShaftOffset;
        }

        VertexBufferData l_Output{};
        l_Output.Vertices.insert( l_Output.Vertices.end(), l_Shaft.Vertices.begin(), l_Shaft.Vertices.end() );
        l_Output.Vertices.insert( l_Output.Vertices.end(), l_Tip.Vertices.begin(), l_Tip.Vertices.end() );
        l_Output.Indices.insert( l_Output.Indices.end(), l_Shaft.Indices.begin(), l_Shaft.Indices.end() );
        l_Output.Indices.insert( l_Output.Indices.end(), l_Tip.Indices.begin(), l_Tip.Indices.end() );

        return l_Output;
    }

} // namespace LTSE::Core::Primitives