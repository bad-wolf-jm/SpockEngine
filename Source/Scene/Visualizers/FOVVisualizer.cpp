#include "FOVVisualizer.h"
// #include <Corrade/Containers/GrowableArray.h>

// using float = Magnum::Math::Rad<float>;

void FOVVisualizer::UpdatePositions()
{
    uint32_t   _positionCount = 0;
    uint32_t   _indexCount    = 0;
    math::vec4 _l             = { 0.0f, Depth, 0.0f, 1 };
    float      _Left          = ( -Width / 2 );
    float      _Right         = ( Width / 2 );
    float      _Top           = ( Height / 2 );
    float      _Bottom        = ( -Height / 2 );
    float      _SegmentStep   = ( Width / (float)Segments );

    math::vec3 _Origin      = { 0.0f, 0.0f, 0.0f };
    math::vec3 _TopLeft     = math::vec3( math::Rotation( _Top, math::x_axis() ) * math::Rotation( _Left, math::z_axis() ) * _l );
    math::vec3 _TopRight    = math::vec3( math::Rotation( _Top, math::x_axis() ) * math::Rotation( _Right, math::z_axis() ) * _l );
    math::vec3 _BottomLeft  = math::vec3( math::Rotation( _Bottom, math::x_axis() ) * math::Rotation( _Left, math::z_axis() ) * _l );
    math::vec3 _BottomRight = math::vec3( math::Rotation( _Bottom, math::x_axis() ) * math::Rotation( _Right, math::z_axis() ) * _l );

    _positionCount = 1 + 2 * ( Segments * 2 + 1 );
    _indexCount    = 12 + 2 * ( Segments * 2 );

    m_Positions.resize( _positionCount );
    m_Indices.resize( _indexCount );
    // Corrade::Containers::arrayResize(m_Positions, _positionCount);
    // Corrade::Containers::arrayResize(m_Indices, _indexCount);

    m_Positions[0] = _Origin;
    m_Positions[1] = _TopLeft;
    m_Positions[2] = _TopRight;
    m_Positions[3] = _BottomLeft;
    m_Positions[4] = _BottomRight;
    uint32_t a[]   = { 0, 1, 0, 2, 0, 3, 0, 4, 1, 3, 2, 4 };
    for( int i = 0; i < 12; i++ )
        m_Indices[i] = a[i];

    if( Segments == 1 )
    {
        uint32_t a[] = { 1, 2, 3, 4 };
        for( int i = 12; i < 16; i++ )
            m_Indices[i] = a[i - 12];
        return;
    }

    math::mat4 _TopRotation    = math::Rotation( _Top, math::x_axis() );
    math::mat4 _BottomRotation = math::Rotation( _Bottom, math::x_axis() );
    math::mat4 _StepRotation   = math::Rotation( _SegmentStep, math::z_axis() );
    math::vec4 _TopSegment     = ( math::Rotation( _Left, math::z_axis() ) * _l );
    math::vec4 _BottomSegment  = ( math::Rotation( _Left, math::z_axis() ) * _l );

    uint32_t _currentIndex    = 12;
    uint32_t _currentPosition = 5;

    m_Positions[_currentPosition] = math::vec3( ( _TopRotation * _TopSegment ) );
    m_Indices[_currentIndex++]    = _currentPosition++;
    for( int i = 0; i < Segments; i++ )
    {
        _TopSegment                   = _StepRotation * _TopSegment;
        m_Positions[_currentPosition] = math::vec3( ( _TopRotation * _TopSegment ) );
        m_Indices[_currentIndex++]    = _currentPosition;
        if( i < Segments - 1 )
        {
            m_Indices[_currentIndex++] = _currentPosition++;
        }
        else
        {
            _currentPosition++;
        }
    }

    m_Positions[_currentPosition] = math::vec3( _BottomRotation * _BottomSegment );
    m_Indices[_currentIndex++]    = _currentPosition++;
    for( int i = 0; i < Segments; i++ )
    {
        _BottomSegment                = _StepRotation * _BottomSegment;
        m_Positions[_currentPosition] = math::vec3( _BottomRotation * _BottomSegment );
        m_Indices[_currentIndex++]    = _currentPosition;
        if( i < Segments - 1 )
        {
            m_Indices[_currentIndex++] = _currentPosition++;
        }
        else
        {
            _currentPosition++;
        }
    }

    // Corrade::Containers::arrayResize(m_VertexData, m_Positions.size());
    m_VertexData.resize( m_Positions.size() );
    for( uint32_t i = 0; i < m_Positions.size(); i++ )
    {
        SE::Scene::VertexData &l_Vertex = m_VertexData[i];
        l_Vertex.Position               = m_Positions[i];
    }
}
