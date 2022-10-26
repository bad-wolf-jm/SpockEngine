#include "TestUtils.h"

namespace TestUtils
{
    bool VectorEqual( math::vec2 x, math::vec2 y )
    {
        auto D = y - x;
        return ( math::dot( D, D ) < EPSILON );
    }

    bool VectorEqual( math::vec2 x, math::vec2 y, float e )
    {
        auto D = y - x;
        return ( math::dot( D, D ) < e );
    }

    std::vector<uint8_t> RandomBool( size_t aSize )
    {
        std::random_device              dev;
        std::mt19937                    rng( dev() );
        std::uniform_int_distribution<> dist6( 0, 1 ); // distribution in range [1, 6]

        auto                 gen = [&dist6, &rng]() { return dist6( rng ); };
        std::vector<uint8_t> x( aSize );
        std::generate( x.begin(), x.end(), gen );

        return x;
    }

    uint8_t RandomBool()
    {
        std::random_device              dev;
        std::mt19937                    rng( dev() );
        std::uniform_int_distribution<> dist6( 0, 1 ); // distribution in range [1, 6]

        return dist6( rng );
    }

} // namespace TestUtils
