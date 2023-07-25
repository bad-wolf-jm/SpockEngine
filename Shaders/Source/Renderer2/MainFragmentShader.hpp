#if defined( __cplusplus )
#    include "Common/Definitions.h"
#    include "Material.hpp"
#    include "Varying.hpp"
#endif

LAYOUT_LOCATION( 0 ) __SHADER_OUTPUT__ float4 outColor;

#if defined( __cplusplus )
void material( out MaterialInput aMaterial )
{
}
#endif

void main()
{
    MaterialInputs lMaterial;
    InitializeMaterial( lMaterial );

    // material( lMaterial );

#if defined( MATERIAL_HAS_EMISSIVE )
    outColor = tonemap( lMaterial.mBaseColor + lMaterial.mEmissive );
#else
    outColor = lMaterial.mBaseColor;
#endif
}