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

#if defined( __cplusplus )
#    include "ShadingModelLit.hpp"
#endif

void main()
{
    MaterialInputs lMaterial;
    InitializeMaterial( lMaterial );

    material( lMaterial );

    float4 lColor = EvaluateMaterial( lMaterial );

    outColor = lColor;
}