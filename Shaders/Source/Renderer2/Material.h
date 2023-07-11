#if defined __cplusplus
#    include "Common/Definitions.h"
#endif

struct MaterialInput
{
    float4 mBaseColor;
};

void InitializeMaterial(out MaterialInput aMaterial)
{
    aMaterial.mBaseColor = float4(1.0);
}