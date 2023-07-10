struct MaterialInputs
{
    vec3  mNormal;
    vec4  mBaseColor;
    float mIsMetal;
    float mRoughness;
    float mOcclusionStrength;
    float mAmbiantOcclusion;
};

void InitializeMaterial(inout MaterialInputs aMaterial)
{
    aMaterial.mNormal            = vec3(0);
    aMaterial.mBaseColor         = vec4(0);
    aMaterial.mIsMetal           = 0.0;
    aMaterial.mRoughness         = 0.0;
    aMaterial.mOcclusionStrength = 0.0;
    aMaterial.mAmbiantOcclusion  = 0.0;
}

vec4 SRGBtoLINEAR( vec4 srgbIn )
{
    vec3 bLess  = step( vec3( 0.04045 ), srgbIn.xyz );
    vec3 linOut = mix( srgbIn.xyz / vec3( 12.92 ), pow( ( srgbIn.xyz + vec3( 0.055 ) ) / vec3( 1.055 ), vec3( 2.4 ) ), bLess );
    return vec4( linOut, srgbIn.w );
}


void Material(inout MaterialInputs aMaterial)
{
    InitializeMaterial(aMaterial);

    sShaderMaterial lMaterial = gMaterials.mArray[material.mMaterialID];

    if( lMaterial.mNormalTextureID == 0 )
        aMaterial.mNormal = normalize( inNormal );
    else
        aMaterial.mNormal = getNormalFromMap( gTextures[lMaterial.mNormalTextureID], inUV0 );

    aMaterial.mBaseColor = SRGBtoLINEAR(texture( gTextures[lMaterial.mBaseColorTextureID], inUV0)) * lMaterial.mBaseColorFactor;

    vec4  lSampledValue  = texture( gTextures[lMaterial.mMetalnessTextureID], inUV0 );
    aMaterial.mIsMetal   = lSampledValue.r * clamp( lMaterial.mMetallicFactor, 0.0, 1.0 );
    aMaterial.mRoughness = lSampledValue.g * clamp( lMaterial.mRoughnessFactor, c_MinRoughness, 1.0 );

    aMaterial.mOcclusionStrength = lMaterial.mOcclusionStrength;
    aMaterial.mAmbiantOcclusion  = texture( gTextures[lMaterial.mOcclusionTextureID], inUV0 ).r * lMaterial.mOcclusionStrength;
}

void main()
{
    MaterialInputs lMaterialData;
    Material(lMaterialData);

    outPosition = vec4( inWorldPos, 1.0 );
    outNormal = vec4( lMaterialData.mNormal, 1.0 );
    outAlbedo = lMaterialData.mBaseColor;
    outOcclusionMetalRough = vec4( lMaterialData.mAmbiantOcclusion, lMaterialData.mIsMetal, lMaterialData.mRoughness, lMaterialData.mOcclusionStrength );
    outObjectID = 0.0f;
}