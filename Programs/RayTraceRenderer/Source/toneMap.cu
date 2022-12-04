#include "Core/GPUResource/CudaAssert.h"

#include "SampleRenderer.h"

using namespace osc;

namespace osc
{

    inline SE_CUDA_DEVICE_FUNCTION_DEF float4 sqrt( float4 f )
    {
        return make_float4( sqrtf( f.x ), sqrtf( f.y ), sqrtf( f.z ), sqrtf( f.w ) );
    }
    inline SE_CUDA_DEVICE_FUNCTION_DEF float  clampf( float f ) { return min( 1.f, max( 0.f, f ) ); }
    inline SE_CUDA_DEVICE_FUNCTION_DEF float4 clamp( float4 f )
    {
        return make_float4( clampf( f.x ), clampf( f.y ), clampf( f.z ), clampf( f.w ) );
    }

    /*! runs a cuda kernel that performs gamma correction and float4-to-rgba conversion */
    CUDA_KERNEL_DEFINITION void computeFinalPixelColorsKernel( uint32_t *finalColorBuffer, float4 *denoisedBuffer, math::ivec2 size )
    {
        int pixelX = threadIdx.x + blockIdx.x * blockDim.x;
        int pixelY = threadIdx.y + blockIdx.y * blockDim.y;
        if( pixelX >= size.x ) return;
        if( pixelY >= size.y ) return;

        int pixelID = pixelX + size.x * pixelY;

        float4 f4     = denoisedBuffer[pixelID];
        f4            = clamp( sqrt( f4 ) );
        uint32_t rgba = 0;
        rgba |= (uint32_t)( f4.x * 255.9f ) << 0;
        rgba |= (uint32_t)( f4.y * 255.9f ) << 8;
        rgba |= (uint32_t)( f4.z * 255.9f ) << 16;
        rgba |= (uint32_t)255 << 24;
        finalColorBuffer[pixelID] = rgba;
    }

    void SampleRenderer::computeFinalPixelColors()
    {
        math::ivec2 fbSize = launchParams.mFrame.mSize;
        math::ivec2 blockSize( 32 );
        math::ivec2 numBlocks = divRoundUp( fbSize, blockSize );
        computeFinalPixelColorsKernel<<<dim3( numBlocks.x, numBlocks.y ), dim3( blockSize.x, blockSize.y )>>>(
            (uint32_t *)finalColorBuffer.RawDevicePtr(), (float4 *)denoisedBuffer.RawDevicePtr(), fbSize );
    }

} // namespace osc
