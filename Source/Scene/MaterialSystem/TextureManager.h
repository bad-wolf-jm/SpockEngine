#pragma once

#include "Core/Math/Types.h"

#ifndef __CUDACC__
#    include "Graphics/API.h"
#endif

#include "Core/CUDA/Array/CudaBuffer.h"

using namespace math::literals;

#ifndef __CUDACC__
using namespace SE::Graphics;
namespace fs = std::filesystem;
#endif

namespace SE::Core
{
    class TextureManager
    {
      public:
        TextureManager()  = default;
        ~TextureManager() = default;

        TextureManager( Ref<IGraphicContext> aGraphicContext );

        Ref<ISampler2D> GetTextureByID( uint64_t aID );

        uint64_t CreateTexture( fs::path aFilePath, sTextureSamplingInfo aSamplingInfo );
        uint64_t CreateTexture( Ref<TextureData2D> aTexture, Ref<TextureSampler2D> aTextureSampler );
        uint64_t CreateTexture( TextureData2D &aTexture, TextureSampler2D &aTextureSampler );

        void UpdateDescriptors();

        Ref<IDescriptorSet> GetDescriptorSet() { return mTextureDescriptorSet; }

        void Clear();

        std::vector<Ref<ISampler2D>> const &GetTextures() const { return mTextureSamplers; }

        Cuda::GPUMemory const &GetCudaTextures() const { return mCudaTextureBuffer; }

      private:
        Ref<IGraphicContext>      mGraphicContext          = nullptr;
        Ref<IDescriptorSet>       mTextureDescriptorSet    = nullptr;
        Ref<IDescriptorSetLayout> mTextureDescriptorLayout = nullptr;
        Cuda::GPUMemory           mCudaTextureBuffer{};

        std::vector<Ref<ISampler2D>> mTextureSamplers = {};

        bool mDirty = false;
    };
} // namespace SE::Core