/// @file   AssetFile.h
///
/// @brief  Read binary asset files
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2022 LeddarTech Inc. All rights reserved.

#pragma once
#include "Core/Math/Types.h"
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>

#include "Core/CUDA/Texture/TextureData.h"
#include "Scene/Importer/ImporterData.h"
#include "Scene/MaterialSystem/MaterialSystem.h"
#include "Scene/VertexData.h"

namespace fs = std::filesystem;

namespace SE::Core
{
    enum class eAssetType : uint32_t
    {
        UNKNOWN        = 0,
        KTX_TEXTURE_2D = 1,
        MESH_DATA      = 2,
        MATERIAL_DATA  = 4,
        ANIMATION_DATA = 5,
        OFFSET_DATA    = 6
    };

    struct sAssetIndex
    {
        eAssetType mType      = eAssetType::UNKNOWN;
        uint32_t   mByteStart = 0;
        uint32_t   mByteEnd   = 0;
    };

    class BinaryAsset
    {
      public:
        BinaryAsset() = default;
        BinaryAsset( fs::path const &aPath );

        ~BinaryAsset();

        /// @brief Retrieve the file's magic value
        static const uint8_t *GetMagic();
        static const uint32_t GetMagicLength();

        /// @brief Retrieve the current file pointer position
        size_t CurrentPosition();

        /// @brief Seek to the given position
        ///
        /// @param aPosition Position to seek to, from the beginning of the file
        void Seek( size_t aPosition );

        /// @brief Check end of file
        bool Eof();

        /// @brief Read an element of type _Ty from the file
        template <typename _Ty>
        _Ty Read()
        {
            _Ty lBuffer;
            mFileStream.read( (char *)&lBuffer, sizeof( _Ty ) );

            return lBuffer;
        }

        template <>
        string_t Read()
        {
            auto lStrLen = Read<uint32_t>();
            auto lBuffer = Read<char>( lStrLen );

            return string_t( lBuffer.data(), lStrLen );
        }

        /// @brief Read `aCount` elements of type _Ty from the file, and return a vector containing them
        template <typename _Ty>
        vector_t<_Ty> Read( size_t aCount )
        {
            vector_t<_Ty> lBuffer( aCount );
            mFileStream.read( (char *)lBuffer.data(), aCount * sizeof( _Ty ) );

            return lBuffer;
        }

        sAssetIndex const &GetIndex( uint32_t aIndex ) const;

        uint32_t CountAssets()
        {
            return mAssetIndex.size();
        }

        void WriteTo( fs::path aPath );

        /// @brief Retrieve the texture stored in the file at index `aIndex`
        std::tuple<TextureData2D, TextureSampler2D> Retrieve( uint32_t aIndex );
        void                                        Retrieve( uint32_t aIndex, TextureData2D &aData, TextureSampler2D &aSampler );
        void Retrieve( uint32_t aIndex, vector_t<VertexData> &aVertexData, vector_t<uint32_t> &aIndexData );
        void Retrieve( uint32_t aIndex, sMaterial &aMaterialData );
        void Retrieve( uint32_t aIndex, sImportedAnimationSampler &aMaterialData );

        void Package( Core::TextureData2D const &aData, sTextureSamplingInfo const &aSampler );
        void Package( vector_t<VertexData> const &aVertexData, vector_t<uint32_t> const &aIndexData );
        void Package( sMaterial const &aMaterialData );
        void Package( sImportedAnimationSampler const &aMaterialData );
        void Package( ref_t<TextureData2D> aData, ref_t<TextureSampler2D> aSampler );
        void Package( sImportedTexture const &aData );

      private:
        fs::path      mFilePath   = "";
        bool          mFileExists = false;
        std::ifstream mFileStream{};
        size_t        mFileSize = 0;

        uint32_t                       mAssetCount = 0;
        vector_t<sAssetIndex>       mAssetIndex{};
        vector_t<vector_t<char>> mPackets{};
        uint32_t                       mTotalPacketSize = 0;
    };

} // namespace SE::Core