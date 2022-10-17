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

#include "TextureData.h"

namespace fs = std::filesystem;

namespace LTSE::Core
{

    enum class eAssetType : uint32_t
    {
        UNKNOWN        = 0,
        KTX_TEXTURE_2D = 1,
        VERTEX_BUFFER  = 2,
        INDEX_BUFFER   = 3,
        MATERIAL_DATA  = 4,
        ANIMATION_DATA = 5
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

        /// @brief Read `aCount` elements of type _Ty from the file, and return a vector containing them
        template <typename _Ty>
        std::vector<_Ty> Read( size_t aCount )
        {
            std::vector<_Ty> lBuffer( aCount );
            mFileStream.read( (char *)lBuffer.data(), aCount * sizeof( _Ty ) );

            return lBuffer;
        }

        /// @brief Retrieve the texture stored in the file at index `aIndex`
        std::tuple<TextureData2D, TextureSampler2D> Retrieve( uint32_t aIndex );

      private:
        fs::path      mFilePath   = "";
        bool          mFileExists = false;
        std::ifstream mFileStream{};
        size_t        mFileSize = 0;

        uint32_t                 mAssetCount = 0;
        std::vector<sAssetIndex> mAssetIndex{};
    };

} // namespace LTSE::Core