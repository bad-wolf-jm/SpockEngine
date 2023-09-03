#pragma once

#include <memory>

#include "Core/Memory.h"
#include "Core/Types.h"

#include "Enums.h"
#include "IGraphicContext.h"

namespace SE::Graphics
{
    using namespace SE::Core;

    /** @brief */
    class IShaderProgram
    {
      public:
        /** @brief */
        IShaderProgram( ref_t<IGraphicContext> aGraphicContext, eShaderStageTypeFlags aShaderType, int aVersion,
                        string_t const &aName );
        IShaderProgram( ref_t<IGraphicContext> aGraphicContext, eShaderStageTypeFlags aShaderType, int aVersion,
                        string_t const &aName, fs::path const &aCacheRoot );

        /** @brief */
        ~IShaderProgram() = default;

        template <typename _GCSubtype>
        ref_t<_GCSubtype> GraphicContext()
        {
            return std::reinterpret_pointer_cast<_GCSubtype>( mGraphicContext );
        }

        int Version()
        {
            return mVersion;
        }
        void SetVersion( int aVersion )
        {
            mVersion = aVersion;
        }

        void AddCode( string_t const &aCode );
        void AddCode( vector_t<uint8_t> const &aCode );
        void AddFile( fs::path const &aPath );

        template <typename _Ty>
        void Define( string_t const &aConstant, _Ty const &aValue )
        {
            AddCode( fmt::format( "#define {} {}", aConstant, aValue ) );
        }

        string_t  Program();
        void         Compile();
        virtual void DoCompile()    = 0;
        virtual void BuildProgram() = 0;
        string_t  Hash();
        size_t       HashNum();

      protected:
        ref_t<IGraphicContext> mGraphicContext = nullptr;

        vector_t<string_t> mCodeBlocks;
        vector_t<uint32_t>    mCompiledByteCode;

        string_t mProgram{};

        int                   mVersion;
        string_t           mName;
        string_t           mCacheFileName;
        fs::path              mCacheRoot;
        eShaderStageTypeFlags mShaderType;
    };
} // namespace SE::Graphics
