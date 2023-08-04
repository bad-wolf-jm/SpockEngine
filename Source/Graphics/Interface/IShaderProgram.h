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
        IShaderProgram( Ref<IGraphicContext> aGraphicContext, eShaderStageTypeFlags aShaderType, int aVersion,
                        std::string const &aName );
        IShaderProgram( Ref<IGraphicContext> aGraphicContext, eShaderStageTypeFlags aShaderType, int aVersion,
                        std::string const &aName, fs::path const &aCacheRoot );

        /** @brief */
        ~IShaderProgram() = default;

        template <typename _GCSubtype>
        Ref<_GCSubtype> GraphicContext()
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

        void AddCode( std::string const &aCode );
        void AddCode( vector_t<uint8_t> const &aCode );
        void AddFile( fs::path const &aPath );

        template <typename _Ty>
        void Define( std::string const &aConstant, _Ty const &aValue )
        {
            AddCode( fmt::format( "#define {} {}", aConstant, aValue ) );
        }

        std::string  Program();
        void         Compile();
        virtual void DoCompile()    = 0;
        virtual void BuildProgram() = 0;
        std::string  Hash();
        size_t       HashNum();

      protected:
        Ref<IGraphicContext> mGraphicContext = nullptr;

        vector_t<std::string> mCodeBlocks;
        vector_t<uint32_t>    mCompiledByteCode;

        std::string mProgram{};

        int                   mVersion;
        std::string           mName;
        std::string           mCacheFileName;
        fs::path              mCacheRoot;
        eShaderStageTypeFlags mShaderType;
    };
} // namespace SE::Graphics
