#pragma once

#include <memory>

#include "Core/Memory.h"
#include "Core/Types.h"

#include "IGraphicContext.h"
#include "Enums.h"

namespace SE::Graphics
{
    using namespace SE::Core;

    /** @brief */
    class IShaderProgram
    {
      public:
        /** @brief */
        IShaderProgram( Ref<IGraphicContext> aGraphicContext, eShaderStageTypeFlags aShaderType, int aVersion = 460 );

        /** @brief */
        ~IShaderProgram() = default;

        template <typename _GCSubtype>
        Ref<_GCSubtype> GraphicContext()
        {
            return std::reinterpret_pointer_cast<_GCSubtype>( mGraphicContext );
        }

        int  Version() { return mVersion; }
        void SetVersion( int aVersion ) { mVersion = aVersion; }

        void AddCode( std::string const &aCode );
        void AddFile( fs::path const &aPath );

        template <typename _Ty>
        void Define( std::string const &aConstant, _Ty const &aValue )
        {
            AddCode( fmt::format( "#define {} {}", aConstant, aValue ) );
        }

        std::string  Program();
        virtual void Compile() = 0;
        std::string  Hash();

      protected:
        Ref<IGraphicContext> mGraphicContext = nullptr;

        std::vector<std::string> mCodeBlocks;
        std::vector<uint32_t>    mCompiledByteCode;

        std::string mProgram{};

        int                   mVersion;
        eShaderStageTypeFlags mShaderType;
    };
} // namespace SE::Graphics
