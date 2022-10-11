#pragma once

#include <functional>
#include <optional>
#include <thread>

namespace LTSE::Core
{
    template <typename _RetType> class Promise
    {
      public:
        Promise( std::function<_RetType()> aAsyncFunction );
        ~Promise();

        void WhenFullfiled( std::function<void( _RetType )> aResolution )
        {
            if (mPromiseResolved && mPromiseValue.has_value())
            {
                aResolution(mPromiseValue.value());
            }
            else
            {
                mResolution = aResolution;
            }
        }

        void Run()
        {

        }

      private:
        bool mPromiseResolved = false;
        std::optional<_RetType> mPromiseValue{};
        std::function<void( _RetType )> mResolution;
    };
} // namespace LTSE::Core