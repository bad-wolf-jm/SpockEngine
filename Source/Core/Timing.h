#pragma once

#include <chrono>

namespace SE::Core
{
    class EveryNMilliseconds
    {
      public:
        EveryNMilliseconds()
            : EveryNMilliseconds( 1 )
        {
        }

        EveryNMilliseconds( int64_t period )
            : mPeriod{ period }
        {
            Reset();
        };

        int64_t Time()
        {
            auto now    = std::chrono::system_clock::now();
            auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>( now );

            return now_ms.time_since_epoch().count();
        };

        int64_t Elapsed() { return Time() - mPrevTrigger; }

        /// Check if the time interval has elapsed
        bool IsReady()
        {
            bool lIsReady = ( Elapsed() >= mPeriod );
            if( lIsReady ) Reset();

            return lIsReady;
        }

        void Reset() { mPrevTrigger = Time(); };

        operator bool() { return IsReady(); }

      private:
        int64_t mPrevTrigger = 0;
        int64_t mPeriod      = 0;
    };

#define CONCAT_HELPER( x, y ) x##y
#define CONCAT_MACRO( x, y )  CONCAT_HELPER( x, y )

#define EVERY_N_MILLISECONDS( N ) EVERY_N_MILLIS_I( CONCAT_MACRO( _TIMER_, __COUNTER__ ), N )

#define EVERY_N_MILLIS_I( NAME, N )      \
    static EveryNMilliseconds NAME( N ); \
    if( NAME )

EVERY_N_MILLISECONDS(23)

} // namespace SE::Core