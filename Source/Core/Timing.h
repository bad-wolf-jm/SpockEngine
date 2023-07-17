#pragma once

namespace SE::Core
{
    class CEveryNTime
    {
      public:
        TIMETYPE mPrevTrigger; ///< Timestamp of the last time the class was "ready"
        TIMETYPE mPeriod;      ///< Timing interval to check

        /// Default constructor
        CEveryNTime()
        {
            reset();
            mPeriod = 1;
        };
        /// Constructor
        /// @param period the time interval between triggers
        CEveryNTime( TIMETYPE period )
        {
            reset();
            setPeriod( period );
        };

        /// Set the time interval between triggers
        void setPeriod( TIMETYPE period ) { mPeriod = period; };

        /// Get the current time according to the class' timekeeper
        TIMETYPE getTime() { return (TIMETYPE)( TIMEGETTER() ); };

        /// Get the time interval between triggers
        TIMETYPE getPeriod() { return mPeriod; };

        /// Get the time elapsed since the last trigger event
        TIMETYPE getElapsed() { return getTime() - mPrevTrigger; }

        /// Get the time until the next trigger event
        TIMETYPE getRemaining() { return mPeriod - getElapsed(); }

        /// Get the timestamp of the most recent trigger event
        TIMETYPE getLastTriggerTime() { return mPrevTrigger; }

        /// Check if the time interval has elapsed
        bool ready()
        {
            bool isReady = ( getElapsed() >= mPeriod );
            if( isReady )
            {
                reset();
            }
            return isReady;
        }

        /// Reset the timestamp to the current time
        void reset() { mPrevTrigger = getTime(); };

        /// Reset the timestamp so it is ready() on next call
        void trigger() { mPrevTrigger = getTime() - mPeriod; };

        /// @copydoc ready()
        operator bool() { return ready(); }
    };

#define CONCAT_HELPER( x, y ) x##y
#define CONCAT_MACRO( x, y )  CONCAT_HELPER( x, y )

#define EVERY_N_MILLISECONDS( N ) EVERY_N_MILLIS_I( CONCAT_MACRO( PER, __COUNTER__ ), N )

#define EVERY_N_MILLIS_I( NAME, N )      \
    static EveryNMilliseconds NAME( N ); \
    if( NAME )

} // namespace SE::Core