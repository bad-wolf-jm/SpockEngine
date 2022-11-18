#pragma once

/// @file   BlockTimer.h
///
/// @brief  simple profiling tool for functions and scopes
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2022 LeddarTech Inc. All rights reserved.

#include "Core/Logging.h"
#include "Core/Memory.h"

#include <chrono>
#include <iostream>
#include <mutex>
#include <thread>

namespace SE::Core
{

    /// @brief sProfilingEvent
    ///
    /// Represents a single event fired during profiling
    ///
    struct sProfilingEvent
    {
        std::string mName = ""; //!< Name of the event

        int64_t mStart       = 0; //!< Start time
        int64_t mElapsedTime = 0; //!< Total event time

        std::thread::id mThreadID; //!< Thread ID for the event
    };

    /// @brief sProfilingSession
    ///
    /// Represents a profiling session, i.e. a sequence of profiling events
    ///
    struct sProfilingSession
    {
        std::string mName = ""; //!< Name of the session

        std::vector<sProfilingEvent> mEvents = {}; //!< List of events fired during the session

        sProfilingSession( std::string const &aName )
            : mName{ aName }
        {
        }
    };

    /// @brief Instrumentation class
    ///
    /// Single-instance class which keeps track of the beginning and ending of profiling events.
    ///
    class Instrumentor
    {
      public:
        Instrumentor( const Instrumentor & ) = delete;
        Instrumentor( Instrumentor && )      = delete;

        void BeginSession( const std::string &aName )
        {
            std::lock_guard lLock( mMutex );

            if( mCurrentSession )
                InternalEndSession();

            mCurrentSession = New<sProfilingSession>( aName );
        }

        Ref<sProfilingSession> EndSession()
        {
            std::lock_guard lLock( mMutex );

            auto lResults = mCurrentSession;
            InternalEndSession();

            return lResults;
        }

        void WriteProfile( const sProfilingEvent &aResult )
        {
            std::lock_guard lLock( mMutex );
            if( mCurrentSession )
                mCurrentSession->mEvents.push_back( aResult );
        }

        static Instrumentor &Get()
        {
            static Instrumentor lInstance;
            return lInstance;
        }

      private:
        Instrumentor()
            : mCurrentSession( nullptr )
        {
        }

        ~Instrumentor() { EndSession(); }

        void InternalEndSession() { mCurrentSession = nullptr; }

      private:
        std::mutex mMutex{};
        Ref<sProfilingSession> mCurrentSession = nullptr;
    };

    /// @brief BlockTimer
    ///
    /// Automatically measure the execution time of a scope. Upon destruction, write the execution
    /// time to the current profiling session, if there is an active one
    ///
    class BlockTimer
    {
      public:
        BlockTimer( std::string aName )
        {
            mName           = aName;
            mStartTimePoint = std::chrono::high_resolution_clock::now();
        }

        ~BlockTimer()
        {
            auto lEndTimePoint = std::chrono::high_resolution_clock::now();

            auto lStartTime = std::chrono::time_point_cast<std::chrono::microseconds>( mStartTimePoint ).time_since_epoch().count();
            auto lEndTime   = std::chrono::time_point_cast<std::chrono::microseconds>( lEndTimePoint ).time_since_epoch().count();

            Instrumentor::Get().WriteProfile( { mName, lStartTime, ( lEndTime - lStartTime ), std::this_thread::get_id() } );
        }

      private:
        std::string mName = "";
        std::chrono::time_point<std::chrono::high_resolution_clock> mStartTimePoint;
    };

#define LTSE_ENABLE_PROFILING 1
#if LTSE_ENABLE_PROFILING
#    if( defined( __FUNCSIG__ ) || ( _MSC_VER ) )
#        define LTSE_FUNC_SIG __func__
#    elif( defined( __INTEL_COMPILER ) && ( __INTEL_COMPILER >= 600 ) ) || ( defined( __IBMCPP__ ) && ( __IBMCPP__ >= 500 ) )
#        define LTSE_FUNC_SIG __FUNCTION__
#    elif defined( __STDC_VERSION__ ) && ( __STDC_VERSION__ >= 199901 )
#        define LTSE_FUNC_SIG __func__
#    elif defined( __cplusplus ) && ( __cplusplus >= 201103 )
#        define LTSE_FUNC_SIG __func__
#    else
#        define LTSE_FUNC_SIG "LTSE_FUNC_SIG unknown!"
#    endif
#    define LTSE_PROFILE_SCOPE_LINE2( name, line ) BlockTimer timer##line( name )
#    define LTSE_PROFILE_SCOPE_LINE( name, line ) LTSE_PROFILE_SCOPE_LINE2( name, line )
#    define LTSE_PROFILE_SCOPE( name ) LTSE_PROFILE_SCOPE_LINE( name, __LINE__ )
#    define LTSE_PROFILE_FUNCTION() LTSE_PROFILE_SCOPE( LTSE_FUNC_SIG )
#else
#    define LTSE_PROFILE_SCOPE( name )
#    define LTSE_PROFILE_FUNCTION()
#endif

} // namespace SE::Core