#pragma once

#include "Core/Memory.h"
#include "Core/Logging.h"

#include <chrono>
#include <iostream>
#include <mutex>
#include <thread>

namespace LTSE::Core
{

    struct ProfileResult
    {
        std::string Name;

        int64_t Start;
        int64_t ElapsedTime;
        std::thread::id ThreadID;
    };

    struct InstrumentationSession
    {
        std::string Name;
        std::vector<ProfileResult> mEvents;

        InstrumentationSession( std::string const &aName )
            : Name{ aName }
        {
        }
    };

    class Instrumentor
    {
      public:
        Instrumentor( const Instrumentor & ) = delete;
        Instrumentor( Instrumentor && )      = delete;

        void BeginSession( const std::string &name )
        {
            std::lock_guard lock( m_Mutex );

            if( m_CurrentSession )
            {
                InternalEndSession();
            }

            m_CurrentSession = New<InstrumentationSession>( name );
            LTSE::Logging::Info("BEGIN_PROFILING_SESSION");
        }

        Ref<InstrumentationSession> EndSession()
        {
            std::lock_guard lock( m_Mutex );

            auto lResults = m_CurrentSession;
            InternalEndSession();

            LTSE::Logging::Info("END_PROFILING_SESSION");
            return lResults;
        }

        void WriteProfile( const ProfileResult &result )
        {
            std::lock_guard lock( m_Mutex );
            if( m_CurrentSession )
            {
                m_CurrentSession->mEvents.push_back( result );
            }
        }

        static Instrumentor &Get()
        {
            static Instrumentor instance;
            return instance;
        }

      private:
        Instrumentor()
            : m_CurrentSession( nullptr )
        {
        }

        ~Instrumentor() { EndSession(); }

        void InternalEndSession() { m_CurrentSession = nullptr; }

      private:
        std::mutex m_Mutex;
        Ref<InstrumentationSession> m_CurrentSession;
    };

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

#define LTSE_PROFILE 1
#if LTSE_PROFILE
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

} // namespace LTSE::Core