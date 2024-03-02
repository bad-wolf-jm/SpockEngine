/// @file   Logging.h
///
/// @brief  Simple logging functions
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2023 Jean-Martin Albert All rights reserved.

#pragma once

#include "Definitions.h"
#include <filesystem>
#include <fmt/core.h>
#include <fstream>
#include <vector>

/**
 * \namespace SE::Logging
 *
 * @brief Basic logging functions.
 */
namespace SE::Logging
{
    using namespace SE::Core;
    namespace fs = std::filesystem;
    enum class LogLevel : uint8_t
    {
        DEBUG   = 1,
        INFO    = 2,
        WARNING = 3,
        ERR     = 4
    };

    struct LogMessage
    {
        LogLevel Level;
        uint64_t Timestamp;
        string_t Message;
    };

    void                  _Log( LogLevel a_Level, string_t a_Message );
    vector_t<LogMessage> &GetLogMessages();

    /** @brief Information level log entry.
     *
     * @param aString  Template string
     * @param aArgList Arguments to be passed to the templated string.
     */
    template <typename... ArgTypes>
    void Info( string_t aString, ArgTypes &&...aArgList )
    {
        // string_t s = fmt::format( aString, std::forward<ArgTypes>( aArgList )... );
        // _Log( LogLevel::INFO, s );
        // // fmt::print( "[ INFO ] {}\n", s );
        // LogLine( fmt::format( "[ INFO ] {}\n", s ) );
    }

    /** @brief Error level log entry.
     *
     * @param aString  Template string
     * @param aArgList Arguments to be passed to the templated string.
     */
    template <typename... ArgTypes>
    void Error( string_t aString, ArgTypes &&...aArgList )
    {
        // string_t s = fmt::format( aString, std::forward<ArgTypes>( aArgList )... );
        // _Log( LogLevel::ERR, s );
        // // fmt::print( "[ ERROR ] {}\n", s );
        // LogLine( fmt::format( "[ ERROR ] {}\n", s ) );
    }

    void LogToFile( string_t const &aFilePath );
    void LogLine( string_t const &aLine );
    void SetLogOutputFile( fs::path aFilePath );

} // namespace SE::Logging
