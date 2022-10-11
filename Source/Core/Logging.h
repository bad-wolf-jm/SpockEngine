/// @file   Logging.h
///
/// @brief  Simple logging functions
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2021 LeddarTech Inc. All rights reserved.

#pragma once

#include <fmt/core.h>
#include <fstream>
#include <vector>
#include <filesystem>



/**
 * \namespace LTSE::Logging
 *
 * @brief Basic logging functions.
 */
namespace LTSE::Logging
{
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
        std::string Message;
    };

    void _Log( LogLevel a_Level, std::string a_Message );
    std::vector<LogMessage> &GetLogMessages();

    /** @brief Information level log entry.
     *
     * @param aString  Template string
     * @param aArgList Arguments to be passed to the templated string.
     */
    template <typename... ArgTypes> void Info( std::string aString, ArgTypes &&...aArgList )
    {
        std::string s = fmt::format( aString, std::forward<ArgTypes>( aArgList )... );
        fmt::print( "[ INFO ] {}\n", s );
        LogLine(fmt::format( "[ INFO ] {}\n", s ));
    }

    /** @brief Error level log entry.
     *
     * @param aString  Template string
     * @param aArgList Arguments to be passed to the templated string.
     */
    template <typename... ArgTypes> void Error( std::string aString, ArgTypes &&...aArgList )
    {
        std::string s = fmt::format( aString, std::forward<ArgTypes>( aArgList )... );
        _Log(LogLevel::ERR, s);
        fmt::print( "[ ERROR ] {}\n", s );
        LogLine(fmt::format( "[ ERROR ] {}\n", s ));
    }

    void LogToFile(std::string const &aFilePath);
    void LogLine(std::string const &aLine);
    void SetLogOutputFile( fs::path aFilePath );

} // namespace LTSE::Logging
