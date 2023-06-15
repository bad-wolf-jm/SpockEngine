#include "Logging.h"

#include <vector>
namespace SE::Logging
{
    static std::ofstream gLogFile;

    void LogToFile( string_t const &aFilePath ) { gLogFile.open( aFilePath ); }

    void LogLine( string_t const &aLine )
    {
        if( gLogFile.is_open() )
            gLogFile << aLine;
    }

    static std::vector<LogMessage> sLogMessages = {};

    void _Log( LogLevel a_Level, string_t a_Message )
    {
        LogMessage l_NewMessage{};
        l_NewMessage.Level     = a_Level;
        l_NewMessage.Timestamp = 0;
        l_NewMessage.Message   = a_Message;
        // sLogMessages.push_back( l_NewMessage );
    }

    std::vector<LogMessage> &GetLogMessages() { return sLogMessages; }

    void SetLogOutputFile( path_t aFilePath ) {}

} // namespace SE::Logging
