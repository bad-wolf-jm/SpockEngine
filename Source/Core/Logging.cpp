#include "Logging.h"

#include <vector>
namespace SE::Logging
{
    static std::ofstream gLogFile;

    void LogToFile( std::string const &aFilePath ) { gLogFile.open( aFilePath ); }

    void LogLine( std::string const &aLine )
    {
        if( gLogFile.is_open() )
            gLogFile << aLine;
    }

    static std::vector<LogMessage> s_LogMessages = {};

    void _Log( LogLevel a_Level, std::string a_Message )
    {
        LogMessage l_NewMessage{};
        l_NewMessage.Level     = a_Level;
        l_NewMessage.Timestamp = 0;
        l_NewMessage.Message   = a_Message;
        s_LogMessages.push_back( l_NewMessage );
    }

    std::vector<LogMessage> &GetLogMessages() { return s_LogMessages; }

    void SetLogOutputFile( fs::path aFilePath ) {}

} // namespace SE::Logging
